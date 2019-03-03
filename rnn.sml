signature RNN =
sig
    type mlvector
    type matrix
    type rnn
    type rnnrecord
    val init : ((int * int) -> real) -> rnn
    val forward : rnn -> matrix -> (matrix * rnnrecord list)
    val backward : rnn -> rnnrecord list -> matrix -> matrix -> real -> unit
end

functor Rnn (structure H: HYPERPARAM
             structure M: MLMATRIX
             structure N: NONLINEAR) : RNN =
struct
type matrix = M.mlmatrix
type mlvector = M.mlvector
structure Mlv = M.Mlv

type rnn =
     {w_i: matrix, u_i : matrix, w_c: matrix, u_c : matrix, w_f: matrix, u_f: matrix, w_o : matrix, u_o : matrix, w_out: matrix}
type rnnrecord =
     {c_t_pre : matrix, h_t_pre: matrix, a_t: matrix, i_t: matrix, f_t: matrix, o_t: matrix, c_t: matrix, h_t: matrix, y_t_0 : matrix, y_t: matrix}

val innode = H.innode
val hidenode = H.hidenode
val outnode = H.outnode

fun init_matrix f (h, w) =
    let
        fun make_mlv i = Mlv.mlvector (fn j => f (i, j)) w
    in
        M.matrix make_mlv h
    end

fun init f =
    let
        val w_i = init_matrix f (hidenode, innode)
        val u_i = init_matrix f (hidenode, hidenode)
        val w_f = init_matrix f (hidenode, innode)
        val u_f = init_matrix f (hidenode, hidenode)
        val w_o = init_matrix f (hidenode, innode)
        val u_o = init_matrix f (hidenode, hidenode)
        val w_c = init_matrix f (hidenode, innode)
        val u_c = init_matrix f (hidenode, hidenode)
        val w_out = init_matrix f (outnode, hidenode)
        (* val _ = print (M.toString w_i) *)
    in
        {w_i = w_i, u_i = u_i, w_c = w_c, u_c = u_c,
         w_f = w_f, u_f = u_f, w_o = w_o, u_o = u_o,
         w_out = w_out}
    end

fun forward rnn input =
    case rnn of {w_i, u_i, w_c, u_c, w_f, u_f, w_o, u_o, w_out} =>
    let
        fun time_t (record, x_t) =
            let
                val _ = print "==== once begin ====\n"
                val x_t = M.transpose x_t
                val (c_t_pre, h_t_pre) =
                    case record of
                        [] => ((M.make (hidenode, 1, 0.0)), (M.make (hidenode, 1, 0.0)))
                      | {c_t = c_t_pre, h_t = h_t_pre, ...} :: _ => (c_t_pre, h_t_pre)
                fun cal_flow w u nonlinear =
                    M.map nonlinear (M.add (M.mul w x_t) (M.mul u h_t_pre))
                val a_t = cal_flow w_c u_c N.tanh
                val i_t = cal_flow w_i u_i N.sigmoid
                val f_t = cal_flow w_f u_f N.sigmoid
                val o_t = cal_flow w_o u_o N.sigmoid
                val c_t = (M.add (M.elemwise i_t a_t) (M.elemwise f_t c_t_pre))
                val h_t = M.elemwise o_t (M.map N.tanh c_t)
                val y_t_0 = M.mul w_out h_t
                val _ = print (M.toString y_t_0)
                val y_t = M.map N.sigmoid y_t_0
                val _ = print "==== once end ====\n"
                val his = {c_t_pre, h_t_pre, a_t, i_t, f_t, o_t, c_t, h_t, y_t_0, y_t}
            in
                his :: record
            end
        val record_l = M.foldrowl time_t [] input
        val y = List.map (fn record =>
                             case record of
                                 {y_t = y_t, ...}
                                 =>
                             M.squeeze2 y_t) record_l
        val y = Array.fromList y
    in
        (y, record_l)
    end

fun backward rnn his_l input ans alpha =
    case rnn of {w_i, u_i, w_c, u_c, w_f, u_f, w_o, u_o, w_out} =>
    let
        fun time_t (backrecord, i, record) =
            case record of
                {c_t_pre, h_t_pre, a_t, i_t, f_t, o_t, c_t, h_t, y_t_0, y_t}
                =>
                  let
                      val y_ans_t = M.row ans i
                      val x_t = M.row input i
                      val (perr_pi_post, perr_pf_post, perr_po_post,
                           perr_pa_post, perr_pc_post, f_t_post) =
                          case backrecord of
                              NONE => (M.make (hidenode, 1, 0.0), M.make (hidenode, 1, 0.0),
                                       M.make (hidenode, 1, 0.0), M.make (hidenode, 1, 0.0),
                                       M.make (hidenode, 1, 0.0), M.make (hidenode, 1, 0.0))
                            | SOME {perr_pi_post, perr_pf_post, perr_po_post,
                               perr_pa_post, perr_pc_post, f_t_post} =>
                              (perr_pi_post, perr_pf_post, perr_po_post,
                               perr_pa_post, perr_pc_post, f_t_post)
                      val y_t = M.squeeze12 y_t
                      val y_ans_t = M.squeeze12 y_ans_t
                      val perr_py = (y_t - y_ans_t) * (N.dsigmoid y_t)
                      val perr_py = M.make (1, 1, perr_py)
                      val perr_ph = List.foldl (fn ((p, w), b) => M.add (M.mul p w) b)
                                                (M.mul perr_py w_out)
                                                [(perr_pi_post, u_i),
                                                 (perr_pf_post, u_f),
                                                 (perr_po_post, u_o),
                                                 (perr_pa_post, u_c)]
                      val perr_po = List.foldl (fn (a ,b) => M.elemwise a b)
                                                perr_ph
                                                [(M.map N.dtanh c_t), (M.map N.dsigmoid o_t)]
                      val perr_pc = M.add
                          (M.elemwise (M.elemwise perr_ph o_t) (M.map N.dtanh c_t))
                          (M.elemwise perr_pc_post f_t_post)
                      val perr_pf = M.elemwise (M.elemwise perr_pc c_t_pre) (M.map N.dsigmoid f_t)
                      val perr_pi = M.elemwise (M.elemwise perr_pc a_t) (M.map N.dsigmoid i_t)
                      val perr_pa = M.elemwise (M.elemwise perr_pc i_t) (M.map N.dsigmoid a_t)
                      val _ = M.add_modify u_i (M.map (fn a => a * alpha) (M.mul h_t_pre (M.transpose perr_pi)))
                      val _ = M.add_modify u_f (M.map (fn a => a * alpha) (M.mul h_t_pre (M.transpose perr_pf)))
                      val _ = M.add_modify u_o (M.map (fn a => a * alpha) (M.mul h_t_pre (M.transpose perr_po)))
                      val _ = M.add_modify u_c (M.map (fn a => a * alpha) (M.mul h_t_pre (M.transpose perr_pa)))
                      val _ = M.add_modify w_i (M.map (fn a => a * alpha) (M.mul x_t (M.transpose perr_pi)))
                      val _ = M.add_modify w_f (M.map (fn a => a * alpha) (M.mul x_t (M.transpose perr_pf)))
                      val _ = M.add_modify w_o (M.map (fn a => a * alpha) (M.mul x_t (M.transpose perr_po)))
                      val _ = M.add_modify w_c (M.map (fn a => a * alpha) (M.mul x_t (M.transpose perr_pa)))
                  in
                      SOME {perr_pi_post = perr_pi, perr_pf_post = perr_pf, perr_po_post = perr_po,
                            perr_pa_post = perr_pa, perr_pc_post = perr_pc, f_t_post = f_t}
                  end
        val _ = list_foldli time_t NONE his_l
    in
        ()
    end


(* fun backword rnn history_l input output ans alpha = *)
(*     let *)
(*         fun cal_err (y_t, y_ans_t) = (mysq (y_t - y_ans_t)) / 2.0 *)
(*         val err = Mlv.map2 cal_err output ans *)
(*         fun cal_err' (y_t, y_ans_t) = (y_t - y_ans_t) * (N.dsigmoid y_t) *)
(*         val err' = Mlv.map2 cal_err' output ans *)
(*         fun backword_time (history_t, x_t, err_t) = *)
(*             case history of *)
(*                 {} => *)
(*                 let *)
(*                     val _ = *)
(*                         M.modifyi (fn (_, j, e) => *)
(*                                       e + alpha * err_t * (Mlv.sub h_1_t j)) w_out *)
(*                     val h_1_delta = List.foldl (fn (e, r) = Mlv.add e r) *)
(*                                              (M.matmulvect w_out y_delta) *)
(*                                              [(M.matmulvect u_i i_0_delta'), *)
(*                                               (M.matmulvect u_f f_0_delta'), *)
(*                                               (M.matmulvect u_o o_0_delta'), *)
(*                                               (M.matmulvect u_c g_0_delta')] *)
(*                     val o_0_delta = Mlv.map3 (fn (h, s, o) => *)
(*                                            h * (N.tanh s) * (dsigmoid o)) *)
(*                                            h_1_delta s_1_t o_0_t *)
(*                     val s_1_delta = Mlv.map5 (fn (h, o, s, s', f') => *)
(*                                                  h * o * (N.dtanh s) * s' * f') *)
(*                                              h_1_delta o_0_t s_1_t s_1_delta' f_0_delta' *)
(*                     val f_0_delta = Mlv.map3 (fn (s_delta, s_0, f) => *)
(*                                                  s_delta * s_0 * (dsigmoid f)) *)
(*                                              s_1_delta s_0_t f_0_t *)
(*                     val i_0_delta = Mlv.map3 (fn (s_delta, g, i) => *)
(*                                                  s_delta * g * (dsigmoid i)) *)
(*                                              s_1_delta g_0_t i_0_t *)
(*                     val g_0_delta = Mlv.map3 (fn (s_delta, i, g) => *)
(*                                                  s_delta * i * (dsigmoid g)) *)
(*                                              s_1_delta i_0_t g_0_t *)
(*                     fun updat_u (u, u_delta) = *)
(*                         M.modifyi (fn (i, j, e) => *)
(*                                       e + alpha * (Mlv.sub u_delta j) * (Mlv.sub h_0_t i)) *)
(*                                   u *)
(*                     val _ = List.foldl (fn (e, _) => update_u e) () *)
(*                                        [(u_i, i_0_delta), *)
(*                                         (u_f, f_0_delta), *)
(*                                         (u_o, o_0_delta), *)
(*                                         (u_c, g_0_delta)] *)
(*                     fun updat_w (w, w_delta) = *)
(*                         M.modifyi (fn (i, j, e) => *)
(*                                       e + alpha * (Mlv.sub w_delta j) * (Mlv.sub x_t i)) *)
(*                                   w *)
(*                     val _ = List.foldl (fn (e, _) => update_w e) () *)
(*                                        [(w_i, i_0_delta), *)
(*                                         (w_f, f_0_delta), *)
(*                                         (w_o, o_0_delta), *)
(*                                         (w_c, g_0_delta)] *)
(*                 in *)

(*                 end *)
(*     in *)
(*     end *)
end
