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
                val x_t = M.transpose x_t
                val (c_t_pre, h_t_pre, y_t_0_pre) =
                    case record of
                        [] => ((M.make (hidenode, 1, 0.0)), (M.make (hidenode, 1, 0.0)),
                               (M.make (outnode, 1, 0.0)))
                      | {c_t = c_t_pre, h_t = h_t_pre, y_t_0 = y_t_0_pre, ...} :: _ =>
                        (c_t_pre, h_t_pre, y_t_0_pre)
                fun cal_flow w u nonlinear =
                    M.map_inplace nonlinear (M.add_inplace (M.mul w x_t) (M.mul u h_t_pre))
                val a_t = cal_flow w_c u_c N.sigmoid
                val i_t = cal_flow w_i u_i N.sigmoid
                val f_t = cal_flow w_f u_f N.sigmoid
                val o_t = cal_flow w_o u_o N.sigmoid
                val c_t = (M.add_inplace (M.elemwise i_t a_t) (M.elemwise f_t c_t_pre))
                val h_t = M.elemwise o_t (M.map N.tanh c_t)
                val y_t_0 = M.mul w_out h_t
                val y_t = M.map N.sigmoid y_t_0
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
        val y = Array.fromList (List.rev y)
    in
        (y, record_l)
    end

fun backward rnn his_l input ans alpha =
    case rnn of {w_i, u_i, w_c, u_c, w_f, u_f, w_o, u_o, w_out} =>
    let
        val size = List.length his_l
        val perr_pi_post = Mlv.make (hidenode, 0.0)
        val perr_pf_post = Mlv.make (hidenode, 0.0)
        val perr_po_post = Mlv.make (hidenode, 0.0)
        val perr_pa_post = Mlv.make (hidenode, 0.0)
        val perr_pc_post = Mlv.make (hidenode, 0.0)
        (* val _ = print ("wdsdsd\n") *)
        fun time_t (backrecord, i, record) =
            case record of
                {c_t_pre, h_t_pre, a_t, i_t, f_t, o_t, c_t, h_t, y_t_0, y_t}
                =>
                  let
                      val y_ans_t = M.sub ans ((size - 1 - i), 0)
                      val y_t' = M.sub y_t (0, 0)
                      val x_t = M.onerow input (size - 1 - i)
                      val f_t_post =
                          case backrecord of
                              NONE => M.make (hidenode, 1, 0.0)
                            | SOME f_t => f_t
                      (* val _ = print ("y_ans_t = " ^ (Real.toString y_ans_t) ^ ", y_t' = " ^ (Real.toString y_t') ^ "\n") *)
                      val perr_py = (y_ans_t - y_t') * (N.dsigmoid y_t')
                      val _ = M.modifyi (fn (_, j, e) =>
                                                e + alpha * (M.sub h_t (j, 0)) * perr_py
                                        ) w_out
                      (* val _ = print ((Real.toString perr_py) ^ "\n") *)
                      (* val _ = print ("w_out = " ^ (M.toString w_out) ^ "\n") *)
                      fun update idx =
                          if idx = hidenode then () else
                          let
                              val w_out_idx = M.sub w_out (0, idx)
                              val u_i_idx = M.onecol u_i idx
                              val u_f_idx = M.onecol u_f idx
                              val u_o_idx = M.onecol u_o idx
                              val u_c_idx = M.onecol u_c idx
                              val c_t_idx = M.sub c_t (idx, 0)
                              val o_t_idx = M.sub o_t (idx, 0)
                              val a_t_idx = M.sub a_t (idx, 0)
                              val f_t_idx = M.sub f_t (idx, 0)
                              val i_t_idx = M.sub i_t (idx, 0)
                              val c_t_pre_idx = M.sub c_t_pre (idx, 0)
                              val h_t_pre_idx = M.sub h_t_pre (idx, 0)
                              val f_t_post_idx = M.sub f_t_post (idx, 0)
                              val perr_pc_post_idx = Mlv.sub perr_pc_post idx
                              val perr_ph = List.foldl (fn ((p, w), b) => (Mlv.dot p w) + b)
                                                       (perr_py * w_out_idx)
                                                       [(perr_pi_post, u_i_idx),
                                                        (perr_pf_post, u_f_idx),
                                                        (perr_po_post, u_o_idx),
                                                        (perr_pa_post, u_c_idx)]
                              (* val _ = print ("perr_ph = " ^ (Real.toString perr_ph) ^ "\n") *)
                              val perr_po = perr_ph * (N.tanh c_t_idx) * (N.dsigmoid o_t_idx)
                              val perr_pc = o_t_idx * perr_ph * (N.dtanh c_t_idx) + perr_pc_post_idx * f_t_post_idx
                              (* val _ = print ("perr_pc = " ^ (Real.toString perr_pc) ^ "\n") *)
                              val perr_pf = c_t_pre_idx * perr_pc * (N.dsigmoid f_t_idx)
                              val perr_pi = a_t_idx *  perr_pc * (N.dsigmoid i_t_idx)
                              val perr_pa = i_t_idx * perr_pc * (N.dsigmoid a_t_idx)
                              fun update_w ((w, p), _) =
                                  let
                                      val w_idx = M.onerow w idx
                                  in
                                      Array.modifyi (fn (j, e) => e + alpha * (Mlv.sub x_t j) * p) w_idx
                                  end
                              fun update_u ((u, p), _) : unit =
                                  let
                                      fun aux (i, j, e) =
                                          if i <> idx then e else
                                          e + alpha * M.sub h_t_pre (j, 0) * p
                                  in
                                      M.modifyi aux u
                                  end
                              val _ = List.foldl update_u ()
                                                 [(u_i, perr_pi),
                                                  (u_f, perr_pf),
                                                  (u_o, perr_po),
                                                  (u_c, perr_pa)]
                              val _ = List.foldl update_w ()
                                                  [(w_i, perr_pi),
                                                  (w_f, perr_pf),
                                                  (w_o, perr_po),
                                                  (w_c, perr_pa)]
                              (* val  _ = print ("w_i = " ^ (M.toString w_i) ^ "\n") *)
                              (* val  _ = print ("u_i = " ^ (M.toString u_i) ^ "\n") *)
                              fun update_flow ((old, new), _) =
                                  Mlv.set old (idx, new)
                              val _ = List.foldl update_flow ()
                                                 [(perr_pi_post, perr_pi),
                                                  (perr_pf_post, perr_pf),
                                                  (perr_po_post, perr_po),
                                                  (perr_pa_post, perr_pa),
                                                  (perr_pc_post, perr_pc)]
                          in
                              update (idx+1)
                          end
                      val _ = update 0
                  in
                      SOME f_t
                  end
        val _ = list_foldli time_t NONE his_l
    in
        ()
    end
end
