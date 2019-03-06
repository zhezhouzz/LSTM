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
                    M.map nonlinear (M.add (M.mul w x_t) (M.mul u h_t_pre))
                val a_t = cal_flow w_c u_c N.sigmoid
                val i_t = cal_flow w_i u_i N.sigmoid
                val f_t = cal_flow w_f u_f N.sigmoid
                val o_t = cal_flow w_o u_o N.sigmoid
                val c_t = (M.add (M.elemwise i_t a_t) (M.elemwise f_t c_t_pre))
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
        fun time_t (backrecord, i, record) =
            case record of
                {c_t_pre, h_t_pre, a_t, i_t, f_t, o_t, c_t, h_t, y_t_0, y_t}
                =>
                  let
                      val y_ans_t = M.row ans (size - 1 - i)
                      val x_t = M.transpose (M.row input (size - 1 - i))
                      val (perr_pi_post, perr_pf_post, perr_po_post,
                           perr_pa_post, perr_pc_post, f_t_post) =
                          case backrecord of
                              NONE => (M.make (1, hidenode, 0.0), M.make (1, hidenode, 0.0),
                                       M.make (1, hidenode, 0.0), M.make (1, hidenode, 0.0),
                                       M.make (1, hidenode, 0.0), M.make (hidenode, 1, 0.0))
                            | SOME {perr_pi_post, perr_pf_post, perr_po_post,
                               perr_pa_post, perr_pc_post, f_t_post} =>
                              (perr_pi_post, perr_pf_post, perr_po_post,
                               perr_pa_post, perr_pc_post, f_t_post)
                      val y_t = M.squeeze12 y_t
                      val y_ans_t = M.squeeze12 y_ans_t
                      val perr_py = (y_ans_t - y_t) * (N.dsigmoid y_t)
                      val _ = M.add_modify w_out (M.mulscalar (M.transpose h_t) (alpha * perr_py) )
                      fun update idx tmp =
                          if idx = hidenode then tmp else
                          let
                              val perr_py = M.make (1, outnode, perr_py)
                              fun onerow mat idx = M.mapi (fn (i, j, e) =>
                                                        if i = idx then e else 0.0) mat
                              fun onecol mat idx = M.mapi (fn (i, j, e) =>
                                                        if j = idx then e else 0.0) mat
                              val w_out_idx = onecol w_out idx
                              val u_i_idx = onecol u_i idx
                              val u_f_idx = onecol u_f idx
                              val u_o_idx = onecol u_o idx
                              val u_c_idx = onecol u_c idx
                              val c_t_idx = onerow c_t idx
                              val o_t_idx = onerow o_t idx
                              val a_t_idx = onerow a_t idx
                              val f_t_idx = onerow f_t idx
                              val i_t_idx = onerow i_t idx
                              val c_t_pre_idx = onerow c_t_pre idx
                              val h_t_pre_idx = onerow h_t_pre idx
                              val f_t_post_idx = onerow f_t_post idx
                              val perr_pc_post_idx = onecol perr_pc_post idx
                              val perr_ph = List.foldl (fn ((p, w), b) => M.add (M.mul p w) b)
                                                       (M.mul perr_py w_out_idx)
                                                       [(perr_pi_post, u_i_idx),
                                                        (perr_pf_post, u_f_idx),
                                                        (perr_po_post, u_o_idx),
                                                        (perr_pa_post, u_c_idx)]
                              val perr_po = List.foldl (fn (a, b) => M.elemwise (M.transpose a) b)
                                                       perr_ph
                                                       [(M.map N.tanh c_t_idx), (M.map N.dsigmoid o_t_idx)]
                              val perr_pc = M.add
                                        (M.elemwise (M.elemwise perr_ph (M.transpose o_t_idx))
                                                    (M.map N.dtanh (M.transpose c_t_idx)))
                                        (M.elemwise perr_pc_post_idx (M.transpose f_t_post_idx))
                              val perr_pf = M.elemwise (M.elemwise perr_pc (M.transpose c_t_pre_idx))
                                                       (M.map N.dsigmoid (M.transpose f_t_idx))
                              val perr_pi = M.elemwise (M.elemwise perr_pc (M.transpose a_t_idx))
                                                       (M.map N.dsigmoid (M.transpose i_t_idx))
                              val perr_pa = M.elemwise (M.elemwise perr_pc (M.transpose i_t_idx))
                                                       (M.map N.dsigmoid (M.transpose a_t_idx))
                              fun update_w ((w, input, p), _) =
                                  M.add_modify w (M.transpose (M.mulscalar (M.mul input p) alpha))
                              val _ = List.foldl update_w ()
                                                 [(u_i, h_t_pre, perr_pi),
                                                  (u_f, h_t_pre, perr_pf),
                                                  (u_o, h_t_pre, perr_po),
                                                  (u_c, h_t_pre, perr_pa),
                                                  (w_i, x_t, perr_pi),
                                                  (w_f, x_t, perr_pf),
                                                  (w_o, x_t, perr_po),
                                                  (w_c, x_t, perr_pa)]
                              val _ =
                                  if i = 0 then () else
                                  let
                                      val _ = M.modifyi (fn (a0, a1, e) =>
                                                            if a1 = idx then (M.sub perr_pi (0, a1)) else e) perr_pi_post
                                      val _ = M.modifyi (fn (a0, a1, e) =>
                                                            if a1 = idx then (M.sub perr_pf (0, a1)) else e) perr_pf_post
                                      val _ = M.modifyi (fn (a0, a1, e) =>
                                                            if a1 = idx then (M.sub perr_po (0, a1)) else e) perr_po_post
                                      val _ = M.modifyi (fn (a0, a1, e) =>
                                                            if a1 = idx then (M.sub perr_pa (0, a1)) else e) perr_pa_post
                                      val _ = M.modifyi (fn (a0, a1, e) =>
                                                            if a1 = idx then (M.sub perr_pc (0, a1)) else e) perr_pc_post
                                  in
                                      ()
                                  end
                              val _ =
                                  case tmp of
                                      (perr_pi_cur, perr_pf_cur, perr_po_cur,
                                       perr_pa_cur, perr_pc_cur) =>
                                      let
                                          val _ = M.add_modify perr_pi_cur perr_pi
                                          val _ = M.add_modify perr_pf_cur perr_pf
                                          val _ = M.add_modify perr_po_cur perr_po
                                          val _ = M.add_modify perr_pa_cur perr_pa
                                          val _ = M.add_modify perr_pc_cur perr_pc
                                      in
                                          ()
                                      end
                          in
                              update (idx+1) tmp
                          end
                      val (perr_pi_cur, perr_pf_cur, perr_po_cur,
                           perr_pa_cur, perr_pc_cur) =
                          update 0 ((M.make (1, hidenode, 0.0)),
                                    (M.make (1, hidenode, 0.0)),
                                    (M.make (1, hidenode, 0.0)),
                                    (M.make (1, hidenode, 0.0)),
                                    (M.make (1, hidenode, 0.0)))
                  in
                      SOME {perr_pi_post = perr_pi_cur,
                            perr_pf_post = perr_pf_cur,
                            perr_po_post = perr_po_cur,
                            perr_pa_post = perr_pa_cur,
                            perr_pc_post = perr_pc_cur,
                            f_t_post = f_t}
                  end
        val _ = list_foldli time_t NONE his_l
    in
        ()
    end
end
