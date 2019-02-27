signature RNN =
sig
    type rnn
    type flow
    val init : (int -> int -> real) -> rnn
    val forward : rnn -> vector -> vector
end

functor Rnn (structure H: HYPERPARAM
             structure M: MATRIX
             structure B: BINARY
             structure N: NONLINEAR) : RNN =
struct
type matrix = M.matrix
type vector = real array

type rnn =
     {w_i: matrix, u_i : matrix, w_g: matrix, u_g : matrix, w_f: matrix, u_f: matrix, w_o : matrix, u_o : matrix, w_out: matrix, x: vector, y : vector}
type flow =
     {v_i: vector, v_g: vector, v_f: vector, v_o: vector, v_s: vector, v_h: vector, v_delta_y: vector}

val innode = H.innode
val hidenode = H.hidenode
val outnode = H.outnode

fun init_matrix f (w, h) =
    let
        val mat = M.matrix ((w, h), 0.0)
        val mat = M.mapi (fn ((i,j), _) => f i j) mat
    in
        mat
    end

fun init f =
    let
        val w_i = init_matrix f (innode, hidenode)
        val u_i = init_matrix f (hidenode, hidenode)
        val w_f = init_matrix f (innode, hidenode)
        val u_f = init_matrix f (hidenode, hidenode)
        val w_o = init_matrix f (innode, hidenode)
        val u_o = init_matrix f (hidenode, hidenode)
        val w_g = init_matrix f (innode, hidenode)
        val u_g = init_matrix f (hidenode, hidenode)
        val w_out = init_matrix f (hidenode, outnode)
        val x = B.zero
        val y = B.zero
    in
        {w_i = w_i, u_i = u_i, w_g = w_g, u_g = u_g, w_f = w_f, u_f = u_f, w_o = w_o, u_o = u_o, w_out = w_out, x = x, y = y}
    end

val init_flow =
    let
        val v_i = Array.array (H.binary_dim, 0.0)
        val v_g = Array.array (H.binary_dim, 0.0)
        val v_f = Array.array (H.binary_dim, 0.0)
        val v_o = Array.array (H.binary_dim, 0.0)
        val v_s = Array.array (H.binary_dim, 0.0)
        val v_h = Array.array (H.binary_dim, 0.0)
        val v_delta_y = Array.array (H.binary_dim, 0.0)
    in
        {v_i = v_i, v_g = v_g, v_f = v_f, v_o = v_o, v_s = v_s, v_h, v_h, v_delta_y = v_delta_y}
    end

fun forward rnn (input, output) =
    let
        val flow = init_flow
        fun once i =
            let
                val h_t = flow.v_h.(i)
                val i_0_t = w_i * x_t + u_i * h_0_t
                val f_0_t = w_f * x_t + u_f * h_0_t
                val o_0_t = w_o * x_t + u_o * h_0_t
                val g_0_t = w_g * x_t + u_g * h_0_t
                val i_1_t = List.map N.sigmoid i_0_t
                val f_1_t = List.map N.sigmoid f_0_t
                val o_1_t = List.map N.sigmoid o_0_t
                val g_1_t = List.map N.sigmoid g_0_t
                val s_1_t = f_1_t  s_0_t + i_1_t * g_1_t
                val h_1_t = o_1_t * N.tanh(s_t)
                                          
            in
            end
                
    in
    end
end
