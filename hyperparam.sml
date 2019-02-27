signature HYPERPARAM =
sig
    val innode : int
    val hidenode : int
    val outnode : int
    val alpha : real
    val binary_dim : int
end

structure Hyperparam : HYPERPARAM =
struct
val innode = 2
val hidenode = 26
val outnode = 1
val alpha = 0.1
val binary_dim = 8
end
