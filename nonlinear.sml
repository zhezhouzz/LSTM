signature NONLINEAR =
sig
    val sigmoid : real -> real
    val dsigmoid : real -> real
    val tanh : real -> real
    val dtanh : real -> real
    val sigmoidf : real -> real
    val sigmoidh : real -> real
    val sigmoidg : real -> real
end

structure Nonlinear =
struct
fun sigmoid (x: real) : real =
    1.0 / (1.0 + (Math.exp (~x)))
fun dsigmoid (y: real) : real =
    y * (1.0 - y)
fun tanh (x: real) = Math.tanh x
fun dtanh (y: real) : real =
    let
        val y' = tanh y
    in
        1.0 - y'*y'
    end
fun sigmoidf (x: real) : real =
    1.0 / (1.0 + (Math.exp (~x)))
fun sigmoidh (x: real) : real =
    2.0 / (1.0 + (Math.exp (~x))) - 1.0
fun sigmoidg (x: real) : real =
    4.0 / (1.0 + (Math.exp (~x))) - 2.0
end
