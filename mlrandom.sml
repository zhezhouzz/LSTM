signature MLRANDOM =
sig
    val init : unit -> unit
    val rand : unit -> int
    val uniformInt : (int * int) -> int
    val uniformReal : (real * real) -> real
end

structure Mlrandom =
struct

exception BAD_ERROR

fun init () =
    case (MLton.Random.seed ()) of NONE => raise BAD_ERROR
                                 | SOME w => MLton.Random.srand w
fun rand () = Word.toInt (MLton.Random.rand ())
fun uniformInt (a, b) =
    let
        val diff = Word.fromInt (b - a)
        val r = MLton.Random.rand ()
    in
        a + (Word.toInt (Word.mod (r, diff)))
    end
fun uniformReal (a, b) =
    let
        val diff = b - a
        val r = uniformInt (0, 10000)
    in
        a + (Real.fromInt r) * diff / 10000.0
    end
end;

(* let *)
(*     val _ = Mlrandom.init () *)
(*     val v = Mlrandom.uniformInt (3, 10) *)
(*     val _ = print ((Int.toString v) ^"\n") *)
(* in *)
(*     () *)
(* end; *)
