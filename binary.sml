signature BINARY =
sig
    type binary
    val length : int
    val max : binary
    val zero: binary
    val fromInt : int -> binary
end

functor Binary (H: HYPERPARAM) : BINARY=
struct
type binary = real array
val length = H.binary_dim
val max =
    Array.array (length, 1.0)
val zero =
    Array.array (length, 0.0)
fun fromInt i =
    let
        val result = Array.array (length, 0.0)
        fun aux i = if i = 1 then 1.0 else 0.0
        fun i2b i num =
            if i = length then () else
            let
                val r = aux (num mod 2)
                val num = num div 2
                val _ = Array.update (result, i, r)
            in
                i2b (i+1) num
            end
        val _ = i2b 0 i
    in
        result
    end
end
