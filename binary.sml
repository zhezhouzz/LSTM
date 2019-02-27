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
type binary = bool list
val length = H.binary_dim
val max =
    let
        fun aux i = if i = 0 then [] else true :: (aux (i - 1))
    in
        aux length
    end
val zero =
    let
        fun aux i = if i = 0 then [] else false :: (aux (i - 1))
    in
        aux length
    end
fun fromInt i =
    let
        fun aux i = if i = 1 then [true] else [false]
        fun i2b i =
            if i < 2 then (aux i) else
            (aux (i mod 2)) @ (i2b (i div 2))
    in
        List.rev (i2b i)
    end
end
