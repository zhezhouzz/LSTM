fun intpow a b =
    if b = 0 then 1 else a * (intpow a (b - 1));
fun mysq (a: real) : real = a * a
fun list_mapi f l =
    let
        fun aux l i =
            case l of [] => []
                    | h :: t =>
                      (f (i, h)) :: (aux t (i + 1))
    in
        aux l 0
    end
fun list_foldli f r l =
    let
        fun aux r l i =
            case l of [] => r
                    | h :: t =>
                      aux (f (r, i, h)) t (i+1)
    in
        aux r l 0
    end
