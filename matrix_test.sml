structure Mlmatrix = Mlmatrix (open Mlvector);

let
    val mat = Mlmatrix.make (10, 10, 1.0)
    val mat' = Mlmatrix.make (10, 10, 2.0)
    val v = Array.array (10, 0.0)
    val _ = Array.modifyi (fn _ => 2.0) v
    val _ = print (Mlvector.toString v)
    val _ = Mlmatrix.add_modify mat mat'
    val _ = print (Mlmatrix.toString mat)
in
    ()
end;
