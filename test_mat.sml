structure Mlmatrix = Mlmatrix (open Mlvector)
structure Rnn = Rnn (structure H = Hyperparam
                     structure M = Mlmatrix
                     structure N = Nonlinear)
structure Binary = Binary (open Hyperparam);

let
    val a = Mlmatrix.fromList
                [[3.7, 1.8, 1.0],
                 [2.4, 5.5, 46.23]]
    val b = Mlmatrix.fromList
                [[4.6, 6.0, ~12.3],
                 [0.9, ~9.8, 11.9]]
    val c = Mlmatrix.transpose (Mlmatrix.elemwise a b)
    val _ = print (Mlmatrix.toString c)
    val _ = print (Mlmatrix.toString b)
    val _ = print (Mlmatrix.toString a)
    val d = Mlmatrix.make (2, 2, 0.0)
    val _ = print (Mlmatrix.toString d)
    val _ = Mlmatrix.set d (0, 0, 1.0)
    val _ = print (Mlmatrix.toString d)
    val _ = Mlmatrix.modifyi (fn (_, _, a) => a + 1.0) d
    val _ = print (Mlmatrix.toString d)
in
    ()
end;
