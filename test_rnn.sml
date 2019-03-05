structure Mlmatrix = Mlmatrix (open Mlvector)
structure Rnn = Rnn (structure H = Hyperparam
                     structure M = Mlmatrix
                     structure N = Nonlinear)
        ;

let
    val _ = Mlrandom.init ()
    (* val rnn = Rnn.init (fn _ => Mlrandom.uniformReal (~1.0, 1.0)) *)
    val rnn = Rnn.init (fn _ => 0.1)
    val input = Mlmatrix.fromList
                    [[0.0, 1.0],
                     [1.0, 0.0],
                     [1.0, 0.0],
                     [0.0, 0.0],
                     [1.0, 0.0],
                     [0.0, 0.0],
                     [1.0, 0.0],
                     [0.0, 0.0]]
    val ans = Mlmatrix.fromList
                    [[1.0],
                     [1.0],
                     [1.0],
                     [0.0],
                     [1.0],
                     [0.0],
                     [1.0],
                     [0.0]]
    fun train k =
        if k <= 0 then () else
        let
            val (output, record_l) = Rnn.forward rnn input
            (* val _ = print (Mlmatrix.toString output) *)
            val _ = Rnn.backward rnn record_l input ans 0.1
        in
            train (k - 1)
        end
    val _ = train 1
    val _ = print (Mlmatrix.toString input)
    val (output, record_l) = Rnn.forward rnn input
    val _ = print (Mlmatrix.toString output)
in
    ()
end;
