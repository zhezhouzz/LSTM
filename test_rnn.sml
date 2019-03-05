structure Mlmatrix = Mlmatrix (open Mlvector)
structure Rnn = Rnn (structure H = Hyperparam
                     structure M = Mlmatrix
                     structure N = Nonlinear)
structure Binary = Binary (open Hyperparam);

fun data_gen () =
    let
        val a = Mlrandom.uniformInt (0, 256)
        val b = Mlrandom.uniformInt (0, 256)
        val c = (a + b) mod 256
        val a' = Binary.fromInt a
        val b' = Binary.fromInt b
        val c' = Binary.fromInt c
        val input = Mlmatrix.make (8, 2, 0.0)
        val _ = Mlmatrix.modifyi (fn (i, j, _) =>
                              if j = 0
                              then
                                  Array.sub (a', i)
                              else
                                  Array.sub (b', i)
                          ) input
        val output = Mlmatrix.make (8, 1, 0.0)
        val _ = Mlmatrix.modifyi (fn (i, j, _) =>
                                     Array.sub (c', i)
                                 ) output
        (* val _ = print ("a = " ^ (Int.toString a) ^ "\n") *)
        (* val _ = print ("b = " ^ (Int.toString b) ^ "\n") *)
        (* val _ = print (Mlmatrix.toString input) *)
        (* val _ = print ("c = " ^ (Int.toString c) ^ "\n") *)
        (* val _ = print (Mlvector.toString c') *)
        (* val _ = print (Mlmatrix.toString output) *)
    in
        (input, output)
    end;

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
            val (input, ans) = data_gen ()
            val (output, record_l) = Rnn.forward rnn input
            val _ = if (k mod 500) = 0
                    then
                        let
                            val _ = print ("epoch: " ^ (Int.toString k) ^ "\n")
                            (* val _ = print (Mlmatrix.toString input) *)
                            val _ = print (Mlmatrix.toString ans)
                            val _ = print (Mlmatrix.toString output)
                        in
                            ()
                        end
                    else
                        ()
            (* val _ = print (Mlmatrix.toString output) *)
            val _ = Rnn.backward rnn record_l input ans 0.5
        in
            train (k - 1)
        end
    val _ = train 3000
    (* val _ = print (Mlmatrix.toString input) *)
    (* val (output, record_l) = Rnn.forward rnn input *)
    (* val _ = print (Mlmatrix.toString output) *)
in
    ()
end;
