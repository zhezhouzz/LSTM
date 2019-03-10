# LSTM

Use LSTM to learn an adder with carry. The result of less significant bit would interfere the more significant bit, thus the LSTM can work here.

#### Network Structure

input layer: 2x26
hidden layer: 26x26
output layer: 26x1

Learning rate: 0.5

Trainning data: randomly generated 8-bit number.

Epoch: 4000

#### Code

```
MATRIX.sml (Header of built-in Matrix module)
RealMatrix.sml (Implementation of built-in Matrix module)
binary.sml 
hyperparam.sml
lstm.mlb (makefile)
mlrandom.sml (built-in Random module)
nonlinear.sml (built-in Math module)
rnn.sml
test_rnn.sml
utils.sml (built-in)
```

+ `test_rnn.sml` is the main file.
+ `rnn.sml` is the core implememtation of RNN, includes init, forward and backward. 
+ `hyperparam.sml` is configuration file. 
+ `binary.sml` is the data structure for this task.

#### Compilation

```bash
mlton -default-ann 'allowRecordPunExps true' lstm.mlb
./lstm
```

+ Profiling:

```
mlton -profile time -default-ann 'allowRecordPunExps true' lstm.mlb
./lstm
mlprof lstm mlmon.out
```

#### Result

input: 8x2 {0, 1} matrix, represent two 8-bits number. the first line is the least significant bit.
output: 8x1 real matrix, represent a 8-bits number(if it is greater than 0.5 then 1, less than 0.5 then 0). the first line is the least significant bit.

Profiling Result:

```
                  function                    cur
-------------------------------------------- -----
_res_Mlmatrix.modifyi.aux.aux'               55.9%
_res_Mlmatrix.modifyi.aux                     9.4%
_res_PrimSequence.Slice.foldli.loop           7.5%
_res_Mlmatrix.mul.f.dot                       6.0%
_res_Rnn.backward.time_t.update.update_u.aux  5.0%
<gc>                                          3.2%
Mlvector.mlvector                             2.5%
_res_Mlmatrix.matrix                          2.2%
_res_Mlmatrix.onecol                          1.8%
Mlvector.dot.fn                               1.8%
_res_Rnn.backward.time_t.update               0.9%
_res_Mlmatrix.sub                             0.9%
Mlvector.dot                                  0.6%
Mlvector.modifyi                              0.4%
Nonlinear.sigmoid                             0.4%
Mlvector.mapi                                 0.3%
Mlvector.map2.fn                              0.3%
_res_Mlmatrix.size                            0.1%
_res_Mlmatrix.set                             0.1%
_res_Mlmatrix.mul.f                           0.1%
_res_Mlmatrix.vec2mat                         0.1%
_res_Mlmatrix.add_modify.fn                   0.1%
PosixError.SysCall.simpleResultAux            0.1%
```

+ `_res_Mlmatrix.modifyi` is the function used to implement add, elemwise, scalar multiply matrix and other matrix operations.
+ `_res_PrimSequence.Slice.foldli.loop` is the iteration function for array, which is used to iterate matrix and vector.
+ `_res_Mlmatrix.mul.f.dot` is used to implement matrix multiplication.

If the lower level provide some efficient way to iterate and modify the matrix and vector, this operations would be faster.
