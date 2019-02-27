structure Mlvector : MLVECTOR =
struct
type mlvector = real array
type scalar = real
exception badDimension
exception notInvertable

fun size v =
    Array.length v
fun mlvector f num =
    let
        val v = Array.array (num, 0.0)
        val _ = Array.modifyi (fn (i, _) => f i) v
    in
        v
    end
fun sub v i =
    Array.sub (v, i)
fun update f v i =
    let
        val value = sub v i
    in
        Array.update (v, i, (f value))
    end
fun modify f v = Array.modify f v
fun modifyi f v = Array.modifyi f v
fun foldl f r v = Array.foldl (fn (e, r) => f (r, e)) r v
fun foldli f r v = Array.foldli (fn (i, e, r) => f (r, i, e)) r v
fun copy v = mlvector (fn i => sub v i) (size v)
fun map f v =
    let
        val v'= copy v
        val _ = modify f v'
    in
        v'
    end
fun mapi f v =
    let
        val v' = copy v
        val _ = modifyi f v'
    in
        v'
    end
fun map2 f v1 v2 =
    mapi (fn (i, e1) => e1 + (sub v2 i)) v1
fun elemwise v1 v2 =
    map2 (fn (a, b) => a + b) v1 v2
fun dot v1 v2 =
    let
        val v3 = map2 (fn (a, b) => a * b) v1 v2
    in
        Array.foldl (fn (e, r) => r + e) 0.0 v3
    end
fun toStringF (f: scalar -> string) v =
    "[" ^ (Array.foldl (fn (e, str) => str ^ ", " ^ (f e)) "]\n" v)
fun toString v = toStringF (fn e => Real.toString e) v
end

functor Mlmatrix (Mlvector: MLVECTOR) : MLMATRIX =
struct
structure Mlv = Mlvector
type scalar = Mlv.scalar
type mlvector = Mlv.mlvector
type mlmatrix = real array array

exception badError

fun size mat =
    let
        val len = Array.length mat
        val len' = Array.length (Array.sub (mat, 0))
    in
        (len, len')
    end
fun matrix f num =
    let
        val v = f 0
        val mat = Array.array (num, v)
        val _ = Array.modifyi (fn (i, _) => f i) mat
    in
        mat
    end
fun make (w, h, v) =
    let
        fun f i = Mlv.mlvector (fn _ => v) w
    in
        matrix f h
    end
fun sub mat (i, j) =
    Array.sub ((Array.sub (mat, i)), j)
fun row mat i =
    Mlv.copy (Array.sub (mat, i))
fun col mat j =
    let
        val len = Array.length mat
        val result = Array.array (len, 0.0)
        val _ = Array.modifyi (fn (i, _) => sub mat (i, j)) result
    in
        result
    end
fun update f mat (i, j) =
    let
        val e = sub mat (i, j)
        val arr = Array.sub (mat, i)
        val _ = Array.update (arr, j, (f e))
    in
        ()
    end
fun modify f mat =
    Array.foldl (fn (arr, _) => Array.modify f arr) () mat
fun modifyi f mat =
    Array.foldli (fn (i, arr, _) => Array.modifyi (fn (j, e) => f (i, j, e)) arr) () mat
fun copy mat =
    let
        val (h, w) = size mat
        fun make_vector i =
            Mlv.mlvector (fn j => sub mat (i, j)) w
    in
        matrix make_vector h
    end
fun map f mat =
    let
        val mat' = copy mat
        val _ = modify f mat'
    in
        mat'
    end
fun mapi f mat =
    let
        val mat' = copy mat
        val _ = modifyi f mat'
    in
        mat'
    end
fun foldl f r mat =
    Array.foldl (fn (arr, r) => Mlv.foldl f r arr) r mat
fun foldli f r mat =
    Array.foldli (fn (i, arr, r) => Mlv.foldli (fn (r, j, e) => f (r, i, j ,e)) r arr) r mat
fun mul mat1 mat2 =
    let
        val (h1, w1) = size mat1
        val (h2, w2) = size mat2
        val mat3 = make (h1, w2, 0.0)
        fun f (i, j, e) =
            let
                val row = row mat1 i
                val col = col mat2 j
                val e' = Mlv.dot row col
            in
                e'
            end
        val _ = modifyi f mat3
    in
        mat3
    end
fun elemwise mat1 mat2 =
    let
        val mat3 = copy mat1
        fun f (i, j, e) =
            e + (sub mat2 (i, j))
        val _ = modifyi f mat3
    in
       mat3
    end
fun fromArray2 mat = mat
fun fromList ll =
    let
        val len = List.length ll
        fun f i = Array.fromList (List.nth (ll, i))
    in
        matrix f len
    end
fun toStringF f mat =
    let
        fun f' arr = Mlv.toStringF f arr
    in
        "[" ^ (Array.foldl (fn (arr, str) => str ^ (f' arr)) "]\n" mat)
    end
fun toString mat =
    toStringF Real.toString mat
end
