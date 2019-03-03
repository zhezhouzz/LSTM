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
fun squeeze v = sub v 0
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
    mapi (fn (i, e1) => f (e1, (sub v2 i))) v1
fun add v1 v2 =
    map2 (fn (a, b) => a + b) v1 v2
fun elemwise v1 v2 =
    map2 (fn (a, b) => a + b) v1 v2
fun dot v1 v2 =
    let
        val v3 = map2 (fn (a, b) => a * b) v1 v2
    in
        Array.foldl (fn (e, r) => r + e) 0.0 v3
    end
fun toStringF (f: scalar -> string) v =
    "[" ^ (Array.foldl (fn (e, str) => str ^ ", " ^ (f e)) "" v) ^ "]\n"
fun toString v = toStringF (fn e => Real.toString e) v
fun fromList l = Array.fromList l
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
fun make (h, w, v) =
    let
        fun f i = Mlv.mlvector (fn _ => v) w
    in
        matrix f h
    end
fun sub mat (i, j) =
    Array.sub ((Array.sub (mat, i)), j)
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
fun row mat i =
    Array.array (1, (Mlv.copy (Array.sub (mat, i))))
fun col mat j =
    let
        val len = Array.length mat
        val result = make (len, 1, 0.0)
        val _ = modifyi (fn (i, _, _) => sub mat (i, j)) result
    in
        result
    end
fun copy mat =
    let
        val (h, w) = size mat
        fun make_vector i =
            Mlv.mlvector (fn j => sub mat (i, j)) w
    in
        matrix make_vector h
    end
fun vec2mat vec =
    Array.array (1, vec)
fun squeeze1 mat =
    Array.sub (mat, 0)
fun squeeze2 mat =
    let
        val vec = Array.array ((Array.length mat), 0.0)
        val _ = Array.modifyi (fn (i, _) => sub mat (i, 0)) vec
    in
        vec
    end
fun squeeze12 mat =
    sub mat (0, 0)
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
fun map2 f mat1 mat2 =
    let
        val mat3 = copy mat1
        val _ = modifyi (fn (i, j, e1) => f (e1, (sub mat2 (i, j)))) mat3
    in
        mat3
    end
fun map2i f mat1 mat2 =
    let
        val mat3 = copy mat1
        val _ = modifyi (fn (i, j, e1) => f (i, j, e1, (sub mat2 (i, j)))) mat3
    in
        mat3
    end
fun foldl f r mat =
    Array.foldl (fn (arr, r) => Mlv.foldl f r arr) r mat
fun foldli f r mat =
    Array.foldli (fn (i, arr, r) => Mlv.foldli (fn (r, j, e) => f (r, i, j ,e)) r arr) r mat
fun foldrowl f r mat =
    Array.foldl (fn (arr, r) => f (r, vec2mat arr)) r mat
fun transpose mat =
    let
        val (h, w) = size mat
        (* val _ = print ("h = " ^ (Int.toString h) ^ ", w = " ^ (Int.toString w) ^ "\n") *)
        val mat' = make (w, h, 0.0)
        val _ = modifyi (fn (i, j, _) => sub mat (j, i)) mat'
    in
        mat'
    end
fun mul mat1 mat2 =
    let
        val (h1, w1) = size mat1
        val (h2, w2) = size mat2
        val mat3 = make (h1, w2, 0.0)
        val mat2' = transpose mat2
        fun f (i, j, e) =
            let
                val row = Array.sub (mat1, i)
                val col = Array.sub (mat2', j)
                val e' = Mlv.dot row col
            in
                e'
            end
        val _ = modifyi f mat3
    in
        mat3
    end
fun add mat1 mat2 =
    map2 (fn (a, b) => a + b) mat1 mat2
fun add_modify (mat1: mlmatrix) mat2 =
    modifyi (fn (i, j, e) => e + (sub mat2 (i, j))) mat1
fun elemwise mat1 mat2 =
    let
        val mat3 = copy mat1
        fun f (i, j, e) =
            e + (sub mat2 (i, j))
        val _ = modifyi f mat3
    in
       mat3
    end
fun matmulvec mat vec =
    let
        val result = Mlv.mlvector (fn _ => 0.0) (Array.length mat)
        fun f (i, _) = Mlv.dot (Array.sub (mat, i)) vec
        val _ = Array.modifyi f result
    in
        result
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
        "\n" ^ (Array.foldl (fn (arr, str) => str ^ (f' arr)) "" mat) ^ "\n"
    end
fun toString mat =
    toStringF Real.toString mat
end
