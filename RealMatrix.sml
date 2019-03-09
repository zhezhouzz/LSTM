structure Mlvector : MLVECTOR =
struct
type mlvector = real array
type scalar = real
exception badDimension
exception notInvertable

fun size v =
    Array.length v
fun make (num, e) =
    Array.array (num, e)
fun mlvector f num =
    let
        val v = Array.array (num, 0.0)
        val _ = Array.modifyi (fn (i, _) => f i) v
    in
        v
    end
fun sub v i =
    Array.sub (v, i)
fun set v (i, e) =
    Array.update (v, i, e)
fun update f v i =
    let
        val value = sub v i
    in
        Array.update (v, i, (f value))
    end
fun modify f v = Array.modify f v
fun modifyi f v = Array.modifyi f v
fun foldl f r v = Array.foldl (fn (e, r) => f (r, e)) r v
fun foldli f r v =
    let
        val len = size v
        fun aux i r =
            if i = len then r else
            let
                val e = Array.sub (v, i)
                val r = f (r, i, e)
            in
                aux (i+1) r
            end
    in
        aux 0 r
    end
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
exception badDimension

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
        fun f i = Mlv.make (w, v)
    in
        matrix f h
    end
fun sub mat (i, j) =
    Array.sub ((Array.sub (mat, i)), j)
fun set mat (i, j, e) =
    let
        val arr = Array.sub (mat, i)
        val _ = Array.update (arr, j, e)
    in
        ()
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
    let
        val (h, w) = size mat
        fun aux i =
            if i = h then () else
            let
                val arr = Array.sub (mat, i)
                fun aux' j =
                    if j = w then () else
                    let
                        val e = Array.sub (arr, j)
                        val e = f (i, j, e)
                        val _ = Array.update (arr, j, e)
                    in
                        aux' (j+1)
                    end
                val _ = aux' 0
                val _ = Array.update (mat, i, arr)
            in
                aux (i+1)
            end
    in
        aux 0
    end
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
fun map_inplace f mat =
    let
        val _ = modify f mat
    in
        mat
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
    let
        val (h, w) = size mat
        fun aux i r =
            if i = h then r else
            let
                val arr = Array.sub (mat, i)
                fun aux' j r =
                    if j = w then r else
                    let
                        val e = Array.sub (arr, j)
                        val r = f (r, i, j, e)
                    in
                        aux' (j+1) r
                    end
                val r = aux' 0 r
            in
                aux (i+1) r
            end
    in
        aux 0 r
    end
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
fun mulscalar mat a =
    map (fn e => a * e) mat
fun mul mat1 mat2 =
    let
        val (h1, w1) = size mat1
        val (h2, w2) = size mat2
    in
        if w1 <> h2 then raise badDimension else
        let
            val mat3 = make (h1, w2, 0.0)
            fun f (i, j) =
                let
                    fun dot k =
                        if k = w1 then 0.0 else
                        let
                            val v = (sub mat1 (i, k)) * (sub mat2 (k, j))
                            val k = k + 1
                        in
                            v + (dot k)
                        end
                    val v = dot 0
                in
                    set mat3 (i, j, v)
                end
            fun aux i j =
                if j = w2 then () else
                let
                    val _ = f (i, j)
                in
                    aux i (j + 1)
                end
            fun aux' i =
                if i = h1 then () else
                let
                    val _ = aux i 0
                in
                    aux' (i + 1)
                end
            val _ = aux' 0
        in
            mat3
        end
    end

fun add mat1 mat2 =
    map2 (fn (a, b) => a + b) mat1 mat2
fun add_modify (mat1: mlmatrix) mat2 =
    modifyi (fn (i, j, e) => e + (sub mat2 (i, j))) mat1
fun add_inplace mat1 mat2 =
    let
        val _ = add_modify mat1 mat2
    in
        mat1
    end
fun elemwise mat1 mat2 =
    let
        val mat3 = copy mat1
        fun f (i, j, e) =
            e * (sub mat2 (i, j))
        val _ = modifyi f mat3
    in
       mat3
    end
fun elemwise_inplace (mat1: mlmatrix) mat2 =
    let
        fun f (i, j, e) =
            e * (sub mat2 (i, j))
        val _ = modifyi f mat1
    in
        mat1
    end
fun matmulvec mat vec =
    let
        val result = Mlv.make ((Array.length mat), 0.0)
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
