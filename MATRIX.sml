signature MLVECTOR =
sig
    type mlvector = real array
    type scalar = real
    exception badDimension
	  exception notInvertable
    val size : mlvector -> int
    val make : (int * scalar) -> mlvector
    val mlvector : (int -> scalar) -> int -> mlvector
    val sub : mlvector -> int -> scalar
    val set : mlvector -> (int * scalar) -> unit
    val update : (scalar -> scalar) -> mlvector -> int -> unit
    val modify : (scalar -> scalar) -> mlvector -> unit
    val modifyi : ((int * scalar) -> scalar) -> mlvector -> unit
    val foldl : (('a * real) -> 'a) -> 'a -> mlvector -> 'a
    val foldli : (('a * int * real) -> 'a) -> 'a -> mlvector -> 'a
    val copy : mlvector -> mlvector
    val squeeze : mlvector -> scalar
    val map : (scalar -> scalar) -> mlvector -> mlvector
    val mapi : ((int * scalar) -> scalar) -> mlvector -> mlvector
    val map2: ((scalar * scalar) -> scalar) -> mlvector -> mlvector -> mlvector
    val elemwise : mlvector -> mlvector -> mlvector
    val add : mlvector -> mlvector -> mlvector
    val dot : mlvector -> mlvector -> real
    val toStringF : (scalar -> string) -> mlvector -> string
    val toString : mlvector -> string
    val fromList : real list -> mlvector
end

signature MLMATRIX =
sig
    structure Mlv: MLVECTOR
    type scalar = Mlv.scalar
    type mlvector = Mlv.mlvector
	  type mlmatrix = real array array

    exception badError

	  val size : mlmatrix -> (int * int)
	  val matrix : (int -> mlvector) -> int -> mlmatrix
    val make : (int * int * real) -> mlmatrix
	  val sub : mlmatrix -> (int * int) -> scalar
	  val row : mlmatrix -> int -> mlmatrix
    val col : mlmatrix -> int -> mlmatrix
    val set : mlmatrix -> (int * int * scalar) -> unit
	  val update : (scalar -> scalar) -> mlmatrix -> (int * int) -> unit
    val modify : (scalar -> scalar) -> mlmatrix -> unit
    val modifyi : ((int * int * scalar) -> scalar) -> mlmatrix -> unit
    val copy : mlmatrix -> mlmatrix
    val vec2mat : mlvector -> mlmatrix
    val squeeze1 : mlmatrix -> mlvector
    val squeeze2 : mlmatrix -> mlvector
    val squeeze12 : mlmatrix -> scalar

	  (* val trans : mlmatrix -> mlmatrix *)
	  (* val inv : mlmatrix -> mlmatrix *)

	  val map : (scalar -> scalar) -> mlmatrix -> mlmatrix
	  val map_inplace : (scalar -> scalar) -> mlmatrix -> mlmatrix
	  val mapi : ((int * int * scalar) -> scalar) -> mlmatrix -> mlmatrix
    val map2 : ((scalar * scalar) -> scalar) -> mlmatrix -> mlmatrix -> mlmatrix
    val map2i : ((int * int * scalar * scalar) -> scalar) -> mlmatrix -> mlmatrix -> mlmatrix

	  val foldl : (('a * scalar) -> 'a) -> 'a -> mlmatrix -> 'a
	  val foldli : (('a * int * int * scalar) -> 'a) -> 'a -> mlmatrix -> 'a
    val foldrowl : ('a * mlmatrix -> 'a) -> 'a -> mlmatrix -> 'a

    val transpose : mlmatrix -> mlmatrix
    val mulscalar : mlmatrix -> scalar -> mlmatrix
    val mul : mlmatrix -> mlmatrix -> mlmatrix
    val add : mlmatrix -> mlmatrix -> mlmatrix
    val add_inplace : mlmatrix -> mlmatrix -> mlmatrix
    val add_modify : mlmatrix -> mlmatrix -> unit
	  val elemwise : mlmatrix -> mlmatrix -> mlmatrix
	  val elemwise_inplace : mlmatrix -> mlmatrix -> mlmatrix
    val matmulvec : mlmatrix -> mlvector -> mlvector
	  val fromArray2 : scalar array array -> mlmatrix
	  val fromList : scalar list list -> mlmatrix
    val toStringF : (scalar -> string) -> mlmatrix -> string
	  val toString : mlmatrix -> string
end
