signature MLVECTOR =
sig
    type mlvector = real array
    type scalar = real
    exception badDimension
	  exception notInvertable
    val size : mlvector -> int
    val mlvector : (int -> scalar) -> int -> mlvector
    val sub : mlvector -> int -> scalar
    val update : (scalar -> scalar) -> mlvector -> int -> unit
    val modify : (scalar -> scalar) -> mlvector -> unit
    val modifyi : ((int * scalar) -> scalar) -> mlvector -> unit
    val foldl : (('a * real) -> 'a) -> 'a -> mlvector -> 'a
    val foldli : (('a * int * real) -> 'a) -> 'a -> mlvector -> 'a
    val copy : mlvector -> mlvector
    val map : (scalar -> scalar) -> mlvector -> mlvector
    val mapi : ((int * scalar) -> scalar) -> mlvector -> mlvector
    val map2: ((scalar * scalar) -> scalar) -> mlvector -> mlvector -> mlvector
    val elemwise : mlvector -> mlvector -> mlvector
    val dot : mlvector -> mlvector -> real
    val toStringF : (scalar -> string) -> mlvector -> string
    val toString : mlvector -> string
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
	  val row : mlmatrix -> int -> mlvector
    val col : mlmatrix -> int -> mlvector
	  val update : (scalar -> scalar) -> mlmatrix -> (int * int) -> unit
    val modify : (scalar -> scalar) -> mlmatrix -> unit
    val modifyi : ((int * int * scalar) -> scalar) -> mlmatrix -> unit
    val copy : mlmatrix -> mlmatrix

	  (* val trans : mlmatrix -> mlmatrix *)
	  (* val inv : mlmatrix -> mlmatrix *)

	  val map : (scalar -> scalar) -> mlmatrix -> mlmatrix
	  val mapi : ((int * int * scalar) -> scalar) -> mlmatrix -> mlmatrix

	  val foldl : (('a * scalar) -> 'a) -> 'a -> mlmatrix -> 'a
	  val foldli : (('a * int * int * scalar) -> 'a) -> 'a -> mlmatrix -> 'a

    val mul : mlmatrix -> mlmatrix -> mlmatrix
	  val elemwise : mlmatrix -> mlmatrix -> mlmatrix
	  val fromArray2 : scalar array array -> mlmatrix
	  val fromList : scalar list list -> mlmatrix
    val toStringF : (scalar -> string) -> mlmatrix -> string
	  val toString : mlmatrix -> string
end
