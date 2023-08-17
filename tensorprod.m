########################################################################
##
## Copyright (C) 2022 Kasper H. Filtenborg
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.
##
########################################################################

## -*- texinfo -*-
## @deftypefn  {} {@var{C} =} tensorprod (@var{A}, @var{B}, @var{dimA}, @var{dimB})
## @deftypefnx {} {@var{C} =} tensorprod (@var{A}, @var{B}, @var{dim})
## @deftypefnx {} {@var{C} =} tensorprod (@var{A}, @var{B})
## @deftypefnx {} {@var{C} =} tensorprod (@var{A}, @var{B}, "all")
## @deftypefnx {} {@var{C} =} tensorprod (@var{A}, @var{B}, @dots{}, "NumDimensionsA", @var{value})
## Compute the tensor product between tensors @var{A} and @var{B}.
##
## The dimensions of @var{A} and @var{B} that are contracted are defined by
## @var{dimA} and @var{dimB}, respectively. @var{dimA} and @var{dimB} are
## vectors with equal length and define the dimensions to match up. The
## matched dimensions of @var{A} and @var{B} must have equal length.
##
## When @var{dim} is used, it is equivalent to @var{dimA} = @var{dimB} =
## @var{dim}.
##
## Running without an additional argument is equivalent to @var{dimA} =
## @var{dimB} = []. This computes the outer product between @var{A} and
## @var{B}.
##
## Using the "all" option results in the inner product between @var{A} and
## @var{B}. For this, it is required that size(@var{A}) == size(@var{B}).
##
## Use the property-value pair with the property name "NumDimensionsA"
## when @var{A} has trailing singleton dimensions that should be transfered to
## @var{C}. The specified value should be the number of dimensions of @var{A}.
##
## @seealso{kron, dot, mtimes}
## @end deftypefn

function C = tensorprod (A, B, varargin)

  if (nargin == 0)
    print_usage ();
  elseif  (nargin < 2)
    error ("tensorprod: too few inputs given");
  elseif  (nargin > 6)
    error ("tensorprod: too many inputs given");
  endif

  ## Check that A and B are single or double
  if (! isfloat (A))
    error ("tensorprod: A must be a single or double precision array");
  endif

  if (! isfloat (B))
    error ("tensorprod: B must be a single or double precision array");
  endif

  ## Check for misplaced NumDimensionsA property
  if (nargin > 2)
    if (strcmpi (varargin{end}, "NumDimensionsA"))
      error (["tensorprod: a value for the NumDimensionsA property must ", ...
        "be provided"]);
    elseif (strcmpi ( strtok (inputname (nargin, false)), "NumDimensionsA"))    # FIXME: Add support for keyword=value syntax
      error (["tensorprod: NumDimensionsA=ndimsA syntax is not yet ", ...
        "supported in Octave - provide the value as a property-value pair"]);
    endif
  endif

  ## Check for NumDimensionsA property
  if (nargin > 3)
    if (strcmpi (varargin{end - 1}, "NumDimensionsA"))
      if (! (isnumeric (varargin{end}) && isscalar (varargin{end})))
        error (["tensorprod: value for NumDimensionsA must be a ", ...
          "numeric scalar"]);
      elseif (varargin{end} < 1 || mod (varargin{end}, 1) != 0)
        error (["tensorprod: value for NumDimensionsA must be a ", ...
          "positive integer"]);
      endif
      NumDimensionsA = varargin{end};
    endif
  endif

  existNumDimensionsA = exist ("NumDimensionsA");
  ndimargs = nargin - 2 - 2 * existNumDimensionsA;

  ## Set dimA and dimB
  if (ndimargs == 0)
    ## Calling without dimension arguments
    dimA = [];
    dimB = [];
  elseif (ndimargs == 1)
    ## Calling with dim or "all" option
    if (isnumeric (varargin{1}))
      if (! (isvector (varargin{1}) || isnull (varargin{1})))
        error ("tensorprod: dim must be a numeric vector of integers or []");
      endif
      ## Calling with dim
      dimA = transpose ([varargin{1}(:)]);
    elseif (ischar (varargin{1}))
      if (strcmpi (varargin{1}, "all"))
        if (! size_equal (A, B))
          error (["tensorprod: size of A and B must be identical when ", ...
            "using the \"all\" option"]);
        endif
      else
        error ("tensorprod: unknown option \"%s\"", varargin{1});
      endif
      ## Calling with "all" option
      dimA = 1:ndims(A);
    else
      error (["tensorprod: third argument must be a numeric vector of ", ...
        "integers, [], or \"all\""]);
    endif
    dimB = dimA;
  elseif (ndimargs == 2)
    ## Calling with dimA and dimB
    if (! (isnumeric (varargin{1}) && (isvector (varargin{1}) || ...
        isnull (varargin{1}))))
      error("tensorprod: dimA must be a numeric vector of integers or []");
    endif

    if (! (isnumeric (varargin{2}) && (isvector (varargin{2}) || ...
        isnull (varargin{2}))))
      error ("tensorprod: dimB must be a numeric vector of integers or []");
    endif

    if (length (varargin{1}) != length (varargin{2}))
      error (["tensorprod: an equal number of dimensions must be ", ...
        "matched for A and B"]);
    endif
    dimA = transpose ([varargin{1}(:)]);
    dimB = transpose ([varargin{2}(:)]);
  else
    ## Something is wrong - try to find the error
    for i = 1:ndimargs
      if (ischar (varargin{i}))
        if (strcmpi (varargin{i}, "NumDimensionsA"))
          error ("tensorprod: misplaced \"NumDimensionsA\" option");
        elseif (strcmpi (varargin{i}, "all"))
          error ("tensorprod: misplaced \"all\" option");
        else
          error ("tensorprod: unknown option \"%s\"", varargin{i});
        endif
      elseif (! isnumeric (varargin{i}))
        error (["tensorprod: optional arguments must be numeric vectors ", ...
          "of integers, [], \"all\", or \"NumDimensionsA\""]);
      endif
    endfor
    error ("tensorprod: too many dimension inputs given");
  endif

  ## Check that dimensions are positive integers ([] will also pass)
  if (any ([dimA < 1, dimB < 1, (mod (dimA, 1) != 0), (mod (dimB, 1) != 0)]))
    error ("tensorprod: dimension(s) must be positive integer(s)");
  endif

  ## Check that the length of matched dimensions are equal
  if (any (size (A, dimA) != size (B, dimB)))
    error (["tensorprod: matched dimension(s) of A and B must have the ", ...
      "same length(s)"]);
  endif

  ## Find size and ndims of A and B
  ndimsA = max ([ndims(A), max(dimA)]);
  sizeA = size (A, 1:ndimsA);
  ndimsB = max ([ndims(B), max(dimB)]);
  sizeB = size (B, 1:ndimsB);

  ## Take NumDimensionsA property into account
  if (existNumDimensionsA)
    if (NumDimensionsA < ndimsA)
      if (ndimargs == 1)
        error (["tensorprod: highest dimension of dim must be less than ", ...
          "or equal to NumDimensionsA"]);
      elseif (ndimargs == 2)
        error (["tensorprod: highest dimension of dimA must be less ", ...
          "than or equal to NumDimensionsA"]);
      else
        error (["tensorprod: NumDimensionsA cannot be smaller than the ", ...
          "number of dimensions of A"]);
      endif
    elseif (NumDimensionsA > ndimsA)
      sizeA = [sizeA, ones(1, NumDimensionsA - ndimsA)];
      ndimsA = NumDimensionsA;
    endif
  endif

  ## Interchange the dimension to sum over the end of A and the front of B
  ## Prepare for A
  remainDimA = setdiff (1:ndimsA, dimA);                                        # Dimensions of A to keep
  newDimOrderA =  [remainDimA, dimA];                                           # New order of dimensions (dimensions to keep first, dimensions to contract last)
  newSizeA = [prod(sizeA(remainDimA)), prod(sizeA(dimA))];                      # Size of temporary 2D representation of A
  remainSizeA = sizeA(remainDimA);                                              # Contribution to size of C from remaining dimensions of A

  ## Prepare for B
  remainDimB = setdiff (1:ndimsB, dimB);                                        # See comments for A
  newDimOrderB =  [remainDimB, dimB];
  newSizeB = [prod(sizeB(remainDimB)), prod(sizeB(dimB))];                      # In principle, prod(sizeB(dimB)) should always be equal to prod(sizeA(dimA))
  remainSizeB = sizeB(remainDimB);

  ## Do reshaping into 2D array
  newA = reshape (permute (A, newDimOrderA), newSizeA);
  newB = reshape (permute (B, newDimOrderB), newSizeB);

  ## Compute
  C = newA * transpose (newB);

  ## If not an inner product, reshape back to tensor
  if (! isscalar (C))
    C = reshape (C, [remainSizeA, remainSizeB]);
  endif

endfunction


%!test
%! rand ("seed", 0);
%! A = rand (3, 2, 5);
%! B = rand (2, 4, 5);
%! v = rand (4, 1);
%! assert ( tensorprod (A, B));
%! assert ( tensorprod (A, A, 1));
%! assert ( tensorprod (A, A, 4));
%! assert ( tensorprod (A, A, []));
%! assert ( tensorprod (A, A, [2, 3]));
%! assert ( tensorprod (A, B, 2, 1));
%! assert ( tensorprod (A, B, 4, 4));
%! assert ( tensorprod (A, B, [2, 3], [1, 3]));
%! assert ( tensorprod (A, A, [], []));
%! assert ( A(1), 0.9999996423721313330669073875470);
%! assert ( tensorprod (A, A, "all"), 11.258348100536915329070518200751);
%! assert ( tensorprod (A, A, "all"));
%! assert ( tensorprod (A, A, 1, "NumDimensionsA", 4));
%! assert ( tensorprod (A, A, 4, "NumDimensionsA", 4));
%! assert ( tensorprod (A, A, 4, "numdimensionsa", 4));
%! assert ( tensorprod (A, A, [2, 3], "NumDimensionsA", 4));
%! assert ( tensorprod (A, A, [], "NumDimensionsA", 4));
%! assert ( tensorprod (A, B, 2, 1, "NumDimensionsA", 4));
%! assert ( tensorprod (A, B, [2, 3], [1, 3], "NumDimensionsA", 4));
%! assert ( tensorprod (A, B, [2, 3], [1; 3], "NumDimensionsA", 4));
%! assert ( tensorprod (A, B, [], [], "NumDimensionsA", 4));
%! assert ( tensorprod (1, 2), 2);
%! assert ( tensorprod (v, v, "all"), dot (v, v));
%! assert ( tensorprod (v, v), reshape (v * transpose (v), [4, 1, 4]));

## Test empty inputs
%!assert ( tensorprod ([], []), zeros (0, 0, 0, 0))
%!assert ( tensorprod ([], 1), [])
%!assert ( tensorprod (1, []), zeros (1, 1, 0, 0))
%!assert ( tensorprod (zeros (0, 0, 0), zeros (0, 0, 0)), zeros (0, 0, 0, 0, 0, 0))
%!assert ( tensorprod ([], [], []), zeros (0, 0, 0, 0))
%!assert ( tensorprod ([], [], 1), [])
%!assert ( tensorprod ([], [], 2), [])
%!assert ( tensorprod ([], [], 3), zeros (0, 0, 0, 0))
%!assert ( tensorprod ([], [], 4), zeros (0, 0, 1, 0, 0))
%!assert ( tensorprod ([], [], 5), zeros (0, 0, 1, 1, 0, 0))
%!assert ( tensorprod ([], [], 3, "NumDimensionsA", 4), zeros (0, 0, 1, 0, 0))
%!assert ( tensorprod ([], [], 3, 4, "NumDimensionsA", 5), zeros (0, 0, 1, 1, 0, 0))

## Test input validation
%!error <Invalid call to tensorprod.  Correct usage is:> tensorprod ()
%!error <too few inputs given> tensorprod (1)
%!error <A must be a single or double precision array> tensorprod ("foo", 1)
%!error <B must be a single or double precision array> tensorprod (1, "bar")
%!error <unknown option "foo"> tensorprod (1, 1, "foo")
%!error <dimA must be a numeric vector of integers or \[\]> tensorprod (1, 1, "foo", 1)
%!error <dimB must be a numeric vector of integers or \[\]> tensorprod (1, 1, 1, "bar")
%!error <unknown option "foo"> tensorprod (1, 1, 1, "foo", 1)
%!error <misplaced "all" option> tensorprod (1, 1, 1, "all", 1)
%!error <misplaced "NumDimensionsA" option> tensorprod (1, 1, "NumDimensionsA", 1, 1)
%!error <optional arguments must be numeric vectors of integers, \[\], "all", or "NumDimensionsA"> tensorprod (1, 1, 1, {}, 1)
%!error <matched dimension\(s\) of A and B must have the same length\(s\)> tensorprod (ones (3, 4), ones (4, 3), 1)
%!error <dimension\(s\) must be positive integer\(s\)> tensorprod (1, 1, 0)
%!error <dimension\(s\) must be positive integer\(s\)> tensorprod (1, 1, -1)
%!error <dimension\(s\) must be positive integer\(s\)> tensorprod (1, 1, 1.5)
%!error <dimension\(s\) must be positive integer\(s\)> tensorprod (1, 1, NaN)
%!error <dimension\(s\) must be positive integer\(s\)> tensorprod (1, 1, Inf)
%!error <third argument must be a numeric vector of integers, \[\], or "all"> tensorprod (1, 1, {})
%!error <dim must be a numeric vector of integers or \[\]> tensorprod (1, 1, zeros(0,0,0))
%!error <dimA must be a numeric vector of integers or \[\]> tensorprod (1, 1, zeros(0,0,0), [])
%!error <dimB must be a numeric vector of integers or \[\]> tensorprod (1, 1, [], zeros(0,0,0))
%!error <matched dimension\(s\) of A and B must have the same length\(s\)> tensorprod (ones (3, 4), ones (4, 3), 1, 1)
%!error <an equal number of dimensions must be matched for A and B> tensorprod (ones (3, 4), ones (4, 3), 1, [1, 2])
%!error <size of A and B must be identical when using the "all" option> tensorprod (ones (3, 4), ones (4, 3), "all")
%!error <a value for the NumDimensionsA property must be provided> tensorprod (1, 1, "NumDimensionsA")
%!error <NumDimensionsA cannot be smaller than the number of dimensions of A> tensorprod (ones (2, 2, 2), 1, "NumDimensionsA", 2)
%!error <highest dimension of dim must be less than or equal to NumDimensionsA> tensorprod (1, 1, 5, "NumDimensionsA", 4)
%!error <highest dimension of dimA must be less than or equal to NumDimensionsA> tensorprod (1, 1, 5, 5, "NumDimensionsA", 4)
%!error <NumDimensionsA=ndimsA syntax is not yet supported in Octave - provide the value as a property-value pair> tensorprod (1, 1, NumDimensionsA=4)
%!error <NumDimensionsA=ndimsA syntax is not yet supported in Octave - provide the value as a property-value pair> tensorprod (1, 1, numdimensionsa=4)
%!error <too many dimension inputs given> tensorprod (1, 1, 2, 1, 1)
%!error <too many dimension inputs given> tensorprod (1, 1, 2, 1, 1, 1)
%!error <too many inputs given> tensorprod (1, 1, 2, 1, 1, 1, 1)
%!error <value for NumDimensionsA must be a numeric scalar> tensorprod (1, 1, 2, 1, "NumDimensionsA", "foo")
%!error <value for NumDimensionsA must be a numeric scalar> tensorprod (1, 1, 2, 1, "NumDimensionsA", {})
%!error <value for NumDimensionsA must be a positive integer> tensorprod (1, 1, 2, 1, "NumDimensionsA", -1)
%!error <value for NumDimensionsA must be a positive integer> tensorprod (1, 1, 2, 1, "NumDimensionsA", 0)
%!error <value for NumDimensionsA must be a positive integer> tensorprod (1, 1, 2, 1, "NumDimensionsA", 1.5)
%!error <value for NumDimensionsA must be a positive integer> tensorprod (1, 1, 2, 1, "NumDimensionsA", NaN)
%!error <value for NumDimensionsA must be a positive integer> tensorprod (1, 1, 2, 1, "NumDimensionsA", Inf)
