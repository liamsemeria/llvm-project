//===- AffineTransformOps.td - Affine transformation ops ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AFFINE_TRANSFORM_OPS
#define AFFINE_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Transform_AffineForOp : Transform_ConcreteOpType<"affine.for">;
def Transform_AffineParallelOp : Transform_ConcreteOpType<"affine.parallel">;

def TileOp
    : Op<Transform_Dialect, "affine.tile",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Tile a perfect affine loop nest";
  let description = [{
    Tiles the maximal perfect `affine.for` nest rooted at the single targeted
    loop. `dimensions` is an ordered list of zero-based dimensions in the
    original nest and `tile_sizes` provides one positive static size for each
    scheduling action. Dimensions may be repeated to create multiple tile
    levels and their order defines the tile-space loop order.

    The optional `point_dimensions` permutation defines the physical nesting
    order of the final point loops. It must contain every source dimension
    exactly once. If omitted, point loops remain in source order.

    The `point_loops` result contains one intra-tile or retained `affine.for`
    per source dimension. Result handles remain in source-dimension order,
    independently of their physical nesting order.
    The `tile_loops` result contains one generated tile-space `affine.for` per
    scheduling action, in scheduling order. Each result is an aggregate typed
    handle and can be decomposed with `transform.split_handle`.

    ```
    %points, %tiles = transform.affine.tile %target
        dimensions [1, 0, 0, 2] tile_sizes [32, 64, 8, 4]
        point_dimensions [0, 2, 1]
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">,
              !transform.op<"affine.for">)
    %i, %j, %k = transform.split_handle %points
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">,
              !transform.op<"affine.for">,
              !transform.op<"affine.for">)
    ```

    All preconditions are checked before modifying the payload. Affine tiling
    requires a perfect nest without loop results, a hyperrectangular iteration
    domain, analyzable memory effects, and a dependence analysis that proves the
    ordered schedule legal. Tile sizes must be positive and the resulting loop
    steps must be representable. Empty dimension and tile-size lists form a
    no-op that returns the original loops as `point_loops` and an empty
    `tile_loops` handle.

    #### Return modes

    The transform consumes the target handle. It produces a silenceable failure
    if the handle does not contain exactly one `affine.for` or if any tiling
    precondition is not satisfied.
  }];

  let arguments = (ins Transform_AffineForOp:$target,
                        DenseI64ArrayAttr:$dimensions,
                        DenseI64ArrayAttr:$tile_sizes,
                        OptionalAttr<DenseI64ArrayAttr>:$point_dimensions);
  let results = (outs Transform_AffineForOp:$point_loops,
                      Transform_AffineForOp:$tile_loops);

  let assemblyFormat = [{
    $target `dimensions` $dimensions `tile_sizes` $tile_sizes
        (`point_dimensions` $point_dimensions^)? attr-dict `:`
        functional-type($target, results)
  }];
}

def AffineVectorizeOp
    : Op<Transform_Dialect, "affine.vectorize",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Vectorize selected loops in affine point-loop bands";
  let description = [{
    Vectorizes only the loops associated with `vectorized_loops`. The `target`
    handle identifies one or more disjoint perfect affine point-loop bands and
    is used to associate selected loops with cloned bands after transformations
    such as unrolling.

    Selected loops must belong to exactly one targeted band and occur in the
    same outer-to-inner order in every band. Unselected loops remain scalar.
    Static single-iteration loops surrounding and nested under selected loops
    are promoted. If `vector_sizes` is omitted, sizes are inferred from the
    selected loops' static trip counts. Explicit sizes must equal those trip
    counts.

    The initial implementation supports fixed-width, unit-step, parallel loops
    without affine reductions. All bands are validated before payload IR is
    modified.

    #### Return modes

    This operation reads the target band and selected-loop handles. It produces
    a silenceable failure if the handles cannot be grouped into equivalent
    perfect nests or if vectorization cannot be proven legal.
  }];

  let arguments = (ins Transform_AffineForOp:$target,
                        Transform_AffineForOp:$vectorized_loops,
                        OptionalAttr<DenseI64ArrayAttr>:$vector_sizes);

  let assemblyFormat = [{
    $target `loops` $vectorized_loops (`vector_sizes` $vector_sizes^)?
        attr-dict `:` functional-type(operands, results)
  }];
}

def ParallelizeOp
    : Op<Transform_Dialect, "affine.parallelize",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Convert independent affine.for loops to affine.parallel";
  let description = [{
    Converts each targeted one-dimensional `affine.for` operation into an
    `affine.parallel` operation with the same bounds and step. The operation is
    intended to make parallel mapping an explicit scheduling decision, for
    example after tiling an affine loop nest:

    ```
    %tiled, %outer = transform.structured.tile_using_for %target
        tile_sizes [32, 0, 0]
    %parallel = transform.affine.parallelize %outer
        : (!transform.op<"affine.for">)
          -> !transform.op<"affine.parallel">
    ```

    All targets are checked before the payload is modified. Each target must be
    an `affine.for` without loop iter arguments and must be proven parallel by
    affine dependence analysis. Duplicate and ancestor/descendant targets are
    rejected because converting one would invalidate the other.

    #### Return modes

    The transform consumes the target handle and returns a handle to the newly
    created `affine.parallel` operations. It emits a silenceable failure if any
    target is not an `affine.for`, has loop iter arguments, cannot be proven
    parallel, or overlaps another target.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs Transform_AffineParallelOp:$parallel);

  let assemblyFormat =
      "$target attr-dict `:` functional-type($target, $parallel)";
}

def SimplifyBoundedAffineOpsOp
    : Op<Transform_Dialect, "affine.simplify_bounded_affine_ops",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Simplify the targeted affine.min / affine.max ops given the supplied
    lower and upper bounds for values that may be used as target op operands.

    Example:
    ```
    %0 = transform.structured.match ops{["affine.min", "affine.max"]} in %arg1
    %1 = transform.structured.match ops{["gpu.lane_id"]} in %arg1
    transform.affine.simplify_bounded_affine_ops %0 with [%1] within [0] and [32]

    // Multiple bounds can be specified.
    transform.affine.simplify_bounded_affine_ops %0 with [%1, %2] within [0, 5] and [32, 50]
    ```

    Bounded op handles (`%1` and `%2) must be mapped to ops that have a single
    result of index type. The sets of target ops and bounded ops must not
    overlap.

    #### Return modes

    Target ops must be affine.min or affine.max ops. This transform consumes the
    target handle and does not produce any handle. It reads the bounded op
    handles.

    TODO: Support affine.apply targets.
    TODO: Allow mixed PDL_Operation/int64_t for lower_bounds and upper_bounds.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       Variadic<TransformHandleTypeInterface>:$bounded_values,
                       DenseI64ArrayAttr:$lower_bounds,
                       DenseI64ArrayAttr:$upper_bounds);
  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
      $target `with` `[` ($bounded_values^ `:` type($bounded_values))? `]`
          `within` $lower_bounds `and` $upper_bounds attr-dict
          `:` type($target)
  }];
}

def SimplifyMinMaxAffineOpsOp :
  Op<Transform_Dialect, "affine.simplify_min_max_affine_ops", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let description = [{
    Simplify the targeted `affine.min` / `affine.max` ops using the
    `mlir::affine::simplifyAffineMinMaxOps` transform.

    Example:
    ```
    %0 = transform.structured.match ops{["affine.max"]} in %arg1
    transform.affine.simplify_min_max_affine_ops %0 : !transform.any_op
    ```

    #### Return modes

    This transform consumes the target handle and does not produce any results.
    This transforms definitely fails if any of the targeted operations is not an
    `affine.min` or `affine.max` operation, or if the canonicalization patterns
    failed to converge.
    This transform silently fails if none of the operations were simplified.
    Otherwise, it succeeds.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = [{
      $target attr-dict `:` type($target)
  }];
}

def SuperVectorizeOp
    : Op<Transform_Dialect, "affine.super_vectorize",
         [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Vectorize to a target independent n-D vector abstraction.
    This operation is an exposition to the transform dialect of the affine-super-vectorize pass.
    To make its usage easier, it ignores inputs which are children of an affine.for op (itself excluded),
    this way the matcher can be simpler.

    This operation may generate operations from the vector dialect.

    Example:
    ```
    %0 = transform.structured.match ops{["affine.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.affine.super_vectorize %0 [8, 16] fastest_varying_pattern=[1,0] vectorize_reductions=true : !transform.any_op
    ```
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       DenseI64ArrayAttr:$vector_sizes,
                       OptionalAttr<DenseI64ArrayAttr>:$fastest_varying_pattern,
                       DefaultValuedAttr<BoolAttr, "false">:$vectorize_reductions);
  let results = (outs);

  let assemblyFormat = [{
      $target $vector_sizes
          (`fastest_varying_pattern` `=` $fastest_varying_pattern^)? 
          (`vectorize_reductions` `=` $vectorize_reductions^)? 
          attr-dict `:` type($target)
  }];
  let hasVerifier = 1;
}

#endif // Affine_TRANSFORM_OPS
