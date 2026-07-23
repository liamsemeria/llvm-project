// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %nest = transform.structured.match ops{["affine.for"]} attributes {tile}
      in %root : (!transform.any_op) -> !transform.op<"affine.for">
    %points, %tiles = transform.affine.tile %nest
        dimensions [0] tile_sizes [32]
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %outer = transform.split_handle %tiles
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">)
    %parallel = transform.affine.parallelize %outer
      : (!transform.op<"affine.for">)
        -> !transform.op<"affine.parallel">
    transform.yield
  }
}

// CHECK-LABEL: func.func @tile_then_parallelize(
func.func @tile_then_parallelize(
    %A: memref<256x512xf32>, %B: memref<512x256xf32>,
    %C: memref<256x256xf32>) {
  // CHECK: affine.parallel (%{{.*}}) = (0) to (256) step (32)
  // CHECK:   affine.for
  // CHECK:     affine.for
  // CHECK:       affine.for
  // CHECK:         affine.load
  // CHECK:         affine.load
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      affine.for %k = 0 to 512 {
        %a = affine.load %A[%i, %k] : memref<256x512xf32>
        %b = affine.load %B[%k, %j] : memref<512x256xf32>
        %c = affine.load %C[%i, %j] : memref<256x256xf32>
        %product = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %product : f32
        affine.store %sum, %C[%i, %j] : memref<256x256xf32>
      }
    }
  } {tile}
  return
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["affine.for"]} attributes {target}
      in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{could not prove that the affine.for loop is parallel}}
    transform.affine.parallelize %target
      : (!transform.any_op) -> !transform.op<"affine.parallel">
    transform.yield
  }
}

func.func @reject_loop_carried_dependence(%A: memref<16xf32>) {
  // expected-note @below {{target payload op}}
  affine.for %i = 1 to 16 {
    %value = affine.load %A[%i - 1] : memref<16xf32>
    affine.store %value, %A[%i] : memref<16xf32>
  } {target}
  return
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["affine.for"]} attributes {target}
      in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{affine.for loops with iter arguments are not supported}}
    transform.affine.parallelize %target
      : (!transform.any_op) -> !transform.op<"affine.parallel">
    transform.yield
  }
}

func.func @reject_iter_args(%init: index) -> index {
  // expected-note @below {{target payload op}}
  %result = affine.for %i = 0 to 16 iter_args(%iter = %init) -> index {
    affine.yield %iter : index
  } {target}
  return %result : index
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match attributes {target}
      in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected target to be an affine.for operation}}
    transform.affine.parallelize %target
      : (!transform.any_op) -> !transform.op<"affine.parallel">
    transform.yield
  }
}

func.func @reject_wrong_target() {
  // expected-note @below {{target payload op}}
  %c0 = arith.constant {target} 0 : index
  return
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %targets = transform.structured.match ops{["affine.for"]} attributes {target}
      in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{overlapping target operations}}
    transform.affine.parallelize %targets
      : (!transform.any_op) -> !transform.op<"affine.parallel">
    transform.yield
  }
}

func.func @reject_overlapping_targets(%A: memref<16x16xf32>) {
  // expected-note @below {{second overlapping target}}
  affine.for %i = 0 to 16 {
    // expected-note @below {{first overlapping target}}
    affine.for %j = 0 to 16 {
      %value = affine.load %A[%i, %j] : memref<16x16xf32>
      affine.store %value, %A[%i, %j] : memref<16x16xf32>
    } {target}
  } {target}
  return
}
