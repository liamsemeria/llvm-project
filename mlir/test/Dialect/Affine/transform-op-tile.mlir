// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["affine.for"]}
        attributes {tile}
        in %root : (!transform.any_op) -> !transform.op<"affine.for">
    %points, %tiles = transform.affine.tile %target
        dimensions [1, 0, 0, 2] tile_sizes [32, 64, 8, 4]
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %ip, %jp, %kp = transform.split_handle %points
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">, !transform.op<"affine.for">,
              !transform.op<"affine.for">)
    %jt, %it0, %it1, %kt = transform.split_handle %tiles
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">, !transform.op<"affine.for">,
              !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.debug.emit_remark_at %ip, "point i"
        : !transform.op<"affine.for">
    transform.debug.emit_remark_at %jp, "point j"
        : !transform.op<"affine.for">
    transform.debug.emit_remark_at %kp, "point k"
        : !transform.op<"affine.for">
    transform.debug.emit_remark_at %jt, "tile j"
        : !transform.op<"affine.for">
    transform.debug.emit_remark_at %it0, "tile i level 0"
        : !transform.op<"affine.for">
    transform.debug.emit_remark_at %it1, "tile i level 1"
        : !transform.op<"affine.for">
    transform.debug.emit_remark_at %kt, "tile k"
        : !transform.op<"affine.for">
    transform.yield
  }
}

// CHECK-LABEL: func.func @ordered_multilevel
func.func @ordered_multilevel(%input: memref<128x64x32xf32>,
                              %output: memref<128x64x32xf32>) {
  // CHECK: affine.for %[[JT:.*]] = 0 to 64 step 32 {
  // CHECK:   affine.for %[[IT0:.*]] = 0 to 128 step 64 {
  // CHECK:     affine.for %[[IT1:.*]] = {{.*}}(%[[IT0]]) to {{.*}}(%[[IT0]]) step 8 {
  // CHECK:       affine.for %[[KT:.*]] = 0 to 32 step 4 {
  // CHECK:         affine.for %[[I:.*]] = {{.*}}(%[[IT1]]) to {{.*}}(%[[IT1]]) {
  // CHECK:           affine.for %[[J:.*]] = {{.*}}(%[[JT]]) to {{.*}}(%[[JT]]) {
  // CHECK:             affine.for %[[K:.*]] = {{.*}}(%[[KT]]) to {{.*}}(%[[KT]]) {
  // CHECK:               affine.load %{{.*}}[%[[I]], %[[J]], %[[K]]]
  // expected-remark@+3 {{point i}}
  // expected-remark@+2 {{tile i level 0}}
  // expected-remark@+1 {{tile i level 1}}
  affine.for %i = 0 to 128 {
    // expected-remark@+2 {{point j}}
    // expected-remark@+1 {{tile j}}
    affine.for %j = 0 to 64 {
      // expected-remark@+2 {{point k}}
      // expected-remark@+1 {{tile k}}
      affine.for %k = 0 to 32 {
        %value = affine.load %input[%i, %j, %k]
            : memref<128x64x32xf32>
        affine.store %value, %output[%i, %j, %k]
            : memref<128x64x32xf32>
      }
    }
  } {tile}
  return
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["affine.for"]}
        attributes {tile}
        in %root : (!transform.any_op) -> !transform.op<"affine.for">
    %points, %tiles = transform.affine.tile %target
        dimensions [] tile_sizes []
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.yield
  }
}

// CHECK-LABEL: func.func @empty_plan
// CHECK: affine.for %{{.*}} = 0 to 10
// CHECK-NOT: affine.for %
func.func @empty_plan(%output: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %zero = arith.constant 0.0 : f32
    affine.store %zero, %output[%i] : memref<10xf32>
  } {tile}
  return
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["affine.for"]}
        attributes {tile}
        in %root : (!transform.any_op) -> !transform.op<"affine.for">
    %points, %tiles = transform.affine.tile %target
        dimensions [0] tile_sizes [4]
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.loop.unroll %points {factor = 4 : i64}
        : !transform.op<"affine.for">
    transform.yield
  }
}

// CHECK-LABEL: func.func @tile_then_unroll
func.func @tile_then_unroll() {
  // CHECK: affine.for %[[TILE:.*]] = 0 to 64 step 4 {
  // CHECK-NOT: affine.for %
  // CHECK-COUNT-4: arith.addi
  affine.for %i = 0 to 64 {
    %unused = arith.addi %i, %i : index
  } {tile}
  return
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["affine.for"]}
        attributes {tile}
        in %root : (!transform.any_op) -> !transform.op<"affine.for">
    %points, %tiles = transform.affine.tile %target
        dimensions [0, 0] tile_sizes [10, 6]
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.yield
  }
}

// CHECK-LABEL: func.func @non_dividing_tile_levels
func.func @non_dividing_tile_levels() {
  // CHECK: affine.for %[[T0:.*]] = 0 to 64 step 10 {
  // CHECK:   affine.for %[[T1:.*]] = {{.*}}(%[[T0]]) to min {{.*}}(%[[T0]]) step 6 {
  // CHECK:     affine.for %[[P:.*]] = {{.*}}(%[[T1]]) to min {{.*}}(%[[T0]], %[[T1]]) {
  // CHECK:       arith.addi %[[P]], %[[P]]
  affine.for %i = 0 to 64 {
    %unused = arith.addi %i, %i : index
  } {tile}
  return
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["affine.for"]}
        attributes {tile}
        in %root : (!transform.any_op) -> !transform.op<"affine.for">
    %points, %tiles = transform.affine.tile %target
        dimensions [0, 1, 2] tile_sizes [1, 1, 1]
        point_dimensions [0, 2, 1]
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.yield
  }
}

// CHECK-LABEL: func.func @scheduled_point_order
func.func @scheduled_point_order() {
  // CHECK: affine.for %{{.*}} = 0 to 4 {
  // CHECK:   affine.for %{{.*}} = 0 to 8 {
  // CHECK:     affine.for %{{.*}} = 0 to 16 {
  // CHECK:       affine.for %[[I:.*]] = {{.*}} {
  // CHECK:         affine.for %[[K:.*]] = {{.*}} {
  // CHECK:           affine.for %[[J:.*]] = {{.*}} {
  // CHECK:             arith.addi %[[I]], %[[K]]
  // CHECK:             arith.addi %{{.*}}, %[[J]]
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      affine.for %k = 0 to 16 {
        %ik = arith.addi %i, %k : index
        %ijk = arith.addi %ik, %j : index
      }
    }
  } {tile}
  return
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["affine.for"]}
        attributes {tile}
        in %root : (!transform.any_op) -> !transform.op<"affine.for">
    // expected-error @below {{expected dimensions and tile sizes to have equal lengths, found 2 and 1}}
    %points, %tiles = transform.affine.tile %target
        dimensions [0, 1] tile_sizes [4]
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.yield
  }
}

func.func @mismatched_plan(%output: memref<10xf32>) {
  // expected-note @below {{target payload op}}
  affine.for %i = 0 to 10 { } {tile}
  return
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["affine.for"]}
        attributes {tile}
        in %root : (!transform.any_op) -> !transform.op<"affine.for">
    // expected-error @below {{dimension 2 is out of range for an affine loop nest of depth 1}}
    %points, %tiles = transform.affine.tile %target
        dimensions [2] tile_sizes [4]
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.yield
  }
}

func.func @out_of_range_dimension() {
  // expected-note @below {{target payload op}}
  affine.for %i = 0 to 10 { } {tile}
  return
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["affine.for"]}
        attributes {tile}
        in %root : (!transform.any_op) -> !transform.op<"affine.for">
    // expected-error @below {{expected positive tile sizes representable as unsigned}}
    %points, %tiles = transform.affine.tile %target
        dimensions [0] tile_sizes [0]
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.yield
  }
}

func.func @zero_tile_size() {
  // expected-note @below {{target payload op}}
  affine.for %i = 0 to 10 { } {tile}
  return
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["affine.for"]}
        attributes {tile}
        in %root : (!transform.any_op) -> !transform.op<"affine.for">
    // expected-error @below {{affine loop nest is not legal to tile in the requested dimension order}}
    %points, %tiles = transform.affine.tile %target
        dimensions [1] tile_sizes [4]
        : (!transform.op<"affine.for">)
          -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.yield
  }
}

func.func @illegal_order(%memref: memref<64x64xf32>) {
  // expected-note @below {{target payload op}}
  affine.for %i = 0 to 63 {
    affine.for %j = 1 to 64 {
      %value = affine.load %memref[%i, %j] : memref<64x64xf32>
      affine.store %value, %memref[%i + 1, %j - 1] : memref<64x64xf32>
    }
  } {tile}
  return
}
