// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %bands = transform.structured.match ops{["affine.for"]}
        attributes {point_root}
        in %root : (!transform.any_op) -> !transform.op<"affine.for">
    %selected = transform.structured.match ops{["affine.for"]}
        attributes {vectorize}
        in %root : (!transform.any_op) -> !transform.op<"affine.for">
    transform.affine.vectorize %bands loops %selected vector_sizes [64]
        : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> ()
    transform.yield
  }
}

// CHECK-LABEL: func.func @vectorize_selected_j
func.func @vectorize_selected_j(
    %A: memref<1x1xf32>, %B: memref<1x64xf32>,
    %C: memref<1x64xf32>) {
  // CHECK-NOT: affine.for %
  // CHECK: vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {{.*}} : memref<1x64xf32>, vector<64xf32>
  // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] {{.*}} : vector<64xf32>, memref<1x64xf32>
  affine.for %i = 0 to 1 {
    affine.for %k = 0 to 1 {
      affine.for %j = 0 to 64 {
        %a = affine.load %A[%i, %k] : memref<1x1xf32>
        %b = affine.load %B[%k, %j] : memref<1x64xf32>
        %c = affine.load %C[%i, %j] : memref<1x64xf32>
        %product = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %product : f32
        affine.store %sum, %C[%i, %j] : memref<1x64xf32>
      } {vectorize}
    }
  } {point_root}
  return
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %bands = transform.structured.match ops{["affine.for"]}
        attributes {point_root}
        in %root : (!transform.any_op) -> !transform.op<"affine.for">
    %selected = transform.structured.match ops{["affine.for"]}
        attributes {vectorize}
        in %root : (!transform.any_op) -> !transform.op<"affine.for">
    transform.affine.vectorize %bands loops %selected
        : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> ()
    transform.yield
  }
}

// CHECK-LABEL: func.func @vectorize_multiple_bands
func.func @vectorize_multiple_bands(
    %input: memref<2x8xf32>, %output: memref<2x8xf32>) {
  // CHECK-COUNT-2: vector.transfer_write
  affine.for %i = 0 to 1 {
    affine.for %j = 0 to 8 {
      %value = affine.load %input[0, %j] : memref<2x8xf32>
      affine.store %value, %output[0, %j] : memref<2x8xf32>
    } {vectorize}
  } {point_root}
  affine.for %i = 0 to 1 {
    affine.for %j = 0 to 8 {
      %value = affine.load %input[1, %j] : memref<2x8xf32>
      affine.store %value, %output[1, %j] : memref<2x8xf32>
    } {vectorize}
  } {point_root}
  return
}
