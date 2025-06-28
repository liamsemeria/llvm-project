
visualize-opt adds visualization passes between each pass in a given pass-pipeline. It can be useful to quickly see how different passes and orders passes change the dialects and number of instructions.


For example try
build/bin/visualize-opt --pass-pipeline="builtin.module(eliminate-empty-tensors,one-shot-bufferize,func.func(buffer-hoisting),buffer-deallocation-pipeline,convert-linalg-to-affine-loops,finalize-memref-to-llvm)" mlir/test/Examples/myExamples/fc_relu.mlir
compared to
build/bin/visualize-opt --pass-pipeline="builtin.module(eliminate-empty-tensors,one-shot-bufferize,func.func(buffer-hoisting),buffer-deallocation-pipeline,finalize-memref-to-llvm,convert-linalg-to-affine-loops)" mlir/test/Examples/myExamples/fc_relu.mlir

