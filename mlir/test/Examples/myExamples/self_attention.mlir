  // used ai to help make this so not 100% on its accuracy, still had to fix it a bunch but u never know
module {

    func.func @attention(%query: tensor<128x64xf32>,%key: tensor<128x64xf32>, %value: tensor<128x64xf32>) -> tensor<128x64xf32> {

    // Calculate Q * K^T (dot product)
    %0 = tensor.empty() : tensor<64x128xf32>
    %q_transpose = linalg.transpose ins(%key: tensor<128x64xf32>) outs(%0: tensor<64x128xf32>) permutation = [1, 0]

    %1 = tensor.empty() : tensor<128x128xf32>
    %qk_matmul = linalg.matmul ins(%query, %q_transpose: tensor<128x64xf32>, tensor<64x128xf32>)
                outs(%1: tensor<128x128xf32>) -> tensor<128x128xf32>

  // Apply Softmax
  %soft_max = linalg.softmax dimension(0) ins(%qk_matmul : tensor<128x128xf32>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>

  // Calculate weighted sum (V * attention_weights)
  %3 = tensor.empty() : tensor<128x64xf32>
  %attention_output = linalg.matmul ins(%soft_max, %value: tensor<128x128xf32>, tensor<128x64xf32>)
                       outs(%3: tensor<128x64xf32>) -> tensor<128x64xf32>
        func.return %attention_output : tensor<128x64xf32>
    }
}