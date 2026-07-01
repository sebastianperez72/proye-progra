#include <iostream>
#include "utec/algebra/shape.h"
#include "utec/algebra/tensor_backend.h"
#include "utec/algebra/tensor_ops.h"
#include "utec/nn/nn_ops.h"

using namespace utec::tf;

int main() {
    std::cout << "--- Iniciando pruebas locales Feature 2 ---\n";

    Shape a{2, 3, 4};
    Shape b(std::vector<int>{2, 3, 4});
    std::cout << "Shape equality: " << std::boolalpha << (a == b) << " (Esperado: true)\n";

    auto left = Tensor<int>::from_data(Shape{2, 2}, {1, 2, 3, 4});
    auto right = Tensor<int>::from_data(Shape{2, 2}, {4, 3, 2, 1});
    auto diff = left - right;
    auto sum = left + right;
    std::cout << "Resta diff(0, 0): " << diff(0, 0) << " (Esperado: -3)\n";
    std::cout << "Suma sum(0, 0): " << sum(0, 0) << " (Esperado: 5)\n";

    auto m1 = Tensor<int>::from_data(Shape{2, 3}, {1, 2, 3, 4, 5, 6});
    auto m2 = Tensor<int>::from_data(Shape{3, 2}, {7, 8, 9, 10, 11, 12});
    auto m3 = algebra::matmul(m1, m2);
    std::cout << "Matmul m3(0, 0): " << m3(0, 0) << " (Esperado: 58)\n";

    auto x = Tensor<float>::ones(Shape{2, 3, 3, 1});
    auto flat = ops::flatten_batch(x);
    std::cout << "Flatten shape[1]: " << flat.shape()[1] << " (Esperado: 9)\n";

    auto input = Tensor<float>::ones(Shape{1, 4, 4, 1});
    auto kernel = Tensor<float>::ones(Shape{2, 2, 1, 1});
    auto out = ops::conv2d(input, kernel, Strides{1, 1}, Padding::Valid);
    std::cout << "Conv2D out shape[1]: " << out.shape()[1] << " (Esperado: 3)\n";

    std::cout << "--- Todas las pruebas finalizaron exitosamente ---\n";
    return 0;
}