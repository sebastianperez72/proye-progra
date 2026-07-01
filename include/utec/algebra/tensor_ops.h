#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_1_TENSOR_OPS_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_1_TENSOR_OPS_H

#include "tensor_backend.h"

namespace utec::tf::ops {

    template <typename T>
    Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Shapes incompatibles para multiply");
        }
        return a * b;
    }


    template <typename T>
    Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {

        if (a.rank() != 2 || b.rank() != 2) {
            throw std::invalid_argument("usar tensores de rango 2");
        }
        if (a.shape()[1] != b.shape()[0]) {
            throw std::invalid_argument("Dimensiones internas incompatibles para matmul");
        }
        int aRow = a.shape()[0];
        int aCol = a.shape()[1];
        int bCol = b.shape()[1];

        Tensor<T> result(Shape{aRow, bCol});
        for (int i = 0; i < aRow; ++i) {
            for (int j = 0; j < bCol; ++j) {
                T sum = 0;
                for (int k = 0; k < aCol; ++k) {
                    sum += a(i, k) * b(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    template <typename T>
    Tensor<T> transpose2d(const Tensor<T>& mat) {
        if (mat.rank() != 2) {
            throw std::invalid_argument("usa un tensor de rango 2");
        }
        int rows = mat.shape()[0];
        int cols = mat.shape()[1];

        Tensor<T> result(Shape{cols, rows});

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(j, i) = mat(i, j);
            }
        }
        return result;
    }
}

#endif
