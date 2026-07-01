#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_OPS_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_OPS_H

#include "utec/algebra/tensor_ops.h"
#include "utec/algebra/tensor_backend.h"
#include <vector>

namespace utec::tf {

    struct Strides {
        int rows, cols;
    };

    enum class Padding { Valid };

    namespace ops {

        template <typename T>
        Tensor<T> flatten_batch(const Tensor<T>& x) {
            if (x.rank() < 2) {
                throw std::invalid_argument("flatten_batch requiere rank >= 2");
            }
            int batch = x.shape()[0];
            int features = 1;

            for (int i = 1; i < x.rank(); ++i) {
                features *= x.shape()[i];
            }
            Tensor<T> result = x;
            result.reshape(Shape{batch, features});
            return result;
        }

        template <typename T>
        Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& kernel,
            Strides strides = {1, 1}, Padding padding = Padding::Valid) {
            if (padding != Padding::Valid) {
                throw std::invalid_argument("solo se soporta Padding::Valid en este feature");
            }
            if (input.rank() != 4 || kernel.rank() != 4) {
                throw std::invalid_argument("conv2d requiere input y kernel de rank 4");
            }if (strides.rows <= 0 || strides.cols <= 0) {
                throw std::invalid_argument("stride invalido");
            }

            const int batch = input.shape()[0];
            const int in_h = input.shape()[1];
            const int in_w = input.shape()[2];
            const int in_c = input.shape()[3];

            const int k_h = kernel.shape()[0];
            const int k_w = kernel.shape()[1];
            const int k_c = kernel.shape()[2];
            const int out_c = kernel.shape()[3];

            if (in_c != k_c) {
                throw std::invalid_argument("conv2d recibio canales incompatibles");
            }
            if (k_h > in_h || k_w > in_w) {
                throw std::invalid_argument("kernel mas grande que la entrada");
            }


            const int out_h = (in_h - k_h) / strides.rows + 1;
            const int out_w = (in_w - k_w) / strides.cols + 1;

            Tensor<T> out = Tensor<T>::zeros(Shape{batch, out_h, out_w, out_c});


            for (int n = 0; n < batch; ++n) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        for (int oc = 0; oc < out_c; ++oc) {
                            T acc = T{};
                            for (int kh = 0; kh < k_h; ++kh) {
                                for (int kw = 0; kw < k_w; ++kw) {
                                    for (int ic = 0; ic < in_c; ++ic) {
                                        const int ih = oh * strides.rows + kh;
                                        const int iw = ow * strides.cols + kw;
                                        acc += input(n, ih, iw, ic) * kernel(kh, kw, ic, oc);
                                    }
                                }
                            }
                            out(n, oh, ow, oc) = acc;
                        }
                    }
                }
            }
            return out;
        }
    }
}

#endif
