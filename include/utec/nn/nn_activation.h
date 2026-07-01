#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_ACTIVATION_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_ACTIVATION_H

#include "utec/algebra/tensor_backend.h"
#include <cmath>
#include <algorithm>

namespace utec::tf {

    enum class Activation {
        Linear,
        Relu,
        Softmax
    };

    inline Tensor<float> apply_activation(const Tensor<float>& logits, Activation act) {
        Tensor<float> result(logits.shape());
        int numel = logits.shape().numel();

        if (act == Activation::Linear) {
            for (int i = 0; i < numel; ++i) {
                result[i] = logits[i];
            }
        }
        else if (act == Activation::Relu) {
            for (int i = 0; i < numel; ++i) {
                result[i] = std::max(0.0f, logits[i]);
            }
        }
        else if (act == Activation::Softmax) {
            if (logits.rank() != 2) {
                throw std::invalid_argument("Softmax requiere un tensor de rango 2 (batch_size, num_classes)");
            }
            int rows = logits.shape().rank() >= 2 ? logits.shape()[0] : 1;
            int cols = numel / rows;

            for (int i = 0; i < rows; ++i) {
                float max_val = logits[i * cols];
                for (int j = 1; j < cols; ++j) {
                    max_val = std::max(max_val, logits[i * cols + j]);
                }

                float sum = 0.0f;
                for (int j = 0; j < cols; ++j) {
                    result[i * cols + j] = std::exp(logits[i * cols + j] - max_val);
                    sum += result[i * cols + j];
                }

                for (int j = 0; j < cols; ++j) {
                    result[i * cols + j] /= sum;
                }
            }
        }
        return result;
    }

    inline Tensor<float> activation_gradient(const Tensor<float>& grad,
                                             const Tensor<float>& pre_activation,
                                             Activation act) {
        Tensor<float> dz(grad.shape());
        int numel = grad.shape().numel();
        for (int i = 0; i < numel; ++i) {
            if (act == Activation::Relu) {
                dz[i] = pre_activation[i] > 0.0f ? grad[i] : 0.0f;
            } else {
                dz[i] = grad[i];
            }
        }
        return dz;
    }

}

#endif
