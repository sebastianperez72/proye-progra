#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_LOSS_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_LOSS_H

#include "utec/algebra/tensor_backend.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace utec::tf::losses {

    class CategoricalCrossentropy {
    public:
        float operator()(const Tensor<float>& y_true, const Tensor<float>& y_pred) const {
            if (y_true.shape() != y_pred.shape()) {
                throw std::invalid_argument("Shapes incompatibles en CategoricalCrossentropy");
            }

            int numel = y_true.shape().numel();
            int batch_size = y_true.shape().rank() >= 2 ? y_true.shape()[0] : 1;
            float loss = 0.0f;

            for (int i = 0; i < numel; ++i) {
                if (y_true[i] > 0.5f) {
                    float prob = std::max(y_pred[i], 1e-7f);
                    loss -= std::log(prob);
                }
            }

            return loss / static_cast<float>(batch_size);
        }

        Tensor<float> gradient(const Tensor<float>& y_true, const Tensor<float>& y_pred) const {
            if (y_true.shape() != y_pred.shape()) {
                throw std::invalid_argument("Shapes incompatibles en CategoricalCrossentropy::gradient");
            }

            int numel = y_true.shape().numel();
            int batch_size = y_true.shape().rank() >= 2 ? y_true.shape()[0] : 1;

            Tensor<float> grad(y_true.shape());
            for (int i = 0; i < numel; ++i) {
                grad[i] = (y_pred[i] - y_true[i]) / static_cast<float>(batch_size);
            }
            return grad;
        }
    };

}

#endif
