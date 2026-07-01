#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_OPTIMIZER_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_OPTIMIZER_H

#include "utec/algebra/tensor_backend.h"
#include <stdexcept>

namespace utec::tf::optimizers {

    struct SGD {
        float learning_rate = 0.01f;

        SGD() = default;

        explicit SGD(float lr) : learning_rate(lr) {
            if (lr <= 0.0f) {
                throw std::invalid_argument("learning_rate debe ser mayor a 0");
            }
        }

        void update(Tensor<float>& param, const Tensor<float>& grad) const {
            if (param.shape() != grad.shape()) {
                throw std::invalid_argument("Shapes incompatibles en SGD::update");
            }
            int numel = param.shape().numel();
            for (int i = 0; i < numel; ++i) {
                param[i] = param[i] - learning_rate * grad[i];
            }
        }
    };

}

#endif
