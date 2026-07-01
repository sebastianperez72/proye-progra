#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_FLATTEN_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_FLATTEN_H

#include "nn_interfaces.h"
#include "nn_ops.h"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "utec/algebra/tensor_backend.h"

namespace utec::tf::layers {

    class Flatten : public Layer {
        Shape input_shape_{1};
        Shape output_shape_{1};
        bool is_built_ = false;

        Shape cached_input_shape_{1};
        bool has_cache_ = false;

    public:
        Flatten() = default;

        void build(const Shape& input_shape) override {
            input_shape_ = input_shape;

            output_shape_ = Shape{input_shape.numel()};
            is_built_ = true;
        }

        Tensor<float> forward(const Tensor<float>& x) override {
            if (!is_built_) {
                throw std::runtime_error("Llamar a build() antes de forward()");
            }
            cached_input_shape_ = x.shape();
            has_cache_ = true;
            return ops::flatten_batch(x);
        }

        Tensor<float> backward(const Tensor<float>& grad) override {
            if (!has_cache_) {
                throw std::logic_error("Flatten::backward requiere un forward previo");
            }
            return grad.reshaped(cached_input_shape_);
        }

        [[nodiscard]] Shape output_shape() const override {
            return output_shape_;
        }

        [[nodiscard]] std::unordered_map<std::string, Tensor<float>> parameters() const override {
            return {};
        }

        std::unique_ptr<Layer> clone() const override {
            auto copy = std::make_unique<Flatten>();
            if (is_built_) {
                copy->input_shape_ = input_shape_;
                copy->output_shape_ = output_shape_;
                copy->is_built_ = is_built_;
            }
            return copy;
        }
    };

}

#endif
