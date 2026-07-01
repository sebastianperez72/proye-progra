#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_INTERFACES_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_INTERFACES_H
#include <memory>
#include <unordered_map>
#include <string>
#include "utec/algebra/tensor_backend.h"
#include "nn_optimizer.h"

namespace utec::tf {

    class Layer {
    public:
        virtual ~Layer() = default;

        virtual void build(const Shape& input_shape) = 0;

        virtual Tensor<float> forward(const Tensor<float>& x) = 0;

        virtual Tensor<float> backward(const Tensor<float>& grad) = 0;

        [[nodiscard]] virtual Shape output_shape() const = 0;

        virtual std::unique_ptr<Layer> clone() const = 0;

        [[nodiscard]] virtual std::unordered_map<std::string, Tensor<float>> parameters() const {
            return {};
        }

        [[nodiscard]] virtual std::unordered_map<std::string, Tensor<float>> gradients() const {
            return {};
        }

        virtual void apply_gradients(const optimizers::SGD& optimizer) {
            (void)optimizer;
        }
    };

    namespace layers {
        class Input : public Layer {
            Shape shape_;
        public:
            explicit Input(const Shape& sample_shape) : shape_(sample_shape) {
                if (sample_shape.rank() == 0) {
                    throw std::invalid_argument("El Shape de entrada no puede estar vacio");
                }
            }

            void build(const Shape& input_shape) override {}

            [[nodiscard]] Shape output_shape() const override {
                return shape_;
            }

            Tensor<float> forward(const Tensor<float>& x) override {
                return x;
            }

            Tensor<float> backward(const Tensor<float>& grad) override {
                return grad;
            }

            std::unique_ptr<Layer> clone() const override {
                return std::make_unique<Input>(shape_);
            }
        };
    }
}

#endif
