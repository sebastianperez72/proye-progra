#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_DENSE_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_DENSE_H

#include "nn_interfaces.h"
#include "nn_activation.h"
#include "utec/algebra/tensor_ops.h"
#include "utec/algebra/tensor_backend.h"
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace utec::tf::layers {

    class Dense : public Layer {
        int units_;
        Activation activation_;
        Shape input_shape_{1};
        Shape output_shape_{1};
        Tensor<float> weights_{Shape{1,1}};
        Tensor<float> bias_{Shape{1}};
        bool is_built_ = false;

        Tensor<float> cached_input_{Shape{1,1}};
        Tensor<float> cached_pre_activation_{Shape{1,1}};
        bool has_cache_ = false;
        Tensor<float> grad_weights_{Shape{1,1}};
        Tensor<float> grad_bias_{Shape{1}};
        bool has_grad_ = false;

    public:
        explicit Dense(int units, Activation activation = Activation::Linear)
            : units_(units), activation_(activation) {
            if (units <= 0) {
                throw std::invalid_argument("La cantidad de units debe ser mayor a 0");
            }
        }

        void build(const Shape& input_shape) override {
            if (input_shape.rank() != 1) {
                throw std::invalid_argument("Dense requiere un input_shape de rango 1 (sin batch)");
            }

            input_shape_ = input_shape;
            int in_features = input_shape[0];
            output_shape_ = Shape{units_};

            weights_ = Tensor<float>(Shape{in_features, units_});
            bias_ = Tensor<float>::zeros(Shape{units_});

            unsigned int state = 0x9E3779B9u + static_cast<unsigned int>(in_features * 131 + units_);
            float scale = 1.0f / static_cast<float>(in_features);
            for (int i = 0; i < weights_.size(); ++i) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                float r = static_cast<float>(state & 0xFFFFu) / 65536.0f;
                weights_[i] = (r - 0.5f) * scale;
            }

            is_built_ = true;
        }

        Tensor<float> forward(const Tensor<float>& x) override {
            if (!is_built_) {
                throw std::runtime_error("No se puede hacer forward sin haber llamado a build()");
            }
            if (x.rank() != 2) {
                throw std::invalid_argument("Dense espera un tensor de entrada de rango 2 (Shape{batch, in_features})");
            }
            if (x.shape()[1] != input_shape_[0]) {
                throw std::invalid_argument("Dimensiones de características de entrada incompatibles");
            }

            int batch = x.shape()[0];

            Tensor<float> out = ops::matmul(x, weights_);

            for (int i = 0; i < batch; ++i) {
                for (int j = 0; j < units_; ++j) {
                    out(i, j) = out(i, j) + bias_[j];
                }
            }

            cached_input_ = x;
            cached_pre_activation_ = out;
            has_cache_ = true;

            return apply_activation(out, activation_);
        }

        Tensor<float> backward(const Tensor<float>& grad) override {
            if (!has_cache_) {
                throw std::logic_error("Dense::backward requiere un forward previo");
            }

            Tensor<float> dz = activation_gradient(grad, cached_pre_activation_, activation_);

            int batch = dz.shape()[0];

            grad_weights_ = ops::matmul(ops::transpose2d(cached_input_), dz);

            grad_bias_ = Tensor<float>::zeros(Shape{units_});
            for (int i = 0; i < batch; ++i) {
                for (int j = 0; j < units_; ++j) {
                    grad_bias_[j] = grad_bias_[j] + dz(i, j);
                }
            }

            has_grad_ = true;

            return ops::matmul(dz, ops::transpose2d(weights_));
        }

        [[nodiscard]] Shape output_shape() const override {
            return output_shape_;
        }

        void set_weights(const Tensor<float>& w) {
            if (w.shape() != weights_.shape()) {
                throw std::invalid_argument("set_weights: shape incompatible con la capa");
            }
            weights_ = w;
        }

        void set_bias(const Tensor<float>& b) {
            if (b.shape() != bias_.shape()) {
                throw std::invalid_argument("set_bias: shape incompatible con la capa");
            }
            bias_ = b;
        }

        [[nodiscard]] std::unordered_map<std::string, Tensor<float>> gradients() const override {
            std::unordered_map<std::string, Tensor<float>> grads;
            if (has_grad_) {
                grads.insert({"weights", grad_weights_});
                grads.insert({"bias", grad_bias_});
            }
            return grads;
        }

        void apply_gradients(const optimizers::SGD& optimizer) override {
            if (!has_grad_) {
                return;
            }
            optimizer.update(weights_, grad_weights_);
            optimizer.update(bias_, grad_bias_);
        }

        [[nodiscard]] std::unordered_map<std::string, Tensor<float>> parameters() const override {
            std::unordered_map<std::string, Tensor<float>> params;
            if (is_built_) {
                params.insert({"weights", weights_});
                params.insert({"bias", bias_});
            }
            return params;
        }

        std::unique_ptr<Layer> clone() const override {
            auto copy = std::make_unique<Dense>(units_, activation_);
            if (is_built_) {
                copy->input_shape_ = input_shape_;
                copy->output_shape_ = output_shape_;
                copy->weights_ = weights_;
                copy->bias_ = bias_;
                copy->is_built_ = is_built_;
            }
            return copy;
        }
    };

}

#endif
