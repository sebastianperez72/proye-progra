#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_CONVOLUTION_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_CONVOLUTION_H

#include "nn_interfaces.h"
#include "nn_activation.h"
#include "nn_ops.h"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include "utec/algebra/tensor_backend.h"

namespace utec::tf::layers {

    class Conv2D : public Layer {
        int filters_;
        std::pair<int, int> kernel_size_;
        Activation activation_;
        Shape input_shape_{1, 1, 1};
        Shape output_shape_{1, 1, 1};
        Tensor<float> weights_{Shape{1, 1, 1}};
        Tensor<float> bias_{Shape{1}};
        bool is_built_ = false;

        Tensor<float> cached_input_{Shape{1, 1, 1, 1}};
        Tensor<float> cached_pre_activation_{Shape{1, 1, 1, 1}};
        bool has_cache_ = false;
        Tensor<float> grad_weights_{Shape{1, 1, 1, 1}};
        Tensor<float> grad_bias_{Shape{1}};
        bool has_grad_ = false;

    public:
        explicit Conv2D(int filters, std::pair<int, int> kernel_size, Activation activation = Activation::Linear)
            : filters_(filters), kernel_size_(kernel_size), activation_(activation) {
            if (filters <= 0) {
                throw std::invalid_argument("La cantidad de filtros debe ser mayor a 0");
            }
            if (kernel_size.first <= 0 || kernel_size.second <= 0) {
                throw std::invalid_argument("El tamano del kernel debe ser mayor a 0");
            }
        }

        void build(const Shape& input_shape) override {
            if (input_shape.rank() != 3) {
                throw std::invalid_argument("Conv2D requiere un input_shape de rango 3 (in_h, in_w, in_c)");
            }

            input_shape_ = input_shape;
            int in_h = input_shape[0];
            int in_w = input_shape[1];
            int in_c = input_shape[2];

            int k_h = kernel_size_.first;
            int k_w = kernel_size_.second;

            if (k_h > in_h || k_w > in_w) {
                throw std::invalid_argument("El kernel no puede ser mas grande que la entrada");
            }

            int out_h = in_h - k_h + 1;
            int out_w = in_w - k_w + 1;
            output_shape_ = Shape{out_h, out_w, filters_};

            weights_ = Tensor<float>(Shape{k_h, k_w, in_c, filters_});
            bias_ = Tensor<float>::zeros(Shape{filters_});

            unsigned int state = 0x85EBCA6Bu + static_cast<unsigned int>(k_h * 1000 + k_w * 100 + in_c * 10 + filters_);
            float scale = 1.0f / static_cast<float>(k_h * k_w * in_c);
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
                throw std::runtime_error("Llamar a build() antes de forward()");
            }
            if (x.rank() != 4) {
                throw std::invalid_argument("Conv2D espera un tensor de entrada de rango 4 (batch, in_h, in_w, in_c)");
            }
            if (x.shape()[3] != input_shape_[2]) {
                throw std::invalid_argument("Los canales de la entrada no coinciden con los parametros");
            }

            Strides strides{1, 1};
            Tensor<float> out = ops::conv2d(x, weights_, strides, Padding::Valid);

            int batch = out.shape()[0];
            int out_h = out.shape()[1];
            int out_w = out.shape()[2];

            for (int n = 0; n < batch; ++n) {
                for (int h = 0; h < out_h; ++h) {
                    for (int w = 0; w < out_w; ++w) {
                        for (int c = 0; c < filters_; ++c) {
                            out(n, h, w, c) = out(n, h, w, c) + bias_[c];
                        }
                    }
                }
            }

            cached_input_ = x;
            cached_pre_activation_ = out;
            has_cache_ = true;

            return apply_activation(out, activation_);
        }

        Tensor<float> backward(const Tensor<float>& grad) override {
            if (!has_cache_) {
                throw std::logic_error("Conv2D::backward requiere un forward previo");
            }

            Tensor<float> dz = activation_gradient(grad, cached_pre_activation_, activation_);

            const int batch = cached_input_.shape()[0];
            const int in_c  = cached_input_.shape()[3];
            const int k_h   = weights_.shape()[0];
            const int k_w   = weights_.shape()[1];
            const int out_c = filters_;
            const int out_h = dz.shape()[1];
            const int out_w = dz.shape()[2];

            grad_weights_ = Tensor<float>::zeros(weights_.shape());
            grad_bias_    = Tensor<float>::zeros(Shape{out_c});
            Tensor<float> dx = Tensor<float>::zeros(cached_input_.shape());

            for (int n = 0; n < batch; ++n) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        for (int oc = 0; oc < out_c; ++oc) {
                            const float g = dz(n, oh, ow, oc);
                            grad_bias_[oc] = grad_bias_[oc] + g;

                            for (int kh = 0; kh < k_h; ++kh) {
                                for (int kw = 0; kw < k_w; ++kw) {
                                    for (int ic = 0; ic < in_c; ++ic) {
                                        const int ih = oh + kh;
                                        const int iw = ow + kw;
                                        grad_weights_(kh, kw, ic, oc) =
                                            grad_weights_(kh, kw, ic, oc) + cached_input_(n, ih, iw, ic) * g;
                                        dx(n, ih, iw, ic) =
                                            dx(n, ih, iw, ic) + weights_(kh, kw, ic, oc) * g;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            has_grad_ = true;
            return dx;
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
            auto copy = std::make_unique<Conv2D>(filters_, kernel_size_, activation_);
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
