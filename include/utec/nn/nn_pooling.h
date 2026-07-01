#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_POOLING_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_POOLING_H

#include "nn_interfaces.h"
#include <array>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <unordered_map>

namespace utec::tf::layers {

    class MaxPooling2D : public Layer {
        std::array<int, 2> pool_size_;
        std::array<int, 2> strides_;
        Shape input_shape_{1,1,1};
        Shape output_shape_{1,1,1};
        bool is_built_ = false;

        Tensor<float> cached_input_{Shape{1,1,1,1}};
        bool has_cache_ = false;

    public:
        explicit MaxPooling2D(std::array<int, 2> pool_size)
            : pool_size_(pool_size), strides_(pool_size) {
            if (pool_size_[0] <= 0 || pool_size_[1] <= 0) {
                throw std::invalid_argument("El tamano de la ventana debe ser mayor a 0");
            }
        }

        void build(const Shape& input_shape) override {
            if (input_shape.rank() != 3) {
                throw std::invalid_argument("MaxPooling2D requiere input_shape de rango 3");
            }

            input_shape_ = input_shape;
            int in_h = input_shape[0];
            int in_w = input_shape[1];
            int channels = input_shape[2];

            if (in_h % pool_size_[0] != 0 || in_w % pool_size_[1] != 0) {
                throw std::invalid_argument("Division inexacta");
            }

            int out_h = in_h / pool_size_[0];
            int out_w = in_w / pool_size_[1];

            output_shape_ = Shape{out_h, out_w, channels};
            is_built_ = true;
        }

        Tensor<float> forward(const Tensor<float>& x) override {
            if (!is_built_) {
                throw std::runtime_error("no has llamado a build antes de forward");
            }
            if (x.rank() != 4) {
                throw std::invalid_argument("MaxPooling2D requiere rango 4");
            }

            int batch = x.shape()[0];
            int in_h = x.shape()[1];
            int in_w = x.shape()[2];
            int channels = x.shape()[3];

            if (in_h != input_shape_[0] || in_w != input_shape_[1] || channels != input_shape_[2]) {
                throw std::invalid_argument("Dimensiones espaciales o canales incompatibles en forward");
            }

            int pool_h = pool_size_[0];
            int pool_w = pool_size_[1];
            int stride_h = strides_[0];
            int stride_w = strides_[1];

            int out_h = output_shape_[0];
            int out_w = output_shape_[1];

            auto out = Tensor<float>::zeros(Shape{batch, out_h, out_w, channels});

            for (int n = 0; n < batch; ++n) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        for (int c = 0; c < channels; ++c) {

                            float best = x(n, oh * stride_h, ow * stride_w, c);

                            for (int kh = 0; kh < pool_h; ++kh) {
                                for (int kw = 0; kw < pool_w; ++kw) {
                                    int ih = oh * stride_h + kh;
                                    int iw = ow * stride_w + kw;
                                    best = std::max(best, x(n, ih, iw, c));
                                }
                            }
                            out(n, oh, ow, c) = best;
                        }
                    }
                }
            }

            cached_input_ = x;
            has_cache_ = true;

            return out;
        }

        Tensor<float> backward(const Tensor<float>& grad) override {
            if (!has_cache_) {
                throw std::logic_error("MaxPooling2D::backward requiere un forward previo");
            }

            const int batch    = cached_input_.shape()[0];
            const int channels = cached_input_.shape()[3];
            const int pool_h   = pool_size_[0];
            const int pool_w   = pool_size_[1];
            const int stride_h = strides_[0];
            const int stride_w = strides_[1];
            const int out_h    = output_shape_[0];
            const int out_w    = output_shape_[1];

            Tensor<float> dx = Tensor<float>::zeros(cached_input_.shape());

            for (int n = 0; n < batch; ++n) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        for (int c = 0; c < channels; ++c) {
                            int best_ih = oh * stride_h;
                            int best_iw = ow * stride_w;
                            float best = cached_input_(n, best_ih, best_iw, c);

                            for (int kh = 0; kh < pool_h; ++kh) {
                                for (int kw = 0; kw < pool_w; ++kw) {
                                    int ih = oh * stride_h + kh;
                                    int iw = ow * stride_w + kw;
                                    float v = cached_input_(n, ih, iw, c);
                                    if (v > best) {
                                        best = v;
                                        best_ih = ih;
                                        best_iw = iw;
                                    }
                                }
                            }

                            dx(n, best_ih, best_iw, c) = dx(n, best_ih, best_iw, c) + grad(n, oh, ow, c);
                        }
                    }
                }
            }

            return dx;
        }

        [[nodiscard]] Shape output_shape() const override {
            return output_shape_;
        }

        [[nodiscard]] std::unordered_map<std::string, Tensor<float>> parameters() const override {
            return {};
        }

        std::unique_ptr<Layer> clone() const override {
            auto copy = std::make_unique<MaxPooling2D>(pool_size_);
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
