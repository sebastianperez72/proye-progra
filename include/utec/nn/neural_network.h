#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NEURAL_NETWORK_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_optimizer.h"
#include "nn_loss.h"
#include "nn_dense.h"
#include "nn_convolution.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <algorithm>
#include "utec/algebra/tensor_backend.h"
#include "nn_pooling.h"
#include "nn_flatten.h"

namespace utec::tf {

    struct FitOptions {
        int epochs;
        int batch_size;
    };

    struct History {
        std::vector<float> loss;
    };

    struct EvalResult {
        float loss;
    };

    class Sequential {
        std::vector<std::unique_ptr<Layer>> layers_;
        bool is_compiled_ = false;
        optimizers::SGD optimizer_{};
        losses::CategoricalCrossentropy loss_;
        std::unordered_map<std::string, Tensor<float>> last_gradients_;

    public:
        Sequential() = default;

        template <typename LayerT>
        void add(LayerT layer) {
            auto new_layer = layer.clone();

            if (!layers_.empty()) {
                new_layer->build(layers_.back()->output_shape());
            }

            layers_.push_back(std::move(new_layer));
        }

        void compile(optimizers::SGD optimizer, losses::CategoricalCrossentropy loss) {
            optimizer_ = optimizer;
            loss_ = loss;
            is_compiled_ = true;
        }

        bool compiled() const {
            return is_compiled_;
        }

        Tensor<float> predict(const Tensor<float>& x) {
            if (layers_.empty()) {
                throw std::runtime_error("El modelo no tiene capas para predecir");
            }

            Shape expected_shape = layers_.front()->output_shape();
            if (x.rank() != expected_shape.rank() + 1) {
                throw std::invalid_argument("El rango del tensor de entrada no coincide con la capa Input");
            }

            for (int i = 0; i < expected_shape.rank(); ++i) {
                if (x.shape()[i + 1] != expected_shape[i]) {
                    throw std::invalid_argument("Dimensiones de la muestra incompatibles con el Input");
                }
            }

            Tensor<float> current_tensor = x;
            for (const auto& layer : layers_) {
                current_tensor = layer->forward(current_tensor);
            }

            return current_tensor;
        }

        [[nodiscard]] std::unordered_map<std::string, Tensor<float>> parameters() const {
            std::unordered_map<std::string, Tensor<float>> all_params;
            int dense_count = 0;
            int conv2d_count = 0;

            for (const auto& layer : layers_) {
                auto layer_params = layer->parameters();
                if (layer_params.empty()) continue;

                std::string prefix;
                if (dynamic_cast<layers::Dense*>(layer.get())) {
                    prefix = "dense_" + std::to_string(dense_count++) + "/";
                } else if (dynamic_cast<layers::Conv2D*>(layer.get())) {
                    prefix = "conv2d_" + std::to_string(conv2d_count++) + "/";
                } else {
                    prefix = "unknown_layer/";
                }

                for (const auto& [key, tensor] : layer_params) {
                    all_params.insert({prefix + key, tensor});
                }
            }

            return all_params;
        }

        Tensor<float> backward(const Tensor<float>& grad) {
            if (!is_compiled_) {
                throw std::logic_error("El modelo debe estar compilado antes de backward");
            }
            Tensor<float> current = grad;
            for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                current = (*it)->backward(current);
            }
            last_gradients_ = collect_gradients();
            return current;
        }

        Tensor<float> backward() {
            if (!is_compiled_) {
                throw std::logic_error("El modelo debe estar compilado antes de backward");
            }
            throw std::logic_error("backward() sin gradiente requiere un forward/perdida previo; usa backward(grad)");
        }

        EvalResult evaluate(const Tensor<float>& x, const Tensor<float>& y) {
            Tensor<float> pred = predict(x);
            if (pred.shape() != y.shape()) {
                throw std::invalid_argument("Las etiquetas no coinciden con la salida del modelo");
            }
            return EvalResult{loss_(y, pred)};
        }

        History fit(const Tensor<float>& x, const Tensor<float>& y, FitOptions options) {
            if (!is_compiled_) {
                throw std::logic_error("El modelo debe estar compilado antes de fit");
            }
            if (options.epochs <= 0 || options.batch_size <= 0) {
                throw std::invalid_argument("epochs y batch_size deben ser mayores a 0");
            }

            Tensor<float> probe = predict(x);
            if (probe.shape() != y.shape()) {
                throw std::invalid_argument("Las etiquetas no coinciden con la salida del modelo");
            }

            History history;
            const int n = x.shape()[0];

            for (int epoch = 0; epoch < options.epochs; ++epoch) {
                for (int start = 0; start < n; start += options.batch_size) {
                    int count = std::min(options.batch_size, n - start);
                    Tensor<float> xb = slice_batch(x, start, count);
                    Tensor<float> yb = slice_batch(y, start, count);

                    Tensor<float> pred = predict(xb);
                    Tensor<float> grad = loss_.gradient(yb, pred);
                    backward(grad);

                    for (auto& layer : layers_) {
                        layer->apply_gradients(optimizer_);
                    }
                }
                history.loss.push_back(evaluate(x, y).loss);
            }

            return history;
        }

        [[nodiscard]] std::unordered_map<std::string, Tensor<float>> last_gradients() const {
            return last_gradients_;
        }

    private:
        static Tensor<float> slice_batch(const Tensor<float>& t, int start, int count) {
            const Shape& s = t.shape();
            std::vector<int> dims;
            dims.push_back(count);
            for (int i = 1; i < s.rank(); ++i) {
                dims.push_back(s[i]);
            }
            Tensor<float> out(Shape{dims});

            int per_sample = 1;
            for (int i = 1; i < s.rank(); ++i) {
                per_sample *= s[i];
            }
            for (int i = 0; i < count * per_sample; ++i) {
                out[i] = t[start * per_sample + i];
            }
            return out;
        }

        std::unordered_map<std::string, Tensor<float>> collect_gradients() const {
            std::unordered_map<std::string, Tensor<float>> all_grads;
            int dense_count = 0;
            int conv2d_count = 0;

            for (const auto& layer : layers_) {
                auto layer_grads = layer->gradients();
                if (layer_grads.empty()) continue;

                std::string prefix;
                if (dynamic_cast<layers::Dense*>(layer.get())) {
                    prefix = "dense_" + std::to_string(dense_count++) + "/";
                } else if (dynamic_cast<layers::Conv2D*>(layer.get())) {
                    prefix = "conv2d_" + std::to_string(conv2d_count++) + "/";
                } else {
                    prefix = "unknown_layer/";
                }

                for (const auto& [key, tensor] : layer_grads) {
                    all_grads.insert({prefix + key, tensor});
                }
            }

            return all_grads;
        }
    };

}

#endif
