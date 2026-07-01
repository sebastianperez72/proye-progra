#ifndef PROG3_PF_EPIC1_FEATURE4_V2026_1_NN_GRAPH_H
#define PROG3_PF_EPIC1_FEATURE4_V2026_1_NN_GRAPH_H

#include "nn_interfaces.h"
#include "utec/algebra/tensor_backend.h"
#include <vector>
#include <memory>
#include <stdexcept>

namespace utec::tf {

    class SequentialGraph {
        std::vector<std::unique_ptr<Layer>> layers_;

    public:
        SequentialGraph() = default;

        template <typename LayerT>
        void add(LayerT layer) {
            layers_.push_back(layer.clone());
        }

        [[nodiscard]] bool empty() const {
            return layers_.empty();
        }

        Tensor<float> forward(const Tensor<float>& x) {
            if (layers_.empty()) {
                throw std::logic_error("SequentialGraph::forward sobre un grafo vacio");
            }
            Tensor<float> current = x;
            for (const auto& layer : layers_) {
                current = layer->forward(current);
            }
            return current;
        }

        Tensor<float> backward(const Tensor<float>& grad) {
            if (layers_.empty()) {
                throw std::logic_error("SequentialGraph::backward sobre un grafo vacio");
            }
            Tensor<float> current = grad;
            for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                current = (*it)->backward(current);
            }
            return current;
        }
    };

}

#endif
