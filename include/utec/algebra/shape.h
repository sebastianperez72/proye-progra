#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_01_SHAPE_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_01_SHAPE_H

#include <vector>
#include <stdexcept>
#include <initializer_list>
#include <cstddef>
#include <ostream>


namespace utec::tf {
    class Shape {
        std::vector<int> dims;

        void validate(const std::vector<int> &dims_) {
            for (auto dim : dims_) {
                if (dim <= 0) {
                    throw std::invalid_argument("Cada dimension debe ser mayor a 0");
                }
            }

        }
    public:

        explicit Shape(std::initializer_list<int> dims_) : dims(dims_) {
            validate(dims_);
        }
        Shape(std::initializer_list<std::size_t> dims_) {
            std::vector<int> tmp;
            tmp.reserve(dims_.size());
            for (auto d : dims_) {
                tmp.push_back(static_cast<int>(d));
            }
            validate(tmp);
            dims = std::move(tmp);
        }
        Shape(std::vector<int> dims_) {
            validate(dims_);
            dims = std::move(dims_);
        }
        int rank() const {
            return dims.size();
        }
        int size() const {
            return dims.size();
        }
        int numel() const {

            if (dims.empty()) return 0;

            int numel_ = 1;
            for (auto dim : dims) {
                numel_ *= dim;
            }
            return numel_;
        }
        int total_size() const {
            return numel();
        }
        int operator[](int index) const {
            return dims.at(index);
        }

        bool operator==(const Shape &other) const {
            return dims == other.dims;
        }


    };

    inline std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << "(";
        for (int i = 0; i < shape.rank(); ++i) {
            if (i > 0) os << ", ";
            os << shape[i];
        }
        os << ")";
        return os;
    }
}

using utec::tf::Shape;

#include "tensor_backend.h"

#endif
