#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_01_TENSOR_BACKEND_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_01_TENSOR_BACKEND_H

#include <Eigen/Dense>
#include <vector>
#include "shape.h"

namespace utec::tf {


template<class T>
    class Tensor {
        Shape shape_;
        Eigen::Array<T, Eigen::Dynamic, 1> data_;

    template <typename... Ix>
        int indexToLocation(Ix... indices) const {

        if (sizeof...(indices) != static_cast<std::size_t>(shape_.rank())) {
            throw std::invalid_argument("numero de indices debe ser igual al rango");
        }

        const int idx_array[] = {static_cast<int>(indices)...};

        int indexOut = 0;
        int dim = 1;

        for (int i = shape_.rank() - 1; i >= 0; --i) {

            if (idx_array[i] < 0 or idx_array[i] >= shape_[i]) {
                throw std::out_of_range("indice fuera de rango");
            }

            indexOut += idx_array[i] * dim;
            dim *= shape_[i];
        }

        return indexOut;
    }

        public:

    Tensor operator-(const Tensor& other) const {
        if (this->shape_ != other.shape_) {
            throw std::invalid_argument("Los shapes deben coincidir para realizar la resta de elementos");
        }

        Tensor result(this->shape_);
        result.data_ = this->data_ - other.data_;
        return result;
    }

    explicit Tensor(const Shape& shape, const T& fill_value)
        : shape_(shape), data_(static_cast<Eigen::Index>(shape.numel())) {
        data_.setConstant(fill_value);
    }

    Tensor operator*(const Tensor& other) const {
        if (this->shape_ != other.shape_) {
            throw std::invalid_argument("Los shapes deben coincidir para realizar la multiplicacion de elementos");
        }
        Tensor result(this->shape_);
        result.data_ = this->data_ * other.data_;
        return result;
    }

    Tensor operator+(const Tensor& other) const {
        if (this->shape_ != other.shape_) {
            throw std::invalid_argument("Los shapes deben coincidir para realizar la suma de elementos");
        }

        Tensor result(this->shape_);

        result.data_ = this->data_ + other.data_;

        return result;
    }

    void reshape(const Shape& new_shape) {
        if (new_shape.numel() != shape_.numel()) {
            throw std::invalid_argument("El nuevo shape debe tener la misma cantidad total de elementos");
        }

        shape_ = new_shape;
    }

    Tensor reshaped(const Shape& new_shape) const {
        Tensor result(*this);
        result.reshape(new_shape);
        return result;
    }

    template <typename... Ix>
    T& operator()(Ix... indices) {
        return data_(static_cast<Eigen::Index>(indexToLocation(indices...)));
    }

    template <typename... Ix>
        const T& operator()(Ix... indices) const {
        return data_(static_cast<Eigen::Index>(indexToLocation(indices...)));
    }

    explicit Tensor(const Shape& shape)
        : shape_(shape), data_(static_cast<Eigen::Index>(shape.numel())) {
        data_.setZero();
    }

    static Tensor zeros(const Shape& shape) {
        Tensor<T> output(shape);
        output.data_.setZero();
        return output;
    }

    static Tensor from_data(const Shape& shape, const std::vector<T>& values) {
        if (values.size() != shape.numel()) {
            throw std::invalid_argument("cantidad de datos incompatible");
        }
        Tensor<T> output(shape);
        for (std::size_t i = 0; i < values.size(); ++i) {
            output.data_(static_cast<Eigen::Index>(i)) = values[i];
        }
        return output;
    }

    Tensor(const Shape& shape, const std::vector<T>& values)
        : shape_(shape), data_(static_cast<Eigen::Index>(shape.numel())) {

        if (values.size() != static_cast<std::size_t>(shape.numel())) {
            throw std::invalid_argument("cantidad de datos incompatible con el shape");
        }
        for (std::size_t i = 0; i < values.size(); ++i) {
            data_(static_cast<Eigen::Index>(i)) = values[i];
        }
    }

    static Tensor ones(const Shape& shape) {
        Tensor<T> out(shape);
        out.data_.setOnes();
        return out;
    }

    int rank() const {
        return shape_.rank();
    }

    int numel() const {
        return shape_.numel();
    }

    int size() const {
        return shape_.total_size();
    }

    const Shape& shape() const {
        return shape_;
    }

    T& operator[](int index) {
        if (index < 0 || index >= shape_.numel()) {
            throw std::out_of_range("Indice lineal fuera de rango");
        }
        return data_(static_cast<Eigen::Index>(index));
    }

    const T& operator[](int index) const {
        if (index < 0 || index >= shape_.numel()) {
            throw std::out_of_range("Indice lineal fuera de rango");
        }
        return data_(static_cast<Eigen::Index>(index));
    }

    T& at(const std::vector<int>& indices) {
        if (indices.size() != static_cast<std::size_t>(shape_.rank())) {
            throw std::invalid_argument("Numero de indices debe ser igual al rango");
        }

        int indexOut = 0;
        int dim = 1;

        for (int i = shape_.rank() - 1; i >= 0; --i) {
            if (indices[i] < 0 || indices[i] >= shape_[i]) {
                throw std::out_of_range("Indice fuera de rango");
            }
            indexOut += indices[i] * dim;
            dim *= shape_[i];
        }
        return data_(static_cast<Eigen::Index>(indexOut));
    }

    const T& at(const std::vector<int>& indices) const {
        if (indices.size() != static_cast<std::size_t>(shape_.rank())) {
            throw std::invalid_argument("Numero de indices debe ser igual al rango");
        }

        int indexOut = 0;
        int dim = 1;

        for (int i = shape_.rank() - 1; i >= 0; --i) {
            if (indices[i] < 0 || indices[i] >= shape_[i]) {
                throw std::out_of_range("Indice fuera de rango");
            }
            indexOut += indices[i] * dim;
            dim *= shape_[i];
        }
        return data_(static_cast<Eigen::Index>(indexOut));
    }




    };

    template <typename T>
    bool allclose(const Tensor<T>& a, const Tensor<T>& b, T atol = static_cast<T>(1e-5)) {
        if (a.shape() != b.shape()) {
            return false;
        }
        const int n = a.shape().numel();
        for (int i = 0; i < n; ++i) {
            T diff = a[i] - b[i];
            if (diff < T{0}) {
                diff = -diff;
            }
            if (diff > atol) {
                return false;
            }
        }
        return true;
    }
}

using utec::tf::Tensor;

#endif
