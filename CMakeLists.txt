cmake_minimum_required(VERSION 3.28)
project(Feature_1)

set(CMAKE_CXX_STANDARD 17)

add_executable(Feature_1
        main.cpp
        src/shape.cpp
        src/tensor_backend.cpp
        src/tensor_backend.cpp
        include/utec/algebra/tensor_backend.h
        include/utec/algebra/tensor_ops.h
        include/utec/nn/nn_ops.h
        include/utec/nn/nn_interfaces.h
        include/utec/nn/nn_activation.h
        include/utec/nn/nn_loss.h
        include/utec/nn/nn_dense.h
        include/utec/nn/nn_convolution.h
        include/utec/nn/nn_pooling.h
        include/utec/nn/nn_flatten.h
        include/utec/nn/nn_optimizer.h
        include/utec/nn/neural_network.h
        include/utec/nn/nn_graph.h
        src/feature4_solution.cpp
)

target_include_directories(Feature_1 PRIVATE include)