# Task #PF-EPIC1-Feature4: Proyecto Final - Epic 1 - Feature 4 - Grafo computacional, gradientes y parámetros

**course:** Programación III  
**unit:** final project  
**cmake project:** `prog3_pf_epic1_feature4_v2026_1`

## Indicaciones Específicas

El tiempo límite para la evaluación es 2 semanas.

La implementación debe seguir la estructura base descrita en `pf_epic_1_feature_4_2026_1.typ`. Los archivos principales se ubican en `include/utec/` y, cuando corresponda, en `src/`:

- `include/utec/algebra/shape.h`
- `include/utec/algebra/tensor_backend.h`
- `include/utec/algebra/tensor_ops.h`
- `include/utec/nn/nn_ops.h`
- `include/utec/nn/nn_interfaces.h`
- `include/utec/nn/nn_dense.h`
- `include/utec/nn/nn_convolution.h`
- `include/utec/nn/nn_pooling.h`
- `include/utec/nn/nn_flatten.h`
- `include/utec/nn/nn_activation.h`
- `include/utec/nn/nn_loss.h`
- `include/utec/nn/nn_optimizer.h`
- `include/utec/nn/nn_graph.h`
- `include/utec/nn/neural_network.h`
- `src/feature4_solution.cpp`

Deberás subir estos archivos directamente a www.gradescope.com o crear un `.zip` que los contenga.

Cada pregunta incluye un `Use Case` tomado del archivo `autograder/tests/question_N/test_1/test_1.cpp`. Se muestra solo el cuerpo del caso de prueba.

## Question #1 - Shape, Tensor y nn_ops - base numérica para backward (1 point)

Implementa la base numérica mínima para entrenamiento: formas, tensores, acceso por índices, `reshape`, comparación aproximada con `allclose` y operaciones como `matmul` y `conv2d`.

**Use Case:**
```cpp
using namespace feature4_tests;

Shape shape{2, 3, 4};
REQUIRE(shape.rank() == 3);
REQUIRE(shape.total_size() == 24);
REQUIRE(shape[1] == 3);

auto tensor = Tensor<float>::ones(shape);
REQUIRE(tensor.shape() == shape);
REQUIRE(tensor.size() == 24);
tensor(1, 2, 3) = 7.0f;
REQUIRE(tensor(1, 2, 3) == Approx(7.0f));

auto reshaped = tensor.reshaped(Shape{4, 6});
require_shape(reshaped.shape(), {4, 6});
REQUIRE(tensor.shape() == shape);
REQUIRE_FALSE(utec::tf::allclose(tensor, Tensor<float>::zeros(shape)));
```

## Question #2 - Contratos de capas, cachés de inferencia y utilidades de gradiente (2 points)

Define una interfaz común de capas con `build`, `forward`, `backward`, `parameters`, `gradients` y clonación. Cada capa debe guardar la información necesaria para retropropagar después del `forward`.

**Use Case:**
```cpp
using namespace feature4_tests;

utec::tf::layers::Dense dense(2, Activation::Linear);
dense.build(Shape{3});
auto params = dense.parameters();
REQUIRE(params.contains("weights"));
REQUIRE(params.contains("bias"));
require_shape(params.at("weights").shape(), {3, 2});
require_shape(params.at("bias").shape(), {2});

auto x = Tensor<float>::ones(Shape{2, 3});
auto y = dense.forward(x);
auto dx = dense.backward(Tensor<float>::ones(y.shape()));
require_shape(dx.shape(), {2, 3});

auto grads = dense.gradients();
REQUIRE(grads.contains("weights"));
REQUIRE(grads.contains("bias"));
require_shape(grads.at("weights").shape(), {3, 2});
require_shape(grads.at("bias").shape(), {2});
```

## Question #3 - Activaciones y pérdida - derivadas para Relu, Softmax y crossentropy (2 points)

Implementa `Relu`, `Softmax`, sus derivadas cuando corresponda, y `CategoricalCrossentropy`. Para `Softmax + CategoricalCrossentropy`, usa la simplificación `dL/dz = y_pred - y_true`.

**Use Case:**
```cpp
using namespace feature4_tests;

auto relu_input = Tensor<float>::from_data(Shape{1, 4},
                                           {-2, 0, 3, 5});
require_tensor_close(utec::tf::apply_activation(relu_input, Activation::Relu),
                     Tensor<float>::from_data(Shape{1, 4},
                                              {0, 0, 3, 5}));

auto logits = Tensor<float>::from_data(Shape{2, 2},
                                       {0, 0,
                                        2, 0});
auto probs = utec::tf::apply_activation(logits, Activation::Softmax);
REQUIRE(probs(0, 0) == Approx(0.5f));
REQUIRE(probs(0, 1) == Approx(0.5f));
REQUIRE(probs(1, 0) + probs(1, 1) == Approx(1.0f));
REQUIRE(probs(1, 0) > probs(1, 1));
```

## Question #4 - Optimizador SGD y estado mínimo de entrenamiento (1 point)

Implementa `optimizers::SGD` con tasa de aprendizaje positiva y actualización en memoria: `param = param - learning_rate * grad`.

**Use Case:**
```cpp
using namespace feature4_tests;

auto parameter = Tensor<float>::from_data(Shape{2, 2},
                                          {1, 2,
                                           3, 4});
auto gradient = Tensor<float>::from_data(Shape{2, 2},
                                         {0.5f, -0.5f,
                                          1.0f, -1.0f});
utec::tf::optimizers::SGD optimizer(0.1f);
optimizer.update(parameter, gradient);

require_tensor_close(parameter,
                     Tensor<float>::from_data(Shape{2, 2},
                                              {0.95f, 2.05f,
                                               2.9f, 4.1f}));
```

## Question #5 - Dense backward - dW, db y dX (2 points)

Implementa la retropropagación de `Dense`: cálculo de `dW`, `db`, `dX`, manejo de activación y registro de gradientes para `weights` y `bias`.

**Use Case:**
```cpp
using namespace feature4_tests;

utec::tf::layers::Dense dense(2, Activation::Linear);
dense.build(Shape{3});
dense.set_weights(Tensor<float>::from_data(Shape{3, 2},
                                           {1, 2,
                                            3, 4,
                                            5, 6}));
dense.set_bias(Tensor<float>::from_data(Shape{2}, {0.5f, -0.5f}));

auto x = Tensor<float>::from_data(Shape{2, 3},
                                  {1, 2, 3,
                                   4, 5, 6});
auto y = dense.forward(x);
require_tensor_close(y,
                     Tensor<float>::from_data(Shape{2, 2},
                                              {22.5f, 27.5f,
                                               49.5f, 63.5f}));

auto grad_out = Tensor<float>::from_data(Shape{2, 2},
                                         {0.1f, -0.2f,
                                          0.3f, 0.4f});
auto dx = dense.backward(grad_out);
require_tensor_close(dx,
                     Tensor<float>::from_data(Shape{2, 3},
                                              {-0.3f, -0.5f, -0.7f,
                                               1.1f, 2.5f, 3.9f}));

auto grads = dense.gradients();
require_tensor_close(grads.at("weights"),
                     Tensor<float>::from_data(Shape{3, 2},
                                              {1.3f, 1.4f,
                                               1.7f, 1.6f,
                                               2.1f, 1.8f}));
require_tensor_close(grads.at("bias"),
                     Tensor<float>::from_data(Shape{2}, {0.4f, 0.2f}));
```

## Question #6 - Conv2D backward - gradientes de parámetros y entrada (2 points)

Implementa `backward` de `Conv2D` usando activaciones `NHWC` y kernels `HWIO`. Debe calcular gradientes respecto a pesos, bias y entrada.

**Use Case:**
```cpp
using namespace feature4_tests;

utec::tf::layers::Conv2D conv(1, {2, 2}, Activation::Linear);
conv.build(Shape{3, 3, 1});
conv.set_weights(Tensor<float>::from_data(Shape{2, 2, 1, 1},
                                          {1, 0,
                                           0, -1}));
conv.set_bias(Tensor<float>::from_data(Shape{1}, {0}));

auto x = Tensor<float>::from_data(Shape{1, 3, 3, 1},
                                  {1, 2, 3,
                                   4, 5, 6,
                                   7, 8, 9});
auto y = conv.forward(x);
require_tensor_close(y,
                     Tensor<float>::from_data(Shape{1, 2, 2, 1},
                                              {-4, -4,
                                               -4, -4}));

auto dx = conv.backward(Tensor<float>::ones(Shape{1, 2, 2, 1}));
require_tensor_close(dx,
                     Tensor<float>::from_data(Shape{1, 3, 3, 1},
                                              {1, 1, 0,
                                               1, 0, -1,
                                               0, -1, -1}));

auto grads = conv.gradients();
require_tensor_close(grads.at("weights"),
                     Tensor<float>::from_data(Shape{2, 2, 1, 1},
                                              {12, 16,
                                               24, 28}));
require_tensor_close(grads.at("bias"),
                     Tensor<float>::from_data(Shape{1}, {4}));
```

## Question #7 - MaxPooling2D backward - máscara o índices ganadores (2 points)

Implementa `backward` de `MaxPooling2D`. El gradiente debe volver solo a la posición ganadora de cada ventana registrada durante el `forward`.

**Use Case:**
```cpp
using namespace feature4_tests;

utec::tf::layers::MaxPooling2D pool({2, 2});
pool.build(Shape{4, 4, 1});
auto x = Tensor<float>::from_data(Shape{1, 4, 4, 1},
                                  {1, 2, 3, 4,
                                   5, 6, 7, 8,
                                   9, 10, 11, 12,
                                   13, 14, 15, 16});
auto y = pool.forward(x);
require_tensor_close(y,
                     Tensor<float>::from_data(Shape{1, 2, 2, 1},
                                              {6, 8,
                                               14, 16}));

auto dx = pool.backward(Tensor<float>::from_data(Shape{1, 2, 2, 1},
                                                 {1, 2,
                                                  3, 4}));
require_tensor_close(dx,
                     Tensor<float>::from_data(Shape{1, 4, 4, 1},
                                              {0, 0, 0, 0,
                                               0, 1, 0, 2,
                                               0, 0, 0, 0,
                                               0, 3, 0, 4}));
```

## Question #8 - Flatten backward y reshape inverso (1 point)

Implementa `Flatten::backward` como el `reshape` inverso hacia la forma original del input guardada en el `forward`.

**Use Case:**
```cpp
using namespace feature4_tests;

utec::tf::layers::Flatten flatten;
flatten.build(Shape{2, 2, 1});
auto x = Tensor<float>::from_data(Shape{2, 2, 2, 1},
                                  {1, 2, 3, 4,
                                   5, 6, 7, 8});
auto y = flatten.forward(x);
require_tensor_close(y,
                     Tensor<float>::from_data(Shape{2, 4},
                                              {1, 2, 3, 4,
                                               5, 6, 7, 8}));

auto dx = flatten.backward(Tensor<float>::ones(Shape{2, 4}));
require_shape(dx.shape(), {2, 2, 2, 1});
REQUIRE(dx.size() == x.size());
```

## Question #9 - Grafo secuencial y flujo backward capa por capa (2 points)

Representa el grafo computacional secuencial y ejecuta la retropropagación recorriendo las capas en orden inverso.

**Use Case:**
```cpp
using namespace feature4_tests;

utec::tf::layers::Flatten flatten;
flatten.build(Shape{2, 2});
utec::tf::layers::Dense dense(1, Activation::Linear);
dense.build(Shape{4});
dense.set_weights(Tensor<float>::from_data(Shape{4, 1},
                                           {1, 2, 3, 4}));
dense.set_bias(Tensor<float>::from_data(Shape{1}, {0}));

utec::tf::SequentialGraph graph;
graph.add(flatten);
graph.add(dense);

auto output = graph.forward(Tensor<float>::from_data(Shape{1, 2, 2},
                                                     {1, 2,
                                                      3, 4}));
require_tensor_close(output, Tensor<float>::from_data(Shape{1, 1}, {30}));

auto dx = graph.backward(Tensor<float>::ones(Shape{1, 1}));
require_shape(dx.shape(), {1, 2, 2});
require_tensor_close(dx,
                     Tensor<float>::from_data(Shape{1, 2, 2},
                                              {1, 2,
                                               3, 4}));
```

## Question #10 - Sequential - fit, evaluate y backward en modelo compilado (2 points)

Implementa el flujo de entrenamiento completo de `Sequential`: `compile`, `predict`, `backward`, `fit`, `evaluate`, validación de shapes y registro de historial de pérdida.

**Use Case:**
```cpp
using namespace feature4_tests;

auto [x, y] = dense_dataset();
auto model = dense_model(0.4f);
const auto before = model.evaluate(x, y).loss;
auto history = model.fit(x, y, FitOptions{.epochs = 40, .batch_size = 2});
const auto after = model.evaluate(x, y).loss;

REQUIRE(history.loss.size() == 40);
REQUIRE(after < before);
REQUIRE(after >= 0.0f);
```

## Question #11 - Parámetros, last_gradients y actualización efectiva con SGD (2 points)

Expón parámetros y gradientes con nombres estables como `conv2d_0/weights`, `conv2d_0/bias`, `dense_0/weights` y `dense_0/bias`. Verifica que `SGD` modifique realmente los parámetros.

**Use Case:**
```cpp
using namespace feature4_tests;

auto [x, y] = conv_dataset();
auto model = conv_model(0.2f);
auto before = model.parameters();
(void)model.fit(x, y, FitOptions{.epochs = 10, .batch_size = 2});
auto after = model.parameters();

REQUIRE(before.contains("conv2d_0/weights"));
REQUIRE(before.contains("conv2d_0/bias"));
REQUIRE(before.contains("dense_0/weights"));
REQUIRE(before.contains("dense_0/bias"));
REQUIRE_FALSE(utec::tf::allclose(before.at("conv2d_0/weights"),
                                 after.at("conv2d_0/weights"), 1.0e-6f));
REQUIRE_FALSE(utec::tf::allclose(before.at("conv2d_0/bias"),
                                 after.at("conv2d_0/bias"), 1.0e-6f));
REQUIRE_FALSE(utec::tf::allclose(before.at("dense_0/weights"),
                                 after.at("dense_0/weights"), 1.0e-6f));
REQUIRE_FALSE(utec::tf::allclose(before.at("dense_0/bias"),
                                 after.at("dense_0/bias"), 1.0e-6f));
```

## Question #12 - Validaciones de etiquetas, estado no compilado y allclose (1 point)

Agrega validaciones de fallos esperados: etiquetas incompatibles, `backward` en modelo no compilado, opciones inválidas de entrenamiento y comparación aproximada de tensores con `allclose`.

**Use Case:**
```cpp
using namespace feature4_tests;

auto [x, y] = dense_dataset();
auto model = dense_model(0.1f);
auto wrong_y = Tensor<float>::ones(Shape{4, 3});
REQUIRE_THROWS_AS(model.fit(x, wrong_y, FitOptions{.epochs = 1,
                                                   .batch_size = 2}),
                  std::invalid_argument);

Sequential bad_model;
bad_model.add(utec::tf::layers::Input(Shape{2}));
bad_model.add(utec::tf::layers::Dense(2, Activation::Softmax));
REQUIRE_THROWS_AS(bad_model.backward(), std::logic_error);
```
