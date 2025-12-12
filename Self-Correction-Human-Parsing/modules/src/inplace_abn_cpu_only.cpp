#include <torch/extension.h>

#include <vector>

// CPU-only version - no CUDA function declarations
// This file is used when compiling CPU-only extensions

std::vector<at::Tensor> mean_var_cpu(at::Tensor x);
at::Tensor forward_cpu(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
                       bool affine, float eps);
std::vector<at::Tensor> edz_eydz_cpu(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
                                     bool affine, float eps);
at::Tensor backward_cpu(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
                         at::Tensor edz, at::Tensor eydz, bool affine, float eps);
void leaky_relu_backward_cpu(at::Tensor z, at::Tensor dz, float slope);
void elu_backward_cpu(at::Tensor z, at::Tensor dz);

std::vector<at::Tensor> mean_var(at::Tensor x) {
  // CPU-only: always use CPU implementation
  return mean_var_cpu(x);
}

at::Tensor forward(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
                   bool affine, float eps) {
  // CPU-only: always use CPU implementation
  return forward_cpu(x, mean, var, weight, bias, affine, eps);
}

std::vector<at::Tensor> edz_eydz(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
                                 bool affine, float eps) {
  // CPU-only: always use CPU implementation
  return edz_eydz_cpu(z, dz, weight, bias, affine, eps);
}

at::Tensor backward(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
                     at::Tensor edz, at::Tensor eydz, bool affine, float eps) {
  // CPU-only: always use CPU implementation
  return backward_cpu(z, dz, var, weight, bias, edz, eydz, affine, eps);
}

void leaky_relu_forward(at::Tensor z, float slope) {
  at::leaky_relu_(z, slope);
}

void leaky_relu_backward(at::Tensor z, at::Tensor dz, float slope) {
  // CPU-only: always use CPU implementation
  leaky_relu_backward_cpu(z, dz, slope);
}

void elu_forward(at::Tensor z) {
  at::elu_(z);
}

void elu_backward(at::Tensor z, at::Tensor dz) {
  // CPU-only: always use CPU implementation
  elu_backward_cpu(z, dz);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mean_var", &mean_var, "Mean and variance computation");
  m.def("forward", &forward, "In-place forward computation");
  m.def("edz_eydz", &edz_eydz, "First part of backward computation");
  m.def("backward", &backward, "Second part of backward computation");
  m.def("leaky_relu_forward", &leaky_relu_forward, "Leaky relu forward computation");
  m.def("leaky_relu_backward", &leaky_relu_backward, "Leaky relu backward computation and inversion");
  m.def("elu_forward", &elu_forward, "Elu forward computation");
  m.def("elu_backward", &elu_backward, "Elu backward computation and inversion");
}

