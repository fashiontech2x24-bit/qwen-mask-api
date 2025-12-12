#pragma once

#include <ATen/ATen.h>

// Use TORCH_CHECK for modern PyTorch (replaces deprecated AT_CHECK/AT_ASSERT)
#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERT
#endif

// Updated macros using modern PyTorch APIs
// .is_cuda() replaces deprecated .type().is_cuda()
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!(x).is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)