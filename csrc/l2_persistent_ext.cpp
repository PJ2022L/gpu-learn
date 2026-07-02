#include <torch/extension.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

void launch_touch_kernel(void* ptr, size_t bytes, int repeats, cudaStream_t stream);

static void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

std::unordered_map<std::string, int64_t> query_l2_props(int device) {
  cudaDeviceProp prop;
  check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  return {
      {"l2CacheSize", static_cast<int64_t>(prop.l2CacheSize)},
      {"persistingL2CacheMaxSize", static_cast<int64_t>(prop.persistingL2CacheMaxSize)},
      {"accessPolicyMaxWindowSize", static_cast<int64_t>(prop.accessPolicyMaxWindowSize)},
  };
}

int64_t set_persisting_limit(int64_t bytes) {
  check_cuda(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, static_cast<size_t>(bytes)),
             "cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)");
  size_t actual = 0;
  check_cuda(cudaDeviceGetLimit(&actual, cudaLimitPersistingL2CacheSize),
             "cudaDeviceGetLimit(cudaLimitPersistingL2CacheSize)");
  return static_cast<int64_t>(actual);
}

void prime_persisting(torch::Tensor tensor, int64_t bytes, double hit_ratio, int repeats) {
  TORCH_CHECK(tensor.is_cuda(), "keeper tensor must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), "keeper tensor must be contiguous");
  TORCH_CHECK(bytes >= 0, "bytes must be non-negative");
  TORCH_CHECK(hit_ratio > 0.0 && hit_ratio <= 1.0, "hit_ratio must be in (0, 1]");
  size_t requested = static_cast<size_t>(bytes);
  TORCH_CHECK(requested <= static_cast<size_t>(tensor.nbytes()), "bytes exceeds keeper tensor size");

  cudaStream_t stream;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

  cudaStreamAttrValue attr;
  attr.accessPolicyWindow.base_ptr = tensor.data_ptr();
  attr.accessPolicyWindow.num_bytes = requested;
  attr.accessPolicyWindow.hitRatio = hit_ratio;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  check_cuda(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr),
             "cudaStreamSetAttribute(cudaStreamAttributeAccessPolicyWindow)");

  launch_touch_kernel(tensor.data_ptr(), requested, repeats, stream);
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  attr.accessPolicyWindow.num_bytes = 0;
  check_cuda(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr),
             "cudaStreamSetAttribute(disable accessPolicyWindow)");
  check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

void reset_persisting_cache() {
  check_cuda(cudaCtxResetPersistingL2Cache(), "cudaCtxResetPersistingL2Cache");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("query_l2_props", &query_l2_props);
  m.def("set_persisting_limit", &set_persisting_limit);
  m.def("prime_persisting", &prime_persisting);
  m.def("reset_persisting_cache", &reset_persisting_cache);
}

