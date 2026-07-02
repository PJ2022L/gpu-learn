#include <cuda_runtime.h>
#include <stdint.h>

__global__ void touch_kernel(uint8_t* ptr, size_t bytes, int repeats) {
  size_t stride = blockDim.x * gridDim.x;
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint8_t acc = 0;
  for (int r = 0; r < repeats; ++r) {
    for (size_t i = tid; i < bytes; i += stride) {
      acc ^= ptr[i];
      ptr[i] = static_cast<uint8_t>(acc + ptr[i]);
    }
  }
}

void launch_touch_kernel(void* ptr, size_t bytes, int repeats, cudaStream_t stream) {
  if (bytes == 0) {
    return;
  }
  int block = 256;
  int grid = static_cast<int>((bytes + block - 1) / block);
  if (grid < 1) {
    grid = 1;
  }
  if (grid > 4096) {
    grid = 4096;
  }
  touch_kernel<<<grid, block, 0, stream>>>(static_cast<uint8_t*>(ptr), bytes, repeats);
}

