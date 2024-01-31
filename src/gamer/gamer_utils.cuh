#ifdef ENABLE_CUDA
#ifndef GPU_ROUTE_GAMER_UTILS_H
#define GPU_ROUTE_GAMER_UTILS_H
#include <vector>
#include <cuda_runtime.h>
#include <memory>
#include "helper_math.h"
#include "helper_cuda.h"
#include "../gr/GRTree.h"
#include "../utils/utils.h"

// --------------------------------
// Common Constant
// -------------------------------

using realT = double;
constexpr int PACK_ROW_SIZE = 1024;
constexpr int MAX_NUM_LAYER = 10;
constexpr int VIA_SEG_SIZE = MAX_NUM_LAYER;
constexpr realT INFINITY_DISTANCE = std::numeric_limits<realT>::infinity();
constexpr int MAX_NUM_TURNS = 15;
constexpr int MAX_ROUTE_LEN_PER_PIN = (MAX_NUM_TURNS + 1) * 2;
constexpr int MAX_SCALE = 5;

// -------------------------------
// Cuda smart ptr
// -------------------------------

template <class T>
struct CudaDeleter
{
  void operator()(T *ptr)
  {
    checkCudaErrors(cudaFree(ptr));
  }
};

template <class T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter<typename std::unique_ptr<T>::element_type>>;

template <class T>
using cuda_shared_ptr = std::shared_ptr<T>;

template <class T, std::enable_if_t<std::is_array_v<T> && std::extent_v<T> == 0, int> = 0>
inline auto cuda_make_unique(const size_t size)
{
  using element_type = std::remove_extent_t<T>;
  element_type *dev_ptr = nullptr;
  checkCudaErrors(cudaMalloc(&dev_ptr, size * sizeof(element_type)));
  return cuda_unique_ptr<T>(dev_ptr, CudaDeleter<element_type>{});
}

template <class T, std::enable_if_t<std::is_array_v<T> && std::extent_v<T> == 0, int> = 0>
inline auto cuda_make_shared(const size_t size)
{
  using element_type = std::remove_extent_t<T>;
  element_type *dev_ptr = nullptr;
  checkCudaErrors(cudaMalloc(&dev_ptr, size * sizeof(element_type)));
  return cuda_shared_ptr<T>(dev_ptr, CudaDeleter<element_type>{});
}

template <class T, class... Args, std::enable_if_t<!std::is_array_v<T> || std::extent_v<T> != 0, int> = 0>
inline void cuda_make_unique(Args &&...) = delete;

template <class T, class... Args, std::enable_if_t<!std::is_array_v<T> || std::extent_v<T> != 0, int> = 0>
inline void cuda_make_shared(Args &&...) = delete;

// -------------------------------
// Common device functions
// -------------------------------

inline __host__ __device__ int3 idxToXYZ(int idx, int DIRECTION, int N)
{
  int layer = idx / N / N;
  return (layer & 1) ^ DIRECTION ? make_int3(idx % N, (idx / N) % N, layer)
                                 : make_int3((idx / N) % N, idx % N, layer);
}

inline __host__ __device__ int xyzToIdx(int x, int y, int z, int DIRECTION, int N)
{
  return (z & 1) ^ DIRECTION ? z * N * N + y * N + x
                             : z * N * N + x * N + y;
}

inline __host__ __device__ realT logistic(realT x, realT slope)
{
  return 1.f / (1.f + exp(x * slope));
}

template<class T>
inline __device__ double myAtomicAdd(T *address, T val)
{
  atomicAdd(address, val);
}

template<>
inline __device__ double myAtomicAdd(double *address, double val)
{
  unsigned long long old = *(unsigned long long *)address;
  unsigned long long assumed;
  do
  {
    assumed = old;
    old = atomicCAS(
      (unsigned long long *)address,
      assumed,
      __double_as_longlong(val + __longlong_as_double(assumed))
    );
  } while (assumed != old);
  return __longlong_as_double(old);
}

#endif
#endif