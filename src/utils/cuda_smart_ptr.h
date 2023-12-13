#include <cuda_runtime.h>
#include "helper_cuda.h"

template <class T>
struct CudaDeleter
{
  void operator()(T *ptr)
  {
    checkCudaErrors(cudaFree(ptr));
  }
};

template <class T, class... Args, std::enable_if_t<!std::is_array_v<T>, int> = 0>
auto cuda_make_unique(Args &&...args) // -> std::unique_ptr<T, CudaDeleter<T>>
{                                     // make a unique_ptr
  T t(std::forward<Args>(args)...);
  T *dev_ptr = nullptr;
  checkCudaErrors(cudaMalloc(&dev_ptr, sizeof(T)));
  checkCudaErrors(cudaMemcpy(dev_ptr, &t, sizeof(T), cudaMemcpyHostToDevice));
  return std::unique_ptr<T, CudaDeleter<T>>(dev_ptr);
}

template <class T, std::enable_if_t<std::is_array_v<T> && std::extent_v<T> == 0, int> = 0>
auto cuda_make_unique(const size_t size) // -> std::unique_ptr<T, CudaDeleter<std::remove_extent_t<T>>>
{                                        // make a unique_ptr
  using element_type = std::remove_extent_t<T>;
  element_type *dev_ptr = nullptr;
  checkCudaErrors(cudaMalloc(&dev_ptr, size * sizeof(element_type)));
  return std::unique_ptr<T, CudaDeleter<element_type>>(dev_ptr);
}

template <class T, class... Args, std::enable_if_t<std::extent_v<T> != 0, int> = 0>
void cuda_make_unique(Args &&...) = delete;

template <class T>
using CorrectCudaDeleter = std::conditional_t<std::is_array_v<T>, CudaDeleter<std::remove_all_extents_t<T>>, CudaDeleter<T>>;

template <class T>
using cuda_unique_ptr = std::unique_ptr<T, CorrectCudaDeleter<T>>;
