#include <cuda_runtime.h>
#include <memory>
#include "helper_cuda.h"

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

// template <class T, class... Args, std::enable_if_t<!std::is_array_v<T>, int> = 0>
// inline auto cuda_make_unique(Args &&...args) // -> std::unique_ptr<T, CudaDeleter<T>>
// {                                            // make a unique_ptr
//   T t(std::forward<Args>(args)...);
//   T *dev_ptr = nullptr;
//   checkCudaErrors(cudaMalloc(&dev_ptr, sizeof(T)));
//   checkCudaErrors(cudaMemcpy(dev_ptr, &t, sizeof(T), cudaMemcpyHostToDevice));
//   return cuda_unique_ptr<T>(dev_ptr);
// }

template <class T, class... Args, std::enable_if_t<!std::is_array_v<T>, int> = 0>
inline void cuda_make_unique(Args &&...args) = delete;

template <class T, std::enable_if_t<std::is_array_v<T> && std::extent_v<T> == 0, int> = 0>
inline auto cuda_make_unique(const size_t size) // -> std::unique_ptr<T, CudaDeleter<std::remove_extent_t<T>>>
{                                               // make a unique_ptr
  using element_type = std::remove_extent_t<T>;
  element_type *dev_ptr = nullptr;
  checkCudaErrors(cudaMalloc(&dev_ptr, size * sizeof(element_type)));
  return cuda_unique_ptr<T>(dev_ptr);
}

template <class T, class... Args, std::enable_if_t<std::extent_v<T> != 0, int> = 0>
inline void cuda_make_unique(Args &&...) = delete;
