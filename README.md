## 1. How to build

**Step 1:** Download the source code. For example,
```bash
$ git clone -b gr_multithread https://github.com/flwave/gr_gpu.git
```

**Step 2:** build by cmake
```bash
$ cd ./gr_gpu/
$ cmake -B build -DCMAKE_BUILD_TYPE=Release
$ cmake --build build
```
**Step 2":** build by cmake (enable cuda)
```bash
$ cd ./gr_gpu/
$ cmake -B build -DENABLE_CUDA=ON
$ cmake --build build
```
**Step 3:** use the executable file
```bash
$ cd ./build/
$ ./route [path_to_cap_file] [path_to_net_file] [path_to_output_file] [num_threads]
```
## 2. Dependencies

* A C/C++ compiler with support for C++17.
* [CMake](https://cmake.org/) (version >= 3.18)
* [CUDA](https://developer.nvidia.com/cuda-toolkit) (optional)
