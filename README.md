## 1. How to build

**Step 1:** Download the source code. For example,
```bash
$ git clone -b ispd24-lite https://github.com/flwave/gr_gpu.git
```

**Step 2:** build by cmake
```bash
$ cd ./gr_gpu/
$ cmake -B build -DCMAKE_BUILD_TYPE=Release [options]
$ cmake --build build
```
You can use the following options:
```
-DCMAKE_BUILD_TYPE=Release/Debug
-DENABLE_CUDA=ON/OFF
-DENABLE_ISSSORT=ON/OFF
-DONLY_PATTERN_ROUTING=ON/OFF
-DCONGESTION_UPDATE=ON/OFF
```
For example, if you want to build with cuda, you can use the following command:
```bash
$ cd ./gr_gpu/
$ cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
$ cmake --build build
```
**Step 3:** use the executable file
```bash
$ cd ./build/
$ ./route -cap [path_to_cap_file] -net [path_to_net_file] -output [path_to_output_file] -threads [num_of_threads(default=1)] 
```
## 2. Dependencies

* A C/C++ compiler with support for C++17.
* [CMake](https://cmake.org/) (version >= 3.18)
* [Boost](https://www.boost.org/) (version >= 1.58)
* [CUDA](https://developer.nvidia.com/cuda-toolkit) (optional)
