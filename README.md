## 1. How to build

**Step 1:** Download the source code. For example,
```bash
$ git clone -b ispd24-analytical_algorithm https://github.com/flwave/gr_gpu.git
```
**Step 1.5：** 可以修改一下run_papr.sh文件里的前三个路径设置，然后直接./run_papr.sh，即可自动测试四种环境和五个小case的表现（也可以自由修改），log文件保存在你设置的Simpleinputs/log/下。如果想要生成数据表格，可以看“生成xlsx脚本”文件夹。


**Step 2:** build by cmake
```bash
$ cd ./gr_gpu/
$ cmake -B build -DCMAKE_BUILD_TYPE=Release
$ cmake --build build
```
**Step 2":** build by cmake (enable cuda)
```bash
$ cd ./gr_gpu/
$ cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
$ cmake --build build
```
**Step 2":** build by cmake (enable iss sort)
```bash
$ cd ./gr_gpu/
$ cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_ISSSORT=ON
$ cmake --build build
```
**Step 2":** build by cmake (enable iss sort and cuda)
```bash
$ cd ./gr_gpu/
$ cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_ISSSORT=ON -DENABLE_CUDA=ON
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
