## 1. How to build
**build on windows10**
```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:/DevTools/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --config Release
```

**build on linux/wsl**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## 2. Dependencies

* A C/C++ compiler with support for C++17 or higher.
* [CMake](https://cmake.org/) (version >= 3.18)
* [Boost](https://www.boost.org/) (version >= 1.58)
* [Rsyn](https://github.com/RsynTeam/rsyn-x) (a trimmed version is used, already added under folder `rsyn`)