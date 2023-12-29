#########
# Setup #
#########

# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.18)

# Message
message(STATUS "Global Router CUGR 2.0")
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})

# Find includes in the build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(ENABLE_CUDA "enable cuda for gpu maze routing" OFF)

# Project's name
project(CUGR LANGUAGES C CXX)

# Set the output folder where your program will be created
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

set(PATH_FLUTE ${CMAKE_CURRENT_SOURCE_DIR}/flute)
set(PATH_CUGR ${CMAKE_CURRENT_SOURCE_DIR}/src)

###############
# Source Code #
###############
file(GLOB_RECURSE CUGR_HEADER_FILE ${PATH_CUGR}/*.h ${PATH_CUGR}/*.hpp)
if(ENABLE_CUDA)
  file(GLOB_RECURSE CUGR_SOURCE_FILES ${PATH_CUGR}/*.c ${PATH_CUGR}/*.cpp ${PATH_CUGR}/*.cu)
else()
  file(GLOB_RECURSE CUGR_SOURCE_FILES ${PATH_CUGR}/*.c ${PATH_CUGR}/*.cpp)
endif()

###########
# Targets #
###########
add_executable(route ${CUGR_SOURCE_FILES} ${CUGR_HEADER_FILE})

##################
# Subdirectories #
##################
add_subdirectory(${PATH_FLUTE})

#######################
# Include Directories #
#######################

##################
# Compile Option #
##################
if(ENABLE_CUDA)
  target_compile_options(route PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_compile_options(route PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  target_compile_definitions(route PUBLIC -DENABLE_CUDA)
endif()

#################
# Linker Flags  #
#################
# TODO: set static under release mode only
if(MSVC)
  if(ENABLE_CUDA)
    target_link_options(route PUBLIC /NODEFAULTLIB:libcmt.lib)
  endif()
else()
  set_target_properties(route PROPERTIES LINK_FLAGS "-static -Wl,--whole-archive -rdynamic -lpthread -Wl,--no-whole-archive")
endif()

##################
# Link Libraries #
##################
target_link_libraries(route PRIVATE flute)

add_custom_command(
  TARGET route POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy 
    ${PATH_FLUTE}/POST9.dat 
    $<TARGET_FILE_DIR:route>/POST9.dat
  COMMAND ${CMAKE_COMMAND} -E copy 
    ${PATH_FLUTE}/POWV9.dat
    $<TARGET_FILE_DIR:route>/POWV9.dat
)