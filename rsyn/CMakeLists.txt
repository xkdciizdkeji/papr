######################
# Check Dependencies #
######################
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
# find_package(Threads)

###############
# Source Code #
###############
set(PATH_RSYN ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE RSYN_SOURCE_FILES
    ${PATH_RSYN}/src/*.cpp
    ${PATH_RSYN}/src/*.cc
    ${PATH_RSYN}/src/*.c
)

file(GLOB_RECURSE RSYN_HEADER_FILES
    ${PATH_RSYN}/src/*.h
    ${PATH_RSYN}/src/*.hpp
)

#################
# Library Paths #
#################
set(LEF_PATH ${CMAKE_CURRENT_SOURCE_DIR}/def)
set(DEF_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lef)

add_subdirectory(${LEF_PATH})
add_subdirectory(${DEF_PATH})

###########
# Targets #
###########
add_library(rsyn ${RSYN_SOURCE_FILES} ${RSYN_HEADER_FILES})

#######################
# Link Libraries #
#######################

target_link_libraries(rsyn lef)
target_link_libraries(rsyn def)
target_link_libraries(rsyn ${Boost_LIBRARIES})

#######################
# Include Directories #
#######################
target_include_directories(rsyn PUBLIC ${Boost_INCLUDE_DIR})
target_include_directories(rsyn PUBLIC ${PATH_RSYN}/src)
target_include_directories(rsyn PUBLIC ${PATH_RSYN}/src/rsyn/export)
target_include_directories(rsyn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../common)

#################
# Linker Flags  #
#################

# TODO: set static under release mode only
set_target_properties(rsyn PROPERTIES LINK_FLAGS "-static -Wl,--whole-archive -rdynamic -lpthread -Wl,--no-whole-archive")
