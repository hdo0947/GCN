cmake_minimum_required(VERSION 2.8)
project(GCNlib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c99")
# add_executable(GCNlib utilities.c kernels.c main.c)

add_library(GCNlib SHARED
    utilities.cpp kernels.cpp main.cpp
)
set_target_properties(GCNlib PROPERTIES PUBLIC_HEADER utilities.h)