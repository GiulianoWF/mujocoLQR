cmake_minimum_required(VERSION 3.16)

set(MSVC_INCREMENTAL_DEFAULT ON)

project(
  Differenciator
  VERSION 0.1.0
  DESCRIPTION "Mujoco differenciator as used by MahanFathi/iLQG-MuJoCo"
)

add_library(DifferenciatorInterface INTERFACE)

target_include_directories(DifferenciatorInterface
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/includes>
)

add_library(Differenciator STATIC mjderivative.cpp)

target_link_libraries(Differenciator PUBLIC 
    DifferenciatorInterface
)

set_property(TARGET Differenciator PROPERTY POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
FetchContent_MakeAvailable(Eigen)

target_link_libraries(Differenciator PUBLIC
    Eigen3::Eigen
)
