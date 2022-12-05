cmake_minimum_required(VERSION 3.16)

set(MSVC_INCREMENTAL_DEFAULT ON)

project(
  Controller
  VERSION 0.1.0
  DESCRIPTION "General propose logger to be used in the robot"
)

add_library(ControllerInterface INTERFACE)

target_include_directories(ControllerInterface
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/includes>
)

# add_subdirectory(source/InfluxDB_logger)

add_library(Controller STATIC source/controller.cc)

target_link_libraries(Controller PUBLIC 
    ControllerInterface
    # InfluxDbLogger
)

set_property(TARGET Controller PROPERTY POSITION_INDEPENDENT_CODE ON)

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

target_link_libraries(Controller PUBLIC
    Eigen3::Eigen
)
