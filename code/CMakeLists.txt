cmake_minimum_required(VERSION 3.10)
project(CCRRT_Planners)

set(CMAKE_CXX_STANDARD 17)  # Required for erf functions

find_package(ompl REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${OMPL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Main executable
add_executable(CCRRT
    CCRRT.cpp
    MotionValidator.cpp
)

target_link_libraries(CCRRT
    ${OMPL_LIBRARIES}
    Eigen3::Eigen
)