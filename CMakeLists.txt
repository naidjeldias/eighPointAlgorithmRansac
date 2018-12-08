cmake_minimum_required(VERSION 3.8)
project(RANSAC)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp eightPoint.cpp eightpoint.hpp)
add_executable(RANSAC ${SOURCE_FILES})

include_directories($(OpenCV_INCLUDE_DIRS))
target_link_libraries(RANSAC ${OpenCV_LIBS})