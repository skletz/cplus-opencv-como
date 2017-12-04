cmake_minimum_required(VERSION 3.9)
project(opencv-como)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB src_dir "como/src/*.cpp")

# set(SOURCE_FILES como/src/COMO.cpp)
add_executable(opencv-como ${src_dir})

target_link_libraries( opencv-como ${OpenCV_LIBS} )
