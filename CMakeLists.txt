cmake_minimum_required(VERSION 3.10)
project(forward_intersect)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV包
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找Eigen包
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# 添加可执行文件
add_executable(forward_intersect forward_intersect.cpp)

# 链接库
target_link_libraries(forward_intersect ${OpenCV_LIBS})

# 设置输出目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 创建输出目录
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/output)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin) 