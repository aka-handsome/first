#!/bin/bash

# 创建build目录
mkdir -p build
cd build

# 编译
cmake ..
make

# 返回上级目录
cd ..

# 创建输出目录
mkdir -p output

# 运行程序
./bin/forward_intersect "$@" 