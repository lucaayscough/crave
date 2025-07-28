#!/usr/bin/env bash

mkdir -p out

CFLAGS="-g -Wall -O3 -framework Accelerate"
CPPFLAGS="-Wno-deprecated -std=c++20"

clang $CFLAGS src/tests/model.c -o out/test_model && out/test_model out/bin/v2_test_weights.bin
clang $CFLAGS src/tests/gemm.c -o out/test_gemm
clang $CFLAGS src/tests/tensor_conv_transpose1d.c -o out/test_tensor_conv_transpose1d
clang $CFLAGS src/examples/model.c -o out/example_model

clang++ $CPPFLAGS $CFLAGS src/tests/model.c -o out/__cpp_test_model
clang++ $CPPFLAGS $CFLAGS src/tests/gemm.c -o out/__cpp_test_gemm
clang++ $CPPFLAGS $CFLAGS src/tests/tensor_conv_transpose1d.c -o out/test_tensor_conv_transpose1d
clang++ $CPPFLAGS $CFLAGS src/examples/model.c -o out/__cpp_example_model
