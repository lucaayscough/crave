#!/usr/bin/env bash

mkdir -p out

CFLAGS="-g -Wall -O3 -DCRV_INTERNAL -framework Accelerate"
CPPFLAGS="-Wno-deprecated -std=c++20"

clang $CFLAGS src/tests/model.c -o out/test_model && out/test_model out/bin/v1_test_weights.bin && out/test_model out/bin/v2_test_weights.bin
clang $CFLAGS src/tests/gemm.c -o out/test_gemm && out/test_gemm
clang $CFLAGS src/pack.c -o out/pack

clang++ $CPPFLAGS $CFLAGS src/tests/model.c -o out/__cpp_test_model
clang++ $CPPFLAGS $CFLAGS src/pack.c -o out/pack
clang++ $CPPFLAGS $CFLAGS src/pack.c -o out/pack
