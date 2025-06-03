#!/usr/bin/env bash

mkdir -p out

COMPILER_FLAGS="-g -Wall -O0 -DCRV_INTERNAL -DCRV_IM2COL"

clang $COMPILER_FLAGS src/tests/v1.c -o out/tests_v1 && out/tests_v1 ~/Desktop/v1_test_weights.bin

clang++ -Wno-deprecated -std=c++20 $COMPILER_FLAGS src/tests/v1.c -o out/__cpp_tests_v1

#COMPILER_FLAGS="-Wall \
#                -Wno-deprecated \
#                -DACCELERATE_NEW_LAPACK \
#                -DACCELERATE_LAPACK_ILP64 \
#                -framework Accelerate \
#                -fopenmp"
#
#if [ -n "$BUILD_ALL" ] && [ "$BUILD_ALL" -eq 1 ]; then
#  clang -g -O3 src/pack.c -o out/pack
#  clang -g -O3 src/examples/v2.c -o out/v2
#  clang -g -O3 src/tests/gemm.c -o out/test_gemm
#fi
#
#clang++ -std=c++20 src/main.c $COMPILER_FLAGS -o out/__cpp_test
#
#if [ -n "$FAST" ] && [ "$FAST" -eq 1 ]; then
#  clang -g -O3 $COMPILER_FLAGS src/main.c -o out/main
#else
#  clang -g -O0 -DINTERNAL $COMPILER_FLAGS src/main.c -o out/main
#fi
