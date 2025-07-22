#!/bin/bash

g++ -I /usr/include/eigen3 -fopenmp -Ofast compute_billiard.cpp -o compute_billiard_cpp.o
for i in $(seq 40 40 400); do
    ./compute_billiard_cpp.o 3 2 0.408 0 1 $i >> tiempos_cpp.txt
done

nvcc -arch=sm_50 -O3 -Xcompiler -Ofast --use_fast_math -o compute_billiard_cuda.o compute_billiard.cu -lcublas -lcusolver
for i in $(seq 40 40 400); do
    ./compute_billiard_cuda.o 3 2 0.408 0 $i >> tiempos_cuda.txt
done

python3 GrafTiempos.py con_opt

