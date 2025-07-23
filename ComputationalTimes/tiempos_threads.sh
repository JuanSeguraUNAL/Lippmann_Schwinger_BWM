#!/bin/bash

g++ -I /usr/include/eigen3 -fopenmp -Ofast compute_billiard.cpp -o compute_billiard_cpp.o
for i in $(seq 1 1 16); do
    ./compute_billiard_cpp.o 3 2 0.408 0 $i 200 >> tiempos_OMP_threads.txt
done

python3 GrafThreadsOMP.py