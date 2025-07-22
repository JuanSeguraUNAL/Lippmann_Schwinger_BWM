#ifndef BWM_CONFOCAL_PARABOLIC_BILLIARD_HPP
#define BWM_CONFOCAL_PARABOLIC_BILLIARD_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct Point{
        double x, y;
};

namespace bwm {

// CUDA kernel to generate a vector of num elements spaced uniformly given the pointer to a vector allocated in memory
__global__ void linspace(double *vector, double start, int num, double step){
    if (num == 1){
        vector[0] = start;
    }
    else{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < num && tid >= 0){
            vector[tid] = start + tid * step;
        }
    }
}

// CUDA kernel to generate a vector of n elements of value v
__global__ void fill_vec(double *vector, double v, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        vector[tid] = v;
    }
}

// CUDA kernel to join or concatenate two vectors
__global__ void concatenate(const double *A, int size_A, const double *B, int size_B, double *C, int size_C){
    int tid = blockIdx.x * blockDim.x +threadIdx.x;
    
    if(tid < size_C){
        if (tid < size_A){C[tid] = A[tid];}
        else {C[tid] = B[tid - size_A];}
    }
}

// CUDA kernel to transform from parabolic coordinates to cartesian coordinates
__global__ void ParToCart(const double *xi, const double *eta, double *x, double *y, int size, double x_disp){
    int tid = blockIdx.x * blockDim.x +threadIdx.x;
    if (tid < size){
        x[tid] = (xi[tid] * xi[tid] - eta[tid] * eta[tid]) / 2.0 - x_disp;
        y[tid] = xi[tid] * eta[tid];
    }
}

// CUDA kernel to generate the inferior part of the boundary by symmetry
__global__ void inferior(const double *x_sup, const double *y_sup, double *x_inf, double *y_inf, int size){
    int tid = blockIdx.x * blockDim.x +threadIdx.x;
    if (tid < size){
        x_inf[size - 1 - tid] = x_sup[tid];
        y_inf[size - 1 - tid] = -y_sup[tid];
    }
}

// CUDA kernel to join coordinates (x,y)
__global__ void combinePoints(const double *x_half, const double *y_half, const double *x_ref, const double *y_ref, Point *pts, int size_half){
    int tid = blockIdx.x * blockDim.x +threadIdx.x;
    if (tid < size_half){
        pts[tid] = {x_half[tid], y_half[tid]};
        pts[tid + size_half] = {x_ref[tid], y_ref[tid]}; 
    }
}

class ConfocalParabolicBilliard{
public:

    ConfocalParabolicBilliard(double xi0_, double eta0_, int num_points_)
        : xi0(xi0_), eta0(eta0_), num_points(num_points_)
    {
        if (num_points % 4 != 0)
            throw std::invalid_argument("num_points must be a multiple of 4");
        boundary = generateBoundaryPoints();
    }

    Point* getBoundary() {return boundary;}

private:
    double xi0, eta0;
    int num_points;
    Point* boundary;

    Point* generateBoundaryPoints() {
        int NUM_THREADS = 256;

        int    n_half = num_points / 2;  // n_half points for the sup part and n_half for the inf part
        double sum    = xi0 + eta0;
        // Determine how many points for each segment of the boundary according to the relative values of xi0 and eta0
        int    n_eta  = static_cast<int>(std::round(n_half * xi0  / sum));
        int    n_xi   = static_cast<int>(std::round(n_half * eta0 / sum));
        double step   = sum / n_half;

        // Construct the superior part of the boundary
        // First segment. η = const (x<0). Left side
        double *h_xi_seg1, *d_xi_seg1, *h_eta_seg1, *d_eta_seg1; // Initialize pointers for cpu (host) and gpu (device)
        size_t bytes = sizeof(double) * n_eta;
        h_xi_seg1 = (double*)malloc(bytes);
        h_eta_seg1 = (double*)malloc(bytes);
        cudaMalloc(&d_xi_seg1, bytes);
        cudaMalloc(&d_eta_seg1, bytes);
        int NUM_BLOCKS = (n_eta + NUM_THREADS - 1) / NUM_THREADS;
        linspace<<<NUM_BLOCKS, NUM_THREADS>>>(d_xi_seg1, step/2.0, n_eta, (xi0 - step) / (n_eta - 1));
        fill_vec<<<NUM_BLOCKS, NUM_THREADS>>>(d_eta_seg1, eta0, n_eta);
        cudaDeviceSynchronize();
        cudaMemcpy(h_xi_seg1, d_xi_seg1, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_eta_seg1, d_eta_seg1, bytes, cudaMemcpyDeviceToHost);

        // Second segment. ξ = const (x>0). Right side
        double *h_eta_seg2, *d_eta_seg2, *h_xi_seg2, *d_xi_seg2;
        bytes = sizeof(double) * n_xi;
        h_eta_seg2 = (double*)malloc(bytes);
        h_xi_seg2 = (double*)malloc(bytes);
        cudaMalloc(&d_eta_seg2, bytes);
        cudaMalloc(&d_xi_seg2, bytes);
        NUM_BLOCKS = (n_xi + NUM_THREADS - 1) / NUM_THREADS;
        linspace<<<NUM_BLOCKS, NUM_THREADS>>>(d_eta_seg2, eta0 - step/2.0, n_xi, (step - eta0) / (n_xi - 1));
        fill_vec<<<NUM_BLOCKS, NUM_THREADS>>>(d_xi_seg2, xi0, n_xi);
        cudaDeviceSynchronize();
        cudaMemcpy(h_eta_seg2, d_eta_seg2, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_xi_seg2, d_xi_seg2, bytes, cudaMemcpyDeviceToHost);
        
        // Correction in the x-axis in order to center the figure
        double x_disp = (h_xi_seg2[0]*h_xi_seg2[0] - h_eta_seg2[0]*h_eta_seg2[0]) / 2.0;

        // Join both segments of the superior part
        double *h_xi_half, *d_xi_half, *h_eta_half, *d_eta_half;
        bytes = sizeof(double) * (n_xi + n_eta);
        h_xi_half = (double*)malloc(bytes);
        h_eta_half = (double*)malloc(bytes);
        cudaMalloc(&d_xi_half, bytes);
        cudaMalloc(&d_eta_half, bytes);
        NUM_BLOCKS = (n_xi + n_eta + NUM_THREADS - 1) / NUM_THREADS;
        concatenate<<<NUM_BLOCKS, NUM_THREADS>>>(d_xi_seg1, n_eta, d_xi_seg2, n_xi, d_xi_half, n_xi + n_eta);
        concatenate<<<NUM_BLOCKS, NUM_THREADS>>>(d_eta_seg1, n_eta, d_eta_seg2, n_xi, d_eta_half, n_xi + n_eta);
        cudaDeviceSynchronize();
        cudaMemcpy(h_xi_half, d_xi_half, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_eta_half, d_eta_half, bytes, cudaMemcpyDeviceToHost);

        // Transform from parabolic coordinates to cartesian coordinates
        double *h_x_half, *d_x_half, *h_y_half, *d_y_half;
        h_x_half = (double*)malloc(bytes);
        h_y_half = (double*)malloc(bytes);
        cudaMalloc(&d_x_half, bytes);
        cudaMalloc(&d_y_half, bytes);
        ParToCart<<<NUM_BLOCKS, NUM_THREADS>>>(d_xi_half, d_eta_half, d_x_half, d_y_half, n_xi + n_eta, x_disp);
        cudaDeviceSynchronize();
        cudaMemcpy(h_x_half, d_x_half, bytes, cudaMemcpyDeviceToHost);

        // Generate the inferior part by symmetry
        double *h_x_ref, *d_x_ref, *h_y_ref, *d_y_ref;
        h_x_ref = (double*)malloc(bytes);
        h_y_ref = (double*)malloc(bytes);
        cudaMalloc(&d_x_ref, bytes);
        cudaMalloc(&d_y_ref, bytes);
        inferior<<<NUM_BLOCKS, NUM_THREADS>>>(d_x_half, d_y_half, d_x_ref, d_y_ref, n_xi + n_eta);
        cudaDeviceSynchronize();
        cudaMemcpy(h_x_ref, d_x_ref, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_y_ref, d_y_ref, bytes, cudaMemcpyDeviceToHost);

        // Construct the entire vector of points
        Point *h_pts, *d_pts;
        bytes = 2 * (n_xi + n_eta) * sizeof(Point);
        h_pts = (Point*)malloc(bytes);
        cudaMalloc(&d_pts, bytes);
        combinePoints<<<NUM_BLOCKS, NUM_THREADS>>>(d_x_half, d_y_half, d_x_ref, d_y_ref, d_pts, n_xi + n_eta);
        cudaDeviceSynchronize();
        cudaMemcpy(h_pts, d_pts, bytes, cudaMemcpyDeviceToHost);

        // Free allocated host memory
        free(h_xi_seg1); free(h_eta_seg1);
        free(h_xi_seg2); free(h_eta_seg2);
        free(h_xi_half); free(h_eta_half);
        free(h_x_half);  free(h_y_half);
        free(h_x_ref);   free(h_y_ref);

        // Free allocated device memory
        cudaFree(d_xi_seg1); cudaFree(d_eta_seg1);
        cudaFree(d_xi_seg2); cudaFree(d_eta_seg2);
        cudaFree(d_xi_half); cudaFree(d_eta_half);
        cudaFree(d_x_half);  cudaFree(d_y_half);
        cudaFree(d_x_ref);   cudaFree(d_y_ref);
        cudaFree(d_pts);
        
        return h_pts;
    }
};
}

#endif
