#ifndef BWM_BOUNDARY_WALL_METHOD_HPP
#define BWM_BOUNDARY_WALL_METHOD_HPP

#include <boost/math/special_functions/hankel.hpp>
#include <iostream>
#include <complex>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include "ConfocalParabolicBilliard.cuh"

int NUM_THREADS = 128;                         // Number of threads per block

// Compute the distance between to point (Euclidean norm of a - b)
__device__ double distance(const Point &a, const Point &b){
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

namespace bwm {

//
__global__ void kernel_normT(const cuDoubleComplex* d_T, double* d_partial_sums, int N){
    extern __shared__ double sdata_normT[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Each thread computes the absolute value of it's element
    double val = 0.0;
    if (idx < N*N){
        val = cuCabs(d_T[idx]);
    }
    sdata_normT[tid] = val;
    __syncthreads();

    //
    for(int s = blockDim.x / 2; s > 0; s>>=1){
        if(tid < s){sdata_normT[tid] += sdata_normT[tid + s];}
        __syncthreads();
    }

    //
    if(tid == 0){d_partial_sums[blockIdx.x] = sdata_normT[0];}
}

__device__ cuDoubleComplex hankel_0_1(double x){
    // H_0^1(x) = J_0(x) + iY_0(x)
    return make_cuDoubleComplex(::j0(x), ::y0(x));
}

__device__ cuDoubleComplex handleDiagonal(double *lengths, int k, double alpha, int N){
    cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
    double avg = 0;
    for(int i = 0; i < N; ++i){
        avg += lengths[i];
    }
    avg = avg / N;

    return cuCmul(cuCmul(I, make_cuDoubleComplex(alpha / 4, 0.0)), hankel_0_1(k * avg / 2.0));
}

__device__ cuDoubleComplex green(double *lengths, const Point r1, const Point r2, double k, double alpha, int N){
    cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
    double R = distance(r1, r2);
    double arg = k * R;

    if (R < 1e-10) return handleDiagonal(lengths, k, alpha, N);

    return cuCmul(cuCmul(I, make_cuDoubleComplex(alpha / 4, 0.0)), hankel_0_1(arg));
}


//
__global__ void TotalWave(cuDoubleComplex *psi, cuDoubleComplex *Tphi, double *lengths, Point *obs, 
                        Point *boundary, double k, double alpha, double angle, size_t n_obs, int N){
    extern __shared__ cuDoubleComplex sdata[];

    int row = threadIdx.y;  // j dentro del bloque
    int col = blockIdx.x;   // Cada bloque maneja un punto de observación
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Inicializar memoria compartida
    sdata[tid] = make_cuDoubleComplex(0.0, 0.0);
    
    // Solo procesar si estamos dentro de los límites
    if(col < n_obs && row < N) {
        cuDoubleComplex G = green(lengths, obs[col], boundary[row], k, alpha, N);
        sdata[tid] = cuCmul(cuCmul(G, make_cuDoubleComplex(lengths[row], 0.0)), Tphi[row]);
    }
    __syncthreads();

    // Reducción en paralelo
    for(int s = blockDim.y/2; s > 0; s >>= 1) {
        if(row < s && (row + s) < N) {
            sdata[tid] = cuCadd(sdata[tid], sdata[tid + s * blockDim.x]);
        }
        __syncthreads();
    }

    // Escribir resultado
    if(row == 0 && col < n_obs) {
        // Sumar la onda incidente al resultado dispersado
        double phase = k * (obs[col].x * cos(angle) + obs[col].y * sin(angle));
        cuDoubleComplex incident = make_cuDoubleComplex(cos(phase), sin(phase));
        psi[col] = cuCadd(incident, sdata[tid]);
    }
}

//
__global__ void kernelMatrixM(cuDoubleComplex *M, double *lengths, Point *boundary, double k, double alpha, int N, int N2){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < N2){
        int i = tid % N;
        int j = tid / N;
        M[tid] = (i == j ? handleDiagonal(lengths,k, alpha, N) 
                         : cuCmul(green(lengths,boundary[i],boundary[j],k,alpha,N), make_cuDoubleComplex(lengths[j], 0.0)));
    }
}

// CUDA kernel to compute the lenght of the segments of the boundary
__global__ void kernelLenght(double *L, const Point *boundary, const size_t N){
    int tid = blockIdx.x * blockDim.x +threadIdx.x;
    if(tid < N){
        size_t j = (tid + 1) % N;                          // The modulus operation closes the boundary
        double dx = boundary[j].x - boundary[tid].x;
        double dy = boundary[j].y - boundary[tid].y;
        L[tid] = sqrt(dx*dx + dy*dy);   // Distance between consecutive boundary positions
    }
}

__device__ double funcion(double x){
    return j0(x);
}

__global__ void kernelPrueba(double *Y, double *X, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N){
        Y[tid] = funcion(X[tid]);
    }
}


class BoundaryWallMethod{
public:
    cublasHandle_t handle;
    cusolverDnHandle_t cusolver_handle;
    cusolverDnParams_t params;

    struct Spectrum{
        std::vector<double> k_values;
        std::vector<double> norm_values;
        std::vector<double> resonances;
    };

    BoundaryWallMethod(Point* boundary_points,
                       size_t boundary_size,
                       double* gamma_vals,
                       size_t gamma_size,
                       double k_, double angle_,
                       double alpha_ = 1.0)
      : boundary(boundary_points),
        N(boundary_size),
        k(k_), angle(angle_), alpha(alpha_)
    {   
        //
        cublasCreate(&handle);
        cusolverDnCreate(&cusolver_handle);
        cusolverDnCreateParams(&params);

        gamma = (double*)malloc(N*sizeof(double));
        if (gamma_size == 1){
            double value = gamma_vals[0];
            for(int i = 0; i < N; ++i){
                gamma[i] = value;
            }
        }
        else {
            for(int i = 0; i < N; ++i){
                gamma[i] = gamma_vals[i];
            }
        }

        kx = k * std::cos(angle);
        ky = k * std::sin(angle);

        segment_lengths = calculateSegmentLenghts();
        M_flat          = buildMMatrixFlat();
        T               = buildTMatrix();
    }

    cuDoubleComplex incidentWave(double x, double y) const{
        double phase = kx * x + ky * y;
        return make_cuDoubleComplex(cos(phase), sin(phase));
    }

    Spectrum scanSpectrum(double k_min=0.1, double k_max=2.0,
                          int num_points=100,
                          double refine_window=0.05,
                          int refine_iterations=2,
                          double tolerance=1e-4)
    {
        auto [ks, norms] = initial_scan(k_min, k_max, num_points);
        auto peaks       = findPeaks(ks, norms, k_min, k_max);
        auto refined     = refineResonances(peaks,
                                            refine_window,
                                            refine_iterations,
                                            tolerance,
                                            num_points/10);
        return {ks, norms, refined};
    }

    cuDoubleComplex* computeScatteredWave(Point *obs, size_t n_obs){
        // Compute the incident wave
        cuDoubleComplex *h_phi, *d_phi;
        size_t bytes = N * sizeof(cuDoubleComplex);
        h_phi = (cuDoubleComplex*)malloc(bytes);
        cudaMalloc(&d_phi, bytes);
        for(size_t i = 0; i < N; ++i){
            h_phi[i] = incidentWave(boundary[i].x, boundary[i].y);
        }
        cudaMemcpy(d_phi, h_phi, bytes, cudaMemcpyHostToDevice);

        // Multiply the interaction matrix T with the vector phi
        cuDoubleComplex *d_Tphi, *d_T;
        cudaMalloc(&d_Tphi, bytes);
        cudaMalloc(&d_T, N * bytes);
        cudaMemcpy(d_T, T, N * bytes, cudaMemcpyHostToDevice);
        cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex zero  = make_cuDoubleComplex(0.0, 0.0);
        cublasZgemv(handle, CUBLAS_OP_N, N, N, &one, d_T, N, d_phi, 1, &zero, d_Tphi, 1);

        // Evaluate the total wave over every observation point
        cuDoubleComplex *psi, *d_psi;
        psi = (cuDoubleComplex*)malloc(n_obs * sizeof(cuDoubleComplex));
        cudaMalloc(&d_psi, n_obs * sizeof(cuDoubleComplex));
        cudaMemset(d_psi, 0, n_obs * sizeof(cuDoubleComplex));
        dim3 blocks(n_obs, 1);  // One block per observation point
        dim3 threads(1, 256);   // 256 threads in the Y-dimention
        size_t shared_mem = threads.y * sizeof(cuDoubleComplex);

        // Allocate memory and copy to the device for segment_lengths, obs and boundary
        double *d_segment_lengths;
        Point *d_obs, *d_boundary;
        cudaMalloc(&d_segment_lengths, N * sizeof(double));
        cudaMalloc(&d_obs, n_obs * sizeof(Point));
        cudaMalloc(&d_boundary, N * sizeof(Point));
        cudaMemcpy(d_segment_lengths, segment_lengths, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_obs, obs, n_obs * sizeof(Point), cudaMemcpyHostToDevice);
        cudaMemcpy(d_boundary, boundary, N * sizeof(Point), cudaMemcpyHostToDevice);

        TotalWave<<<blocks, threads, shared_mem>>>(d_psi, d_Tphi, d_segment_lengths, d_obs, d_boundary, k, alpha, angle, n_obs, N);
        cudaMemcpy(psi, d_psi, n_obs * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        // Free memory
        cudaFree(d_T); cudaFree(d_phi); cudaFree(d_Tphi); cudaFree(d_psi); free(h_phi);

        return psi;
    }

private:
    // Index of a 2D matrix with column major
    size_t idx(size_t i, size_t j){return i + j * N;}
    // Given the index, calculate (i,j) position in the matrix (column major)
    std::pair<size_t, size_t> ij(size_t id){return {id%N, id/N}; }

    // Compute the lenght of the segments of the boundary
    double* calculateSegmentLenghts() {
        double *h_L, *d_L;
        Point *d_boundary;
        size_t bytes = N * sizeof(double);
        h_L = (double*)malloc(bytes);
        cudaMalloc(&d_L, bytes);
        cudaMalloc(&d_boundary, N*sizeof(Point));
        cudaMemcpy(d_boundary, boundary, N*sizeof(Point), cudaMemcpyHostToDevice);
        int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
        kernelLenght<<<NUM_BLOCKS, NUM_THREADS>>>(d_L, d_boundary, N);
        cudaDeviceSynchronize();

        // Verifica errores del kernel
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Error en kernelLenght: " << cudaGetErrorString(err) << std::endl;
        }

        cudaMemcpy(h_L, d_L, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_L);

        return h_L;
    }

    // Treat the points near the boundary by approximating them to the Green function in a mean boundary point times the segment lenght
    cuDoubleComplex handleDiagonal(){
        std::vector<double> lengths(segment_lengths, segment_lengths + N);
        double avg = std::accumulate(lengths.begin(), lengths.end(), 0.0) / N;

        //std::cout << "El valor de k*(avg/2.0) es: " << k*(avg/2.0) << "\n";

        std::complex<double> hankel = boost::math::cyl_hankel_1(0, k*(avg/2.0));
        double hankel_real = alpha * 0.25 * hankel.real() * avg;
        double hankel_imag = alpha * 0.25 * hankel.imag() * avg;
        return make_cuDoubleComplex(-hankel_imag, hankel_real);
    }

    // Build the boundary interaction matrix M
    cuDoubleComplex* buildMMatrixFlat(){
        cuDoubleComplex *M, *d_M;
        double *d_segment_lengths;
        Point *d_boundary;
        M = (cuDoubleComplex*)malloc(N * N * sizeof(cuDoubleComplex));
        cudaMalloc(&d_M, N * N * sizeof(cuDoubleComplex));
        cudaMalloc(&d_segment_lengths, N * sizeof(double));
        cudaMalloc(&d_boundary, N * sizeof(Point));
        cudaMemcpy(d_segment_lengths, segment_lengths, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_boundary, boundary, 2 * N * sizeof(double), cudaMemcpyHostToDevice);
        int NUM_BLOCKS = (N*N + NUM_THREADS - 1) / NUM_THREADS;
        kernelMatrixM<<<NUM_BLOCKS, NUM_THREADS>>>(d_M, d_segment_lengths, d_boundary, k, alpha, N, N*N);

        cudaMemcpy(M, d_M, N * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        return M;
    }

    // Build the interaction matrix
    cuDoubleComplex* buildTMatrix(){
        cuDoubleComplex *d_M;
        cudaMalloc(&d_M, N * N * sizeof(cuDoubleComplex));
        cudaMemcpy(d_M, M_flat, N * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        // Verify if all gamma values are infinite
        bool all_inf = std::all_of(gamma, gamma + N, [](double g){return std::isinf(g);});

        // Bytes for matrices
        size_t bytes_M = N*N*sizeof(cuDoubleComplex);
        
        // If all values are infinite, return the negative of the inverse of M
        if(all_inf){
            int* d_pivot;
            int* d_info;
            cuDoubleComplex *h_invM;
            h_invM = (cuDoubleComplex*)malloc(bytes_M);
            cudaMalloc(&d_pivot, N * sizeof(int));
            cudaMalloc(&d_info, sizeof(int));

            // Workspace query
            int Lwork;
            cuDoubleComplex *d_work;
            cusolverDnZgetrf_bufferSize(cusolver_handle, N, N, d_M, N, &Lwork);
            cudaMalloc(&d_work, Lwork * sizeof(cuDoubleComplex));

            // LU decomposition
            cusolverDnZgetrf(cusolver_handle, N, N, d_M, N, d_work, d_pivot, d_info);

            // Check if the factorization was successful
            int h_info = 0;
            cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
            if(h_info !=0){
                std::cerr << "LU factorization failed with info: = " << h_info << "\n";
                return nullptr;
            }

            // Allocate memory for the identity matrix
            cuDoubleComplex* d_I;
            cudaMalloc(&d_I, N * N * sizeof(cuDoubleComplex));

            // Create the identity matrix on host and then copy to device
            cuDoubleComplex *h_I = (cuDoubleComplex*)malloc(bytes_M);
            for(int i = 0; i < N*N; i++){h_I[i] = make_cuDoubleComplex(0.0, 0.0);}         // Set all the elements to 0
            for(int i = 0; i < N; i++){h_I[i + i * N] = make_cuDoubleComplex(1.0, 0.0);}   // Set the elements in the diagonal to 1
            cudaMemcpy(d_I, h_I, bytes_M, cudaMemcpyHostToDevice);

            // Solve M * X = I   (X will be the inverse of M) (The result will overwrite d_I)
            cusolverDnZgetrs(cusolver_handle, CUBLAS_OP_N, N, N, d_M, N, d_pivot, d_I, N, d_info); 

            // Check for errors
            cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
            if(h_info != 0){
                std::cerr << "Matrix inversion failed with info: " << h_info << "\n";
                return nullptr;
            }

            // Multiply by -1
            cuDoubleComplex alpha = make_cuDoubleComplex(-1.0, 0.0);
            cublasZscal(handle, N*N, &alpha, d_I, 1);

            // Copy results from the device to the host
            cudaMemcpy(h_invM, d_I, bytes_M, cudaMemcpyDeviceToHost);

            // Free memory 
            cudaFree(d_M); cudaFree(d_pivot); cudaFree(d_info); cudaFree(d_work); cudaFree(d_I); free(h_I);

            return h_invM;
        }

        // Otherwise, compute T = G * (I - G * M)^{-1}, where G is a diagonal matrix with the gamma values
        else{
            cuDoubleComplex *d_G;
            cudaMalloc(&d_G, bytes_M);
            cudaMemset(d_G, 0, bytes_M);  // Set the allocated memory with value 0

            // Create the identity matrix on host
            cuDoubleComplex *h_I = (cuDoubleComplex*)malloc(bytes_M);
            for(int i = 0; i < N*N; i++){h_I[i] = make_cuDoubleComplex(0.0, 0.0);}         // Identity matrix: Set all the elements to 0
            for(int i = 0; i < N; i++){h_I[i + i * N] = make_cuDoubleComplex(1.0, 0.0);}   // Identity matrix: Set the elements in the diagonal to 1

            // Copy gamma to G as a diagonal matrix in the host, then copy to device
            cuDoubleComplex *h_G = (cuDoubleComplex*)malloc(bytes_M);
            for(int i=0;i < N*N; i++){h_G[i] = make_cuDoubleComplex(0.0,0.0);}           // Set all elements to 0
            for(int i=0;i < N; i++){h_G[i + N*i] = make_cuDoubleComplex(gamma[i], 0.0);} // Set the diagonal to the gamma values
            cudaMemcpy(d_G, h_G, bytes_M, cudaMemcpyHostToDevice);

            // Compute G * M
            cuDoubleComplex *d_GM;
            cudaMalloc(&d_GM, bytes_M);
            cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
            cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, d_G, N, d_M, N, &zero, d_GM, N); // The result will be stored in d_GM

            // Compute I - GM where I is the identity matrix
            cuDoubleComplex *d_I_min_GM;
            cuDoubleComplex neg_one = make_cuDoubleComplex(-1.0, 0.0);
            cudaMalloc(&d_I_min_GM, bytes_M);
            cudaMemcpy(d_I_min_GM, h_I, bytes_M, cudaMemcpyHostToDevice);
            cublasZaxpy(handle, N*N, &neg_one, d_GM, 1, d_I_min_GM,1);   // The result will be stored in d_I_GM

            // Compute (I - G * M)^{-1} using cuSOLVER
            cuDoubleComplex *d_inv_IGM, *d_work;
            int Lwork; int *d_pivot, *d_info; int h_info = 0;
            cudaMalloc(&d_inv_IGM, bytes_M); cudaMalloc(&d_pivot, N*sizeof(int)); cudaMalloc(&d_info, sizeof(int));
            cusolverDnZgetrf_bufferSize(cusolver_handle, N, N, d_I_min_GM, N, &Lwork);                          // Workspace query
            cudaMalloc(&d_work, Lwork * sizeof(cuDoubleComplex));
            cusolverDnZgetrf(cusolver_handle, N, N, d_I_min_GM, N, d_work, d_pivot, d_info);                    // LU decomposition
            cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
            if(h_info != 0){
                std::cerr << "LU factorization failed with info: " << h_info << "\n";
                return nullptr;
            }
            cudaMemcpy(d_inv_IGM, h_I, bytes_M, cudaMemcpyHostToDevice);                          // Copy the identity matrix from h_I to d_inv
            cusolverDnZgetrs(cusolver_handle, CUBLAS_OP_N, N, N, d_I_min_GM, N, d_pivot, d_inv_IGM, N, d_info); // Compute the inverse in d_inv
            cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);                                   // Check for errors
            if(h_info != 0){ 
                std::cerr << "Matrix inversion failed with info: " << h_info << "\n";
                return nullptr;
            }

            // Compute T = G * (I - G * M)^{-1}
            cuDoubleComplex *d_T, *h_T;
            h_T = (cuDoubleComplex*)malloc(bytes_M);
            cudaMalloc(&d_T, bytes_M);
            cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, d_G, N, d_inv_IGM, N, &zero, d_T, N);  // Store G*(I-G*M)^{-1} in d_T

            // Copy the results of T from the device to the host
            cudaMemcpy(h_T, d_T, bytes_M, cudaMemcpyDeviceToHost);

            // Free memory
            cudaFree(d_G); cudaFree(d_GM); cudaFree(d_I_min_GM); cudaFree(d_inv_IGM); 
            cudaFree(d_work); cudaFree(d_pivot); cudaFree(d_info); cudaFree(d_T);
            free(h_I); free(h_G); 

            return h_T;
        }
    }
    
    double normT(cuDoubleComplex* h_T){
        int NUM_BLOCKS = (N*N + NUM_THREADS - 1) / NUM_THREADS;
        cuDoubleComplex *d_T; double *d_partial_sums;
        cudaMalloc(&d_T, N*N*sizeof(cuDoubleComplex));
        cudaMalloc(&d_partial_sums, NUM_BLOCKS * sizeof(double));
        cudaMemcpy(d_T, h_T, N*N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        kernel_normT<<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS*sizeof(double)>>>(d_T, d_partial_sums, N);

        double *h_partial_sums = (double*)malloc(NUM_BLOCKS * sizeof(double));
        cudaMemcpy(h_partial_sums, d_partial_sums, NUM_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);

        double total = 0.0;
        for (int i = 0; i < NUM_BLOCKS; ++i){
            total += h_partial_sums[i];
        }
        cudaFree(d_partial_sums); cudaFree(d_T);

        return total;
    }

    // Initial scan in order to find possible resonances given a wave number range
    std::pair<std::vector<double>, std::vector<double>> initial_scan(double k_min, double k_max, int np){
        // Initializes ks (stores the wave numbers) and norms (stores the norms of the matrix T evaluated in the wave numbers)
        double *ks    = (double*)malloc(np);
        double *norms = (double*)malloc(np);
        double dk = (k_max - k_min) / (np - 1);
        for (int i = 0; i < np; ++i){
            ks[i] = k_min + i * dk;
            updateK(ks[i]);
            norms[i] = normT(T);
        }
        std::vector<double> ks_vec(ks, ks + np);
        std::vector<double> norms_vec(norms, norms + np);
        return {ks_vec, norms_vec};
    }

    // Finds local peaks given the norms generated by initial scan (Not possible to do parallel)
    std::vector<double> findPeaks(const std::vector<double>& ks,
                                  const std::vector<double>& norms,
                                  double k_min, double k_max) const
    {   
        // Initialize auxiliary vectors
        std::vector<bool> valid(norms.size(), true);  // Indicates if the points can be considered in the peak search
        std::vector<double> peaks;                    // Stores the values k in which there is a peak in norms

        // This while loop will break up only if there are no peaks left to detect
        while (true) {
            double mv = -std::numeric_limits<double>::infinity(); // Stablish norms biggest value to -infinity
            int midx = -1;                                        // Index of the biggest value
            // Find the biggest value in norms
            for (size_t i=0;i<norms.size();++i){
                if (valid[i] && norms[i]>mv){
                    mv = norms[i];
                    midx = i;
                }
            }
            if (midx<0) break; // If nothing could be found, break the while loop

            // Deactivate the peak and it's near values
            valid[midx] = false;
            for (int i=midx+1; i+1<(int)norms.size(); ++i) {
                if (!valid[i]) continue;
                if (norms[i]>norms[i-1]) break;
                valid[i] = false;
            }
            for (int i=midx-1; i>0; --i) {
                if (!valid[i]) continue;
                if (norms[i]>norms[i+1]) break;
                valid[i] = false;
            }

            // Store the peak if it isn't in the extremes of the k values
            double kp = ks[midx];
            if (kp!=k_min && kp!=k_max)
                peaks.push_back(kp);
        }
        return peaks;
    }

    // Refine with more precision the resonances found by findpeaks()
    std::vector<double> refineResonances(const std::vector<double>& peaks,
                                         double window, int iter,
                                         double tol, int ppw)
    {
        std::vector<double> refined;        // This vector stores the refined frecuencies
        // Iterate over every aproximate peak
        for (double kg : peaks) {
            // ck is the actual value which will be refined and w is the initial width of the local search interval around ck
            double ck = kg, w = window;
            // The refine process will be repeated iter times, each iteration reduces the width w in order to obtain better accuracy
            for (int it=0; it<iter; ++it) {
                // kws : vector with ppw (points per window) elements centered at ck and of width 2w
                std::vector<double> kws(ppw), vals(ppw);
                const double W = (2*w)/(ppw-1);          // Auxiliary value
                for (int j=0;j<ppw;++j) {
                    kws[j] = ck - w + j*W;
                    updateK(kws[j]);
                    vals[j] = normT(T);     // Compute the norm as the sum of absolute values of every element of matrix T
                }
                // Choose the best local peak (k value which maximizes the "norm" of T inside the interval)
                auto im = std::max_element(vals.begin(), vals.end());
                double nk = kws[std::distance(vals.begin(), im)];     // New candidate to be the refined peak
                // Verify convergence
                if (std::abs(nk - ck) < tol) break;
                ck = nk; w *= 0.1;
            }
            refined.push_back(ck);  // Store the refined value
        }
        return refined;
    }

    // Updates the wave number and every computation which depends of it

    void updateK(double newk){
        k = newk;
        kx = k * std::cos(angle);
        ky = k * std::sin(angle);
        M_flat = buildMMatrixFlat();
        T      = buildTMatrix();
    }

    Point*            boundary;
    size_t            N;
    double*           gamma;
    double            k, angle, alpha, kx, ky;
    double*           segment_lengths;
    cuDoubleComplex*  M_flat;
    cuDoubleComplex*  T;
};

} 

#endif 
