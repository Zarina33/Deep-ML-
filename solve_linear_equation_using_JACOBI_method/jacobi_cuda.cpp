"""
Solve Linear Equations using Jacobi Method
Medium
Linear Algebra


Write a Python function that uses the Jacobi method to solve a system of linear equations given by Ax = b. The function should iterate n times, rounding each intermediate solution to four decimal places, and return the approximate solution x.
Initialize the solution vector x to all zeros before beginning the iterations.
Example:
Input:
A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]], b = [-1, 2, 3], n=2
Output:
[0.146, 0.2032, -0.5175]
Reasoning:
The Jacobi method iteratively solves each equation for x[i] using the formula x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i)), where a_ii is the diagonal element of A and a_ij are the off-diagonal elements.


"""
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

// Функция для имитации округления Python (к ближайшему четному)
__device__ double python_round(double val) {
    double factor = 10000.0;
    double x = val * factor;
    double rounded = round(x);
    // Проверка на .5 (если расстояние до округленного в точности 0.5)
    if (abs(x - rounded) == 0.5) {
        if (fmod(rounded, 2.0) != 0.0) {
            rounded = (x > rounded) ? rounded + 1.0 : rounded - 1.0;
        }
    }
    return rounded / factor;
}

__global__ void jacobi_iteration_kernel(
    const float* A,
    const float* b,
    const double* x_old,
    double* x_new,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double sum = (double)b[i];
        for (int j = 0; j < n; j++) {
            if (i != j) {
                sum -= (double)A[i * n + j] * x_old[j];
            }
        }
        double val = sum / (double)A[i * n + i];
        
        // Используем специальное округление
        x_new[i] = python_round(val);
    }
}

std::vector<float> solve_jacobi(
    const std::vector<std::vector<float>>& A_host,
    const std::vector<float>& b_host,
    int n_iterations
) {
    int n = b_host.size();
    
    std::vector<float> A_flat(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A_flat[i * n + j] = A_host[i][j];

    float *d_A, *d_b;
    double *d_x_old, *d_x_new;

    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_x_old, n * sizeof(double));
    cudaMalloc(&d_x_new, n * sizeof(double));

    cudaMemcpy(d_A, A_flat.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_host.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_x_old, 0, n * sizeof(double));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (int it = 0; it < n_iterations; it++) {
        jacobi_iteration_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_b, d_x_old, d_x_new, n);
        cudaDeviceSynchronize();
        std::swap(d_x_old, d_x_new);
    }

    std::vector<double> result_d(n);
    cudaMemcpy(result_d.data(), d_x_old, n * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<float> result(n);
    for(int i = 0; i < n; i++) result[i] = (float)result_d[i];

    cudaFree(d_A); cudaFree(d_b); cudaFree(d_x_old); cudaFree(d_x_new);
    return result;
}