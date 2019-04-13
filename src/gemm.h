#ifndef GEMM_H
#define GEMM_H

#ifdef OPENCL
        #include "opencl_tool.h"
#endif

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef GPU
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
#endif
#ifdef OPENCL
void gemm_cl(int TA, int TB, int M, int N, int K, float ALPHA, 
        cl_mem A, int lda, 
        cl_mem B, int ldb,
        float BETA,
        cl_mem C, int ldc,
        int offset_a,int offset_b,int offset_c);
void gemm_xx_cl(int M, int N, int K, float ALPHA, 
        cl_mem A, int lda, 
        cl_mem B, int ldb,
        cl_mem C, int ldc,char *kernelFun,
        int offset_a,int offset_b,int offset_c);
#endif
#endif
