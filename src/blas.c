#include "blas.h"

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] = s1*out[out_index] + s2*add[add_index];
                }
            }
        }
    }
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial)
{
    int b,f,i;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < spatial; ++i){
            float sum = 0;
            for(f = 0; f < filters; ++f){
                int index = b*filters*spatial + f*spatial + i;
                sum += powf(x[index], 2);
            }
            sum = sqrtf(sum);
            for(f = 0; f < filters; ++f){
                int index = b*filters*spatial + f*spatial + i;
                x[index] /= sum;
                dx[index] = (1 - x[index]) / sum;
            }
        }
    }
}


void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i){
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i){
            OUT[index++] = Y[j*NY + i];
        }
    }
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}

void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}

void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p) - (1-t)*log(1-p);
        delta[i] = t-p;
    }
}

void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h*stride; ++j){
                for(i = 0; i < w*stride; ++i){
                    int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if(forward) out[out_index] = scale*in[in_index];
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}


#ifdef OPENCL

void scale_bias_cl(cl_mem output, cl_mem biases, int batch, int n, int size,int off1,int off2)
{
    cl_int err;
    size_t globalSize[3]={((size-1)/BLOCK + 1)*BLOCK,n,batch};
    size_t localSize[3]={BLOCK,1,1};

    *clKernel=clCreateKernel(*clProgram,"scale_bias_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&output);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&biases);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&n);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&size);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"scale_bias_cl");
}

void backward_scale_cl(cl_mem x_norm, cl_mem delta, int batch, int n, int size, cl_mem scale_updates,int off1,int off2,int off3)
{
    cl_int err;
    size_t globalSize[3]={n*BLOCK,1,1};
    size_t localSize[3]={BLOCK,1,1};

    *clKernel=clCreateKernel(*clProgram,"backward_scale_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&x_norm);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&delta);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&n);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&size);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_mem),&scale_updates);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off3);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"backward_scale_cl");
}

void add_bias_cl(cl_mem output, cl_mem biases, int batch, int n, int size,int off1,int off2)
{
    int num = n*size*batch;
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(num,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"add_bias_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&output);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&biases);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&n);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&size);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"add_bias_cl");
}

void backward_bias_cl(cl_mem bias_updates, cl_mem delta, int batch, int n, int size,int off1,int off2)
{
    
    if(size == 1){
        cl_int err;
        size_t globalSize[3],localSize[3];
        setWorkItemSize(n,globalSize,localSize);
        *clKernel=clCreateKernel(*clProgram,"backward_bias_conn_opencl",&err);
        err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&bias_updates);
        err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&delta);
        err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&batch);
        err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&n);
        err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&off1);
        err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off2);
        err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
        cl_error(err,"backward_bias_cl");
    }else{
        cl_int err;
        size_t globalSize[3]={n*BLOCK,1,1};
        size_t localSize[3]={BLOCK,1,1};

        *clKernel=clCreateKernel(*clProgram,"backward_bias_opencl",&err);
        err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&bias_updates);
        err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&delta);
        err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&batch);
        err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&n);
        err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&size);
        err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
        err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
        err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
        cl_error(err,"backward_bias_cl");
    }
}

void adam_cl(int n, cl_mem x, cl_mem m, cl_mem v, float B1, float B2, float rate, float eps, int t,int off1,int off2,int off3)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(n,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"adam_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&n);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&x);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&m);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&v);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_float),&B1);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_float),&B2);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_float),&rate);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_float),&eps);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&t);
    err|=clSetKernelArg(*clKernel,9,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,10,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,11,sizeof(cl_int),&off3);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"adam_cl");
}

void adam_update_cl(cl_mem w, cl_mem d, cl_mem m, cl_mem v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t,int off1,int off2,int off3,int off4)
{
    scal_cl(n, B1, m, 1,off3);
    scal_cl(n, B2, v, 1,off4);
    axpy_cl(n, -decay*batch, w, 1, d, 1,off1,off2);

    axpy_cl(n, (1-B1), d, 1, m, 1,off2,off3);
    mul_cl(n, d, 1, d, 1,off2,off2);
    axpy_cl(n, (1-B2), d, 1, v, 1,off2,off4);

    adam_cl(n, w, m, v, B1, B2, rate, eps, t,off1,off3,off4);
    fill_cl(n, 0, d, 1,off2);
}

void normalize_delta_cl(cl_mem x, cl_mem mean, cl_mem variance, cl_mem mean_delta, cl_mem variance_delta, int batch, int filters, int spatial, cl_mem delta,
        int off1,int off2,int off3,int off4,int off5,int off6)
{
    size_t N = batch*filters*spatial;
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"normalize_delta_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&x);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&mean);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&variance);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_mem),&mean_delta);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_mem),&variance_delta);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&filters);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&spatial);
    err|=clSetKernelArg(*clKernel,9,sizeof(cl_mem),&delta);
    err|=clSetKernelArg(*clKernel,10,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,11,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,12,sizeof(cl_int),&off3);
    err|=clSetKernelArg(*clKernel,13,sizeof(cl_int),&off4);
    err|=clSetKernelArg(*clKernel,14,sizeof(cl_int),&off5);
    err|=clSetKernelArg(*clKernel,15,sizeof(cl_int),&off6);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"normalize_delta_cl");
}

void mean_delta_cl(cl_mem delta, cl_mem variance, int batch, int filters, int spatial, cl_mem mean_delta,int off1,int off2,int off3)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(filters,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"mean_delta_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&delta);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&variance);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&filters);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&spatial);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_mem),&mean_delta);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off3);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"mean_delta_cl");
}

void fast_mean_delta_cl(cl_mem delta,cl_mem variance, int batch, int filters, int spatial,cl_mem mean_delta,int off1,int off2,int off3)
{
    cl_int err;
    size_t globalSize[3]={filters*BLOCK,1,1};
    size_t localSize[3]={BLOCK,1,1};

    *clKernel=clCreateKernel(*clProgram,"fast_mean_delta_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&delta);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&variance);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&filters);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&spatial);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_mem),&mean_delta);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off3);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"fast_mean_delta_cl");
}

void fast_variance_delta_cl(cl_mem x,cl_mem delta,cl_mem mean,cl_mem variance, int batch, int filters, int spatial,cl_mem variance_delta,
        int off1,int off2,int off3,int off4,int off5)
{
    cl_int err;
    size_t globalSize[3]={filters*BLOCK,1,1};
    size_t localSize[3]={BLOCK,1,1};

    *clKernel=clCreateKernel(*clProgram,"fast_variance_delta_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&x);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&delta);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&mean);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&variance);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&filters);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&spatial);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_mem),&variance_delta);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,9,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,10,sizeof(cl_int),&off3);
    err|=clSetKernelArg(*clKernel,11,sizeof(cl_int),&off4);
    err|=clSetKernelArg(*clKernel,12,sizeof(cl_int),&off5);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"fast_variance_delta_cl");
}

void normalize_cl(cl_mem x, cl_mem mean, cl_mem variance, int batch, int filters, int spatial,int off1,int off2,int off3)
{
    size_t N = batch*filters*spatial;
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"normalize_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&x);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&mean);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&variance);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&filters);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&spatial);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,9,sizeof(cl_int),&off3);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"normalize_cl");
}

void l2normalize_cl(cl_mem x, cl_mem dx, int batch, int filters, int spatial,int off1,int off2)
{
    size_t N = batch*spatial;
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"l2norm_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&x);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&dx);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&filters);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&spatial);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"l2normalize_cl");
}

void fast_mean_cl(cl_mem x, int batch, int filters, int spatial,cl_mem mean,int off1,int off2)
{
    cl_int err;
    size_t globalSize[3]={filters*BLOCK,1,1};
    size_t localSize[3]={BLOCK,1,1};

    *clKernel=clCreateKernel(*clProgram,"fast_mean_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&x);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&filters);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&spatial);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_mem),&mean);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"fast_mean_cl");
}

void fast_variance_cl(cl_mem x,cl_mem mean, int batch, int filters, int spatial,cl_mem variance,int off1,int off2,int off3)
{
    cl_int err;
    size_t globalSize[3]={filters*BLOCK,1,1};
    size_t localSize[3]={BLOCK,1,1};
    
    *clKernel=clCreateKernel(*clProgram,"fast_variance_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&x);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&mean);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&filters);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&spatial);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_mem),&variance);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off3);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"fast_variance_cl");
}

void mean_cl(cl_mem x, int batch, int filters, int spatial, cl_mem mean,int off1,int off2)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(filters,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"mean_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&x);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&filters);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&spatial);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_mem),&mean);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"mean_cl");
}

void variance_cl(cl_mem x, cl_mem mean, int batch, int filters, int spatial, cl_mem variance,int off1,int off2,int off3)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(filters,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"variance_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&x);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&mean);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&filters);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&spatial);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_mem),&variance);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off3);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"variance_cl");
}

void axpy_cl(int N, float ALPHA, cl_mem  X, int INCX, cl_mem  Y, int INCY, int OFFX, int OFFY)
{
    axpy_cl_offset(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
}

void pow_cl(int N, float ALPHA, cl_mem  X, int INCX, cl_mem  Y, int INCY,int off1,int off2)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"pow_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_float),&ALPHA);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&INCX);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_mem),&Y);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&INCY);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"pow_cl");
}

void axpy_cl_offset(int N, float ALPHA, cl_mem  X, int OFFX, int INCX, cl_mem  Y, int OFFY, int INCY)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"axpy_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_float),&ALPHA);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&OFFX);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&INCX);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_mem),&Y);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&OFFY);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&INCY);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"axpy_cl_offset");
}

void copy_cl(int N, cl_mem  X, int INCX, cl_mem  Y, int INCY, int OFFX, int OFFY)
{
    copy_cl_offset(N, X, OFFX, INCX, Y, OFFY, INCY);
}

void mul_cl(int N, cl_mem  X, int INCX, cl_mem  Y, int INCY,int off1,int off2)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"mul_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&INCX);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&Y);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&INCY);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"mul_cl");
}

void copy_cl_offset(int N, cl_mem  X, int OFFX, int INCX, cl_mem  Y, int OFFY, int INCY)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"copy_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&OFFX);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&INCX);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_mem),&Y);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&OFFY);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&INCY);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"copy_cl_offset");
}

void flatten_cl(cl_mem x, int spatial, int layers, int batch, int forward, cl_mem out,int off1,int off2)
{
    int size = spatial*batch*layers;
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(size,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"flatten_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&size);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&x);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&spatial);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&layers);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&forward);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_mem),&out);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"flatten_cl");
}

void reorg_cl(cl_mem x, int w, int h, int c, int batch, int stride, int forward, cl_mem out,int off1,int off2)
{
    int size = w*h*c*batch;
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(size,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"reorg_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&size);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&x);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&w);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&h);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&c);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&stride);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&forward);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_mem),&out);
    err|=clSetKernelArg(*clKernel,9,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,10,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"reorg_cl");
}

void mask_cl(int N, cl_mem  X, float mask_num, cl_mem  mask, float val,int off1,int off2)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"mask_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_float),&mask_num);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&mask);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_float),&val);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"mask_cl");
}

void scale_mask_cl(int N, cl_mem  X, float mask_num, cl_mem  mask, float scale,int off1,int off2)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"scale_mask_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_float),&mask_num);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&mask);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_float),&scale);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"scale_mask_cl");
}

void const_cl(int N, float ALPHA, cl_mem  X, int INCX,int off1)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"const_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_float),&ALPHA);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&INCX);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&off1);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"const_cl");
}

void constrain_cl(int N, float ALPHA, cl_mem  X, int INCX,int off1)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"constrain_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_float),&ALPHA);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&INCX);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&off1);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"constrain_cl");
}

void add_cl(int N, float ALPHA, cl_mem  X, int INCX,int off1)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"add_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_float),&ALPHA);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&INCX);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&off1);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"add_cl");
}

void scal_cl(int N, float ALPHA, cl_mem  X, int INCX,int off1)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"scal_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_float),&ALPHA);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&INCX);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&off1);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"scal_cl");
}

void supp_cl(int N, float ALPHA, cl_mem  X, int INCX,int off1)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"supp_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_float),&ALPHA);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&INCX);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&off1);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"supp_cl");
}

void fill_cl(int N, float ALPHA, cl_mem  X, int INCX,int off1)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(N,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"fill_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&N);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_float),&ALPHA);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&INCX);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&off1);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"fill_cl");
}

void shortcut_cl(int batch, int w1, int h1, int c1, cl_mem add, int w2, int h2, int c2, float s1, float s2, cl_mem out,int off1,int off2)
{
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    int size = batch * minw * minh * minc;
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(size,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"shortcut_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&size);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_int),&minw);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&minh);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&minc);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&stride);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&sample);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&w1);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&h1);
    err|=clSetKernelArg(*clKernel,9,sizeof(cl_int),&c1);
    err|=clSetKernelArg(*clKernel,10,sizeof(cl_mem),&add);
    err|=clSetKernelArg(*clKernel,11,sizeof(cl_int),&w2);
    err|=clSetKernelArg(*clKernel,12,sizeof(cl_int),&h2);
    err|=clSetKernelArg(*clKernel,13,sizeof(cl_int),&c2);
    err|=clSetKernelArg(*clKernel,14,sizeof(cl_float),&s1);
    err|=clSetKernelArg(*clKernel,15,sizeof(cl_float),&s2);
    err|=clSetKernelArg(*clKernel,16,sizeof(cl_mem),&out);
    err|=clSetKernelArg(*clKernel,17,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,18,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"shortcut_cl");
}

void smooth_l1_cl(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error,int off1,int off2,int off3,int off4)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(n,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"smooth_l1_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&n);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&pred);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&truth);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&delta);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_mem),&error);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off3);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off4);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"smooth_l1_cl");
}

void softmax_x_ent_cl(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error,int off1,int off2,int off3,int off4)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(n,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"softmax_x_ent_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&n);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&pred);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&truth);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&delta);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_mem),&error);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off3);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off4);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"softmax_x_ent_cl");
}

void logistic_x_ent_cl(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error,int off1,int off2,int off3,int off4)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(n,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"logistic_x_ent_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&n);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&pred);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&truth);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&delta);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_mem),&error);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off3);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off4);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"logistic_x_ent_cl");
}

void l2_cl(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error,int off1,int off2,int off3,int off4)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(n,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"l2_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&n);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&pred);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&truth);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&delta);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_mem),&error);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off3);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off4);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"l2_cl");
}

void l1_cl(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error,int off1,int off2,int off3,int off4)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(n,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"l1_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&n);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&pred);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&truth);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&delta);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_mem),&error);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off3);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off4);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"l1_cl");
}

void wgan_cl(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error,int off1,int off2,int off3,int off4)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(n,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"wgan_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&n);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&pred);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&truth);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&delta);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_mem),&error);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off3);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off4);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"wgan_cl");
}

void deinter_cl(int NX, cl_mem X, int NY, cl_mem Y, int B, cl_mem OUT,int off1,int off2,int off3)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize((NX+NY)*B,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"deinter_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&NX);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&NY);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&Y);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&B);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_mem),&OUT);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off3);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"deinter_cl");
}

void inter_cl(int NX, cl_mem X, int NY, cl_mem Y, int B, cl_mem OUT,int off1,int off2,int off3)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize((NX+NY)*B,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"inter_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&NX);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&X);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&NY);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&Y);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&B);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_mem),&OUT);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off3);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"inter_cl");
}

void weighted_sum_cl(cl_mem a, cl_mem b, cl_mem s, int num, cl_mem c,int off1,int off2,int off3,int off4)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(num,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"weighted_sum_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&num);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&a);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&b);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&s);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_mem),&c);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&off3);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off4);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"weighted_sum_cl");
}

void weighted_delta_cl(cl_mem a, cl_mem b, cl_mem s, cl_mem da, cl_mem db, cl_mem ds, int num, cl_mem dc,int off1,int off2,int off3,int off4,int off5,int off6,int off7)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(num,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"weighted_delta_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&num);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&a);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&b);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&s);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_mem),&da);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_mem),&db);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_mem),&ds);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_mem),&dc);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,9,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,10,sizeof(cl_int),&off3);
    err|=clSetKernelArg(*clKernel,11,sizeof(cl_int),&off4);
    err|=clSetKernelArg(*clKernel,12,sizeof(cl_int),&off5);
    err|=clSetKernelArg(*clKernel,13,sizeof(cl_int),&off6);
    err|=clSetKernelArg(*clKernel,14,sizeof(cl_int),&off7);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"weighted_delta_cl");
}

void mult_add_into_cl(int num, cl_mem a, cl_mem b, cl_mem c,int off1,int off2,int off3)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(num,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"mult_add_into_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&num);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&a);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_mem),&b);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_mem),&c);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&off2);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&off3);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"mult_add_into_cl");
}

void softmax_tree(cl_mem input, int spatial, int batch, int stride, float temp,cl_mem output, tree hier,int off1,int off2)
{
    cl_mem tree_groups_size = cl_make_int_array(hier.group_size, hier.groups);
    cl_mem tree_groups_offset = cl_make_int_array(hier.group_offset, hier.groups);
    /*
       static int *tree_groups_size = 0;
       static int *tree_groups_offset = 0;
       if(!tree_groups_size){
       tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
       tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
       }
     */
    int num = spatial*batch*hier.groups;
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(num,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"softmax_tree_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&input);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_int),&spatial);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&stride);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_float),&temp);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_mem),&output);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&hier.groups);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_mem),&tree_groups_size);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_mem),&tree_groups_offset);
    err|=clSetKernelArg(*clKernel,9,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,10,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"softmax_tree");
    cl_free(tree_groups_size);
    cl_free(tree_groups_offset);
}

void softmax_cl(cl_mem input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem output,int off1,int off2)
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(batch*groups,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"softmax_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_mem),&input);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_int),&n);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&batch_offset);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&groups);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&group_offset);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&stride);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_float),&temp);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_mem),&output);
    err|=clSetKernelArg(*clKernel,9,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,10,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"softmax_cl");
}

void upsample_cl(cl_mem in, int w, int h, int c, int batch, int stride, int forward, float scale, cl_mem out,int off1,int off2)
{
    size_t size = w*h*c*batch*stride*stride;
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(size,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram,"upsample_opencl",&err);
    err|=clSetKernelArg(*clKernel,0,sizeof(cl_int),&size);
    err|=clSetKernelArg(*clKernel,1,sizeof(cl_mem),&in);
    err|=clSetKernelArg(*clKernel,2,sizeof(cl_int),&w);
    err|=clSetKernelArg(*clKernel,3,sizeof(cl_int),&h);
    err|=clSetKernelArg(*clKernel,4,sizeof(cl_int),&c);
    err|=clSetKernelArg(*clKernel,5,sizeof(cl_int),&batch);
    err|=clSetKernelArg(*clKernel,6,sizeof(cl_int),&stride);
    err|=clSetKernelArg(*clKernel,7,sizeof(cl_int),&forward);
    err|=clSetKernelArg(*clKernel,8,sizeof(cl_float),&scale);
    err|=clSetKernelArg(*clKernel,9,sizeof(cl_mem),&out); 
    err|=clSetKernelArg(*clKernel,10,sizeof(cl_int),&off1);
    err|=clSetKernelArg(*clKernel,11,sizeof(cl_int),&off2);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"upsample_cl");
}
#endif