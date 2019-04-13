#include "avgpool_layer.h"
#include "cuda.h"
#include <stdio.h>

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_avgpool_layer_gpu;
    l.backward_gpu = backward_avgpool_layer_gpu;
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    #ifdef OPENCL
    l.forward_cl = forward_avgpool_layer_cl;
    l.backward_cl = backward_avgpool_layer_cl;
    l.output_cl  = cl_make_array(l.output, output_size);
    l.delta_cl   = cl_make_array(l.delta, output_size);
    #endif
    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}

void forward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }
}

void backward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    }
}
#ifdef OPENCL
void forward_avgpool_layer_cl(avgpool_layer layer, network net)
{
    size_t n = layer.c*layer.batch;
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(n,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram, "forward_avgpool_layer_opencl", &err);
    err|=clSetKernelArg(*clKernel, 0, sizeof(cl_int), &n);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_int), &layer.w);
    err|=clSetKernelArg(*clKernel, 2, sizeof(cl_int), &layer.h);
    err|=clSetKernelArg(*clKernel, 3, sizeof(cl_int), &layer.c);
    err|=clSetKernelArg(*clKernel, 4, sizeof(cl_mem), &net.input_cl);
    err|=clSetKernelArg(*clKernel, 5, sizeof(cl_mem), &layer.output_cl);
    err|=clEnqueueNDRangeKernel(*clCommandQueue, *clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"forward_avgpool_layer_cl");
}
void backward_avgpool_layer_cl(avgpool_layer layer, network net)
{
    size_t n = layer.c*layer.batch;
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(n,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram, "backward_avgpool_layer_opencl", &err);
    err|=clSetKernelArg(*clKernel, 0, sizeof(cl_int), &n);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_int), &layer.w);
    err|=clSetKernelArg(*clKernel, 2, sizeof(cl_int), &layer.h);
    err|=clSetKernelArg(*clKernel, 3, sizeof(cl_int), &layer.c);
    err|=clSetKernelArg(*clKernel, 4, sizeof(cl_mem), &net.delta_cl);
    err|=clSetKernelArg(*clKernel, 5, sizeof(cl_mem), &layer.delta_cl);
    err|=clEnqueueNDRangeKernel(*clCommandQueue, *clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"backward_avgpool_layer_cl");
}
#endif
