#include "activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {0};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = calloc(batch*inputs, sizeof(float*));
    l.delta = calloc(batch*inputs, sizeof(float*));

    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
#ifdef OPENCL
    l.forward_cl = forward_activation_layer_cl;
    l.backward_cl = backward_activation_layer_cl;

    l.output_cl = cl_make_array(l.output, inputs*batch);
    l.delta_cl = cl_make_array(l.delta, inputs*batch);
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}

void forward_activation_layer(layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_activation_layer_gpu(layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_gpu(layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
#ifdef OPENCL
void activate_array_cl(cl_mem x, int n, ACTIVATION a,int off_x) 
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(n,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram, "activate_array_opencl", &err);
    err|=clSetKernelArg(*clKernel, 0, sizeof(cl_mem), &x);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_int), &n);
    err|=clSetKernelArg(*clKernel, 2, sizeof(ACTIVATION), &a);
    err|=clSetKernelArg(*clKernel, 3, sizeof(cl_int), &off_x);
    err|=clEnqueueNDRangeKernel(*clCommandQueue, *clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"activate_array_cl");
}
void gradient_array_cl(cl_mem x, int n, ACTIVATION a, cl_mem delta, int offset_x,int offset_delta) 
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(n,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram, "gradient_array_opencl", &err);
    err|=clSetKernelArg(*clKernel, 0, sizeof(cl_mem), &x);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_int), &n);
    err|=clSetKernelArg(*clKernel, 2, sizeof(ACTIVATION), &a);
    err|=clSetKernelArg(*clKernel, 3, sizeof(cl_mem), &delta);
    err|=clSetKernelArg(*clKernel, 4, sizeof(cl_int), &offset_x);
    err|=clSetKernelArg(*clKernel, 5, sizeof(cl_int), &offset_delta);
    err|=clEnqueueNDRangeKernel(*clCommandQueue, *clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"gradient_array_cl");
}
void binary_gradient_array_cl(cl_mem x, cl_mem dx, int n, int size, BINARY_ACTIVATION a,cl_mem y,int off_x,int off_dx,int off_y) 
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    n/=2;
    setWorkItemSize(n,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram, "binary_gradient_array_opencl", &err);
    err|=clSetKernelArg(*clKernel, 0, sizeof(cl_mem), &x);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_mem), &dx);
    err|=clSetKernelArg(*clKernel, 2, sizeof(cl_int), &n);
    err|=clSetKernelArg(*clKernel, 3, sizeof(cl_int), &size);
    err|=clSetKernelArg(*clKernel, 4, sizeof(BINARY_ACTIVATION), &a);
    err|=clSetKernelArg(*clKernel, 5, sizeof(cl_mem), &y);
    err|=clSetKernelArg(*clKernel, 6, sizeof(cl_int), &off_x);
    err|=clSetKernelArg(*clKernel, 7, sizeof(cl_int), &off_dx);
    err|=clSetKernelArg(*clKernel, 8, sizeof(cl_int), &off_y);
    err|=clEnqueueNDRangeKernel(*clCommandQueue, *clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"binary_gradient_array_cl");
}
void binary_activate_array_cl(cl_mem x, int n, int size, BINARY_ACTIVATION a,cl_mem y,int off_x,int off_y) 
{
    cl_int err;
    size_t globalSize[3],localSize[3];
    n/=2;
    setWorkItemSize(n,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram, "binary_activate_array_opencl", &err);
    err|=clSetKernelArg(*clKernel, 0, sizeof(cl_mem), &x);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_int), &n);
    err|=clSetKernelArg(*clKernel, 2, sizeof(cl_int), &size);
    err|=clSetKernelArg(*clKernel, 3, sizeof(BINARY_ACTIVATION), &a);
    err|=clSetKernelArg(*clKernel, 4, sizeof(cl_mem), &y);
    err|=clSetKernelArg(*clKernel, 5, sizeof(cl_int), &off_x);
    err|=clSetKernelArg(*clKernel, 6, sizeof(cl_int), &off_y);
    err|=clEnqueueNDRangeKernel(*clCommandQueue, *clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"binary_activate_array_cl");
}
void forward_activation_layer_cl(layer l, network net)
{
    copy_cl(l.outputs*l.batch, net.input_cl, 1, l.output_cl, 1,0,0);
    activate_array_cl(l.output_cl, l.outputs*l.batch, l.activation,0);
}
void backward_activation_layer_cl(layer l, network net)
{
    gradient_array_cl(l.output_cl, l.outputs*l.batch, l.activation, l.delta_cl,0,0);
    copy_cl(l.outputs*l.batch, l.delta_cl, 1, net.delta_cl, 1,0,0);
}
#endif
