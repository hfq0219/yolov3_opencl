#include "dropout_layer.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
    dropout_layer l = {0};
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);
    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
    #ifdef GPU
    l.forward_gpu = forward_dropout_layer_gpu;
    l.backward_gpu = backward_dropout_layer_gpu;
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
    #endif
    #ifdef OPENCL
    l.forward_cl = forward_dropout_layer_cl;
    l.backward_cl = backward_dropout_layer_cl;
    l.rand_cl = cl_make_array(l.rand, inputs*batch);
    #endif
    fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
} 

void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
    #ifdef GPU
    cuda_free(l->rand_gpu);
    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
    #endif
    #ifdef OPENCL
    cl_free(l->rand_cl);
    l->rand_cl = cl_make_array(l->rand, inputs*l->batch);
    #endif
}

void forward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if (!net.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        if(r < l.probability) net.input[i] = 0;
        else net.input[i] *= l.scale;
    }
}

void backward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if(!net.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l.rand[i];
        if(r < l.probability) net.delta[i] = 0;
        else net.delta[i] *= l.scale;
    }
}
#ifdef OPENCL
void forward_dropout_layer_cl(dropout_layer layer, network net)
{
    if (!net.train) return;
    int size = layer.inputs*layer.batch;
    cl_random(layer.rand_cl, size);
    /*
    int i;
    for(i = 0; i < size; ++i){
        layer.rand[i] = rand_uniform();
    }
    cuda_push_array(layer.rand_gpu, layer.rand, size);
    */
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(size,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram, "yoloswag420blazeit360noscope", &err);
    err|=clSetKernelArg(*clKernel, 0, sizeof(cl_mem), &net.input_cl);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_int), &size);
    err|=clSetKernelArg(*clKernel, 2, sizeof(cl_mem), &layer.rand_cl);
    err|=clSetKernelArg(*clKernel, 3, sizeof(cl_float), &layer.probability);
    err|=clSetKernelArg(*clKernel, 4, sizeof(cl_float), &layer.scale);
    err|=clEnqueueNDRangeKernel(*clCommandQueue, *clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"forward_dropout_layer_cl");
}

void backward_dropout_layer_cl(dropout_layer layer, network net)
{
    if(!net.delta_cl) return;
    int size = layer.inputs*layer.batch;

    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(size,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram, "yoloswag420blazeit360noscope", &err);
    err|=clSetKernelArg(*clKernel, 0, sizeof(cl_mem), &net.delta_cl);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_int), &size);
    err|=clSetKernelArg(*clKernel, 2, sizeof(cl_mem), &layer.rand_cl);
    err|=clSetKernelArg(*clKernel, 3, sizeof(cl_float), &layer.probability);
    err|=clSetKernelArg(*clKernel, 4, sizeof(cl_float), &layer.scale);
    err|=clEnqueueNDRangeKernel(*clCommandQueue, *clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"backward_dropout_layer_cl");
}
#endif