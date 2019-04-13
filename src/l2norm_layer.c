#include "l2norm_layer.h"
#include "activations.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

layer make_l2norm_layer(int batch, int inputs)
{
    fprintf(stderr, "l2norm                                         %4d\n",  inputs);
    layer l = {0};
    l.type = L2NORM;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = calloc(inputs*batch, sizeof(float));
    l.scales = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));

    l.forward = forward_l2norm_layer;
    l.backward = backward_l2norm_layer;
    #ifdef GPU
    l.forward_gpu = forward_l2norm_layer_gpu;
    l.backward_gpu = backward_l2norm_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.scales_gpu = cuda_make_array(l.output, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif
    #ifdef OPENCL
    l.forward_cl = forward_l2norm_layer_cl;
    l.backward_cl = backward_l2norm_layer_cl;

    l.output_cl = cl_make_array(l.output, inputs*batch); 
    l.scales_cl = cl_make_array(l.output, inputs*batch); 
    l.delta_cl = cl_make_array(l.delta, inputs*batch); 
    #endif
    return l;
}

void forward_l2norm_layer(const layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    l2normalize_cpu(l.output, l.scales, l.batch, l.out_c, l.out_w*l.out_h);
}

void backward_l2norm_layer(const layer l, network net)
{
    //axpy_cpu(l.inputs*l.batch, 1, l.scales, 1, l.delta, 1);
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_l2norm_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    l2normalize_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_w*l.out_h);
}

void backward_l2norm_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.scales_gpu, 1, l.delta_gpu, 1);
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}

#endif
#ifdef OPENCL

void forward_l2norm_layer_cl(const layer l, network net)
{
    copy_cl(l.outputs*l.batch, net.input_cl, 1, l.output_cl, 1,0,0);
    l2normalize_cl(l.output_cl, l.scales_cl, l.batch, l.out_c, l.out_w*l.out_h,0,0);
}

void backward_l2norm_layer_cl(const layer l, network net)
{
    axpy_cl(l.batch*l.inputs, 1, l.scales_cl, 1, l.delta_cl, 1,0,0);
    axpy_cl(l.batch*l.inputs, 1, l.delta_cl, 1, net.delta_cl, 1,0,0);
}

#endif
