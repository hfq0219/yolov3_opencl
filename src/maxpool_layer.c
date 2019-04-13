#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(0, output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    #ifdef OPENCL
    l.forward_cl = forward_maxpool_layer_cl;
    l.backward_cl = backward_maxpool_layer_cl;
    l.indexes_cl = cl_make_int_array(0, output_size);
    l.output_cl  = cl_make_array(l.output, output_size);
    l.delta_cl   = cl_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size)/l->stride + 1;
    l->out_h = (h + l->pad - l->size)/l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
    #ifdef OPENCL
    cl_free(l->indexes_cl);
    cl_free(l->output_cl);
    cl_free(l->delta_cl);
    l->indexes_cl = cl_make_int_array(0, output_size);
    l->output_cl  = cl_make_array(l->output, output_size);
    l->delta_cl   = cl_make_array(l->delta,  output_size);
    #endif
}

void forward_maxpool_layer(const maxpool_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        net.delta[index] += l.delta[i];
    }
}
#ifdef OPENCL
void forward_maxpool_layer_cl(maxpool_layer layer, network net)
{
    int h = layer.out_h;
    int w = layer.out_w;
    int c = layer.c;

    size_t n = h*w*c*layer.batch;
    size_t globalWorkSize[3],localWorkSize[3];
    setWorkItemSize(n,globalWorkSize,localWorkSize);
    cl_int err;
    *clKernel=clCreateKernel(*clProgram, "forward_maxpool_layer_opencl", &err);
    err|=clSetKernelArg(*clKernel, 0, sizeof(cl_int), &n);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_int), &layer.h);
    err|=clSetKernelArg(*clKernel, 2, sizeof(cl_int), &layer.w);
    err|=clSetKernelArg(*clKernel, 3, sizeof(cl_int), &layer.c);
    err|=clSetKernelArg(*clKernel, 4, sizeof(cl_int), &layer.stride);
    err|=clSetKernelArg(*clKernel, 5, sizeof(cl_int), &layer.size);
    err|=clSetKernelArg(*clKernel, 6, sizeof(cl_int), &layer.pad);
    err|=clSetKernelArg(*clKernel, 7, sizeof(cl_mem), &net.input_cl);
    err|=clSetKernelArg(*clKernel, 8, sizeof(cl_mem), &layer.output_cl);
    err|=clSetKernelArg(*clKernel, 9, sizeof(cl_mem), &layer.indexes_cl);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel,3,NULL,globalWorkSize,localWorkSize,0,NULL,NULL);
    cl_error(err,"forward_maxpool_layer_cl");
}

void backward_maxpool_layer_cl(maxpool_layer layer, network net)
{
    size_t n = layer.h*layer.w*layer.c*layer.batch;
    size_t globalWorkSize[3],localWorkSize[3];
    setWorkItemSize(n,globalWorkSize,localWorkSize);
    cl_int err;
    *clKernel=clCreateKernel(*clProgram, "backward_maxpool_layer_opencl", &err);
    err|=clSetKernelArg(*clKernel, 0, sizeof(cl_int), &n);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_int), &layer.h);
    err|=clSetKernelArg(*clKernel, 2, sizeof(cl_int), &layer.w);
    err|=clSetKernelArg(*clKernel, 3, sizeof(cl_int), &layer.c);
    err|=clSetKernelArg(*clKernel, 4, sizeof(cl_int), &layer.stride);
    err|=clSetKernelArg(*clKernel, 5, sizeof(cl_int), &layer.size);
    err|=clSetKernelArg(*clKernel, 6, sizeof(cl_int), &layer.pad);
    err|=clSetKernelArg(*clKernel, 7, sizeof(cl_mem), &layer.delta_cl);
    err|=clSetKernelArg(*clKernel, 8, sizeof(cl_mem), &net.delta_cl);
    err|=clSetKernelArg(*clKernel, 9, sizeof(cl_mem), &layer.indexes_cl);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel,3,NULL,globalWorkSize,localWorkSize,0,NULL,NULL);
    cl_error(err,"backward_maxpool_layer_cl");
}
#endif
