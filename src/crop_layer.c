#include "crop_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_crop_image(crop_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;
    return float_to_image(w,h,c,l.output);
}

void backward_crop_layer(const crop_layer l, network net){}
void backward_crop_layer_gpu(const crop_layer l, network net){}
void backward_crop_layer_cl(const crop_layer l, network net){}

crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure)
{
    fprintf(stderr, "Crop Layer: %d x %d -> %d x %d x %d image\n", h,w,crop_height,crop_width,c);
    crop_layer l = {0};
    l.type = CROP;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.scale = (float)crop_height / h;
    l.flip = flip;
    l.angle = angle;
    l.saturation = saturation;
    l.exposure = exposure;
    l.out_w = crop_width;
    l.out_h = crop_height;
    l.out_c = c;
    l.inputs = l.w * l.h * l.c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.output = calloc(l.outputs*batch, sizeof(float));
    l.forward = forward_crop_layer;
    l.backward = backward_crop_layer;

    #ifdef GPU
    l.forward_gpu = forward_crop_layer_gpu;
    l.backward_gpu = backward_crop_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    l.rand_gpu   = cuda_make_array(0, l.batch*8);
    #endif
    #ifdef OPENCL
    l.forward_cl = forward_crop_layer_cl;
    l.backward_cl = backward_crop_layer_cl;
    l.output_cl = cl_make_array(l.output, l.outputs*batch);
    l.rand_cl   = cl_make_array(0, l.batch*8);
    #endif
    return l;
}

void resize_crop_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->out_w =  l->scale*w;
    l->out_h =  l->scale*h;

    l->inputs = l->w * l->h * l->c;
    l->outputs = l->out_h * l->out_w * l->out_c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    #ifdef GPU
    cuda_free(l->output_gpu);
    l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);
    #endif
    #ifdef OPENCL
    cl_free(l->output_cl);
    l->output_cl = cl_make_array(l->output, l->outputs*l->batch);
    #endif
}


void forward_crop_layer(const crop_layer l, network net)
{
    int i,j,c,b,row,col;
    int index;
    int count = 0;
    int flip = (l.flip && rand()%2);
    int dh = rand()%(l.h - l.out_h + 1);
    int dw = rand()%(l.w - l.out_w + 1);
    float scale = 2;
    float trans = -1;
    if(l.noadjust){
        scale = 1;
        trans = 0;
    }
    if(!net.train){
        flip = 0;
        dh = (l.h - l.out_h)/2;
        dw = (l.w - l.out_w)/2;
    }
    for(b = 0; b < l.batch; ++b){
        for(c = 0; c < l.c; ++c){
            for(i = 0; i < l.out_h; ++i){
                for(j = 0; j < l.out_w; ++j){
                    if(flip){
                        col = l.w - dw - j - 1;    
                    }else{
                        col = j + dw;
                    }
                    row = i + dh;
                    index = col+l.w*(row+l.h*(c + l.c*b)); 
                    l.output[count++] = net.input[index]*scale + trans;
                }
            }
        }
    }
}
#ifdef OPENCL
void forward_crop_layer_cl(crop_layer layer, network net)
{
    cl_random(layer.rand_cl, layer.batch*8);

    float radians = layer.angle*3.14159265f/180.f;

    float scale = 2;
    float translate = -1;
    if(layer.noadjust){
        scale = 1;
        translate = 0;
    }

    int size = layer.batch * layer.w * layer.h;
    cl_int err;
    size_t globalSize[3],localSize[3];
    setWorkItemSize(size,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram, "levels_image_opencl",&err);
    err|=clSetKernelArg(*clKernel, 0, sizeof(cl_mem), &net.input_cl);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_mem), &layer.rand_cl);
    err|=clSetKernelArg(*clKernel, 2, sizeof(cl_int), &layer.batch);
    err|=clSetKernelArg(*clKernel, 3, sizeof(cl_int), &layer.w);
    err|=clSetKernelArg(*clKernel, 4, sizeof(cl_int), &layer.h);
    err|=clSetKernelArg(*clKernel, 5, sizeof(cl_int), &net.train);
    err|=clSetKernelArg(*clKernel, 6, sizeof(cl_float), &layer.saturation);
    err|=clSetKernelArg(*clKernel, 7, sizeof(cl_float), &layer.exposure);
    err|=clSetKernelArg(*clKernel, 8, sizeof(cl_float), &translate);
    err|=clSetKernelArg(*clKernel, 9, sizeof(cl_float), &scale);
    err|=clSetKernelArg(*clKernel, 10, sizeof(cl_float), &layer.shift);
    err|=clEnqueueNDRangeKernel(*clCommandQueue, *clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"forward_crop_layer_cl");

    size = layer.batch*layer.c*layer.out_w*layer.out_h;

    setWorkItemSize(size,globalSize,localSize);
    *clKernel=clCreateKernel(*clProgram, "forward_crop_layer_opencl", &err);
    err|=clSetKernelArg(*clKernel, 0, sizeof(cl_mem), &net.input_cl);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_mem), &layer.rand_cl);
    err|=clSetKernelArg(*clKernel, 2, sizeof(cl_int), &size);
    err|=clSetKernelArg(*clKernel, 3, sizeof(cl_int), &layer.c);
    err|=clSetKernelArg(*clKernel, 4, sizeof(cl_int), &layer.h);
    err|=clSetKernelArg(*clKernel, 5, sizeof(cl_int), &layer.w);
    err|=clSetKernelArg(*clKernel, 6, sizeof(cl_int), &layer.out_h);
    err|=clSetKernelArg(*clKernel, 7, sizeof(cl_int), &layer.out_w);
    err|=clSetKernelArg(*clKernel, 8, sizeof(cl_int), &net.train);
    err|=clSetKernelArg(*clKernel, 9, sizeof(cl_int), &layer.flip);
    err|=clSetKernelArg(*clKernel, 10, sizeof(cl_float), &radians);
    err|=clSetKernelArg(*clKernel, 11, sizeof(cl_mem), &layer.output_cl);
    err|=clEnqueueNDRangeKernel(*clCommandQueue, *clKernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    cl_error(err,"forward_crop_layer_cl");
}
#endif
