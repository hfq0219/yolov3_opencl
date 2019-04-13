#include <stdio.h>
#include <math.h>
#include "col2im.h"
void col2im_add_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}
//This one might be too, can't remember.
void col2im_cpu(float* data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, float* data_im) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad, val);
            }
        }
    }
}
#ifdef OPENCL
void col2im_cl(cl_mem data_col,int channels, int height, int width,int ksize,
        int stride, int pad, cl_mem data_im,int offset_data,int offset_im){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height * width;
    size_t globalWorkSize[3]={(num_kernels+BLOCK-1)/BLOCK*BLOCK,1,1};
    size_t localWorkSize[3]={BLOCK,1,1};

    cl_int err;
    *clKernel=clCreateKernel(*clProgram, "col2im_opencl", &err);
    err|=clSetKernelArg(*clKernel, 0, sizeof(cl_int), &num_kernels);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_mem), &data_col);
    err|=clSetKernelArg(*clKernel, 2, sizeof(cl_int), &height);
    err|=clSetKernelArg(*clKernel, 3, sizeof(cl_int), &width);
    err|=clSetKernelArg(*clKernel, 4, sizeof(cl_int), &ksize);
    err|=clSetKernelArg(*clKernel, 5, sizeof(cl_int), &pad);
    err|=clSetKernelArg(*clKernel, 6, sizeof(cl_int), &stride);
    err|=clSetKernelArg(*clKernel, 7, sizeof(cl_int), &height_col);
    err|=clSetKernelArg(*clKernel, 8, sizeof(cl_int), &width_col);
    err|=clSetKernelArg(*clKernel, 9, sizeof(cl_mem), &data_im);
    err|=clSetKernelArg(*clKernel, 10, sizeof(cl_int), &offset_data);
    err|=clSetKernelArg(*clKernel, 11, sizeof(cl_int), &offset_im);
    err|=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel,3,NULL,globalWorkSize,localWorkSize,0,NULL,NULL);
    cl_error(err,"col2im_cl");
}
#endif
