#ifndef IM2COL_H
#define IM2COL_H

#ifdef OPENCL
        #include "opencl_tool.h"
#endif

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

#ifdef GPU

void im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);

#endif
#ifdef OPENCL

void im2col_cl(cl_mem im,
         int channels, int height, int width,
         int ksize, int stride, int pad,cl_mem data_col,int offset_im,int offset_data);

#endif
#endif
