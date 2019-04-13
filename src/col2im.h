#ifndef COL2IM_H
#define COL2IM_H

#ifdef OPENCL
        #include "opencl_tool.h"
#endif

void col2im_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);

#ifdef GPU
void col2im_gpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im);
#endif
#ifdef OPENCL
void col2im_cl(cl_mem data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, cl_mem data_im,int offset_data,int offset_im);
#endif
#endif
