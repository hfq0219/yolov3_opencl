#ifndef OPENCL_TOOL_H
#define OPENCL_TOOL_H

#include "CL/cl.h"
#include <stdio.h>
#include <stdlib.h>

extern cl_platform_id *clPlatform;
extern cl_device_id *clDevice;
extern cl_context *clContext;
extern cl_command_queue *clCommandQueue;
extern cl_program *clProgram;
extern cl_kernel *clKernel;
#define BLOCK 512

int CreateTool(cl_platform_id *platform,cl_device_id *device,cl_context *context,
                cl_command_queue *commandQueue,cl_program *program,const char *fileName);

void clean(cl_context *context,cl_command_queue *commandQueue,cl_program *program,cl_kernel *kernel);

void setWorkItemSize(size_t num,size_t global_work_size[3],size_t local_work_size[3]);
void cl_error(cl_int err,char * funName);
cl_mem cl_make_array(float *x, size_t n);
void cl_random(cl_mem x_cl, size_t n);
float cl_compare(cl_mem x_cl,float *x, size_t n, char *s);
cl_mem cl_make_int_array(int *x, size_t n);
void cl_free(cl_mem x_cl);
void cl_push_array(cl_mem x_cl, float *x, size_t n,int off1);
void cl_pull_array(cl_mem x_cl, float *x, size_t n,int off1);
float cl_mag_array(cl_mem x_cl, size_t n,int off1);

#endif