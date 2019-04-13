#include "opencl_tool.h"
#include <math.h>
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>
extern cl_kernel *clKernel;
/**创建平台、设备、上下文、命令队列、程序对象,对大部分 OpenCL 程序相同。
 */
int CreateTool(cl_platform_id *platform,cl_device_id *device,cl_context *context,
                cl_command_queue *commandQueue,cl_program *program,const char *fileName){
    cl_int err;
    cl_uint num;
    //获得第一个可用平台
    err=clGetPlatformIDs(1, platform, &num);
    if(err!=CL_SUCCESS||num<=0||platform==NULL){
        fprintf(stderr,"no platform.");
        return -1;
    }
    //获得第一个可用设备
    err=clGetDeviceIDs(*platform, CL_DEVICE_TYPE_GPU, 1, device, &num);
    if(err!=CL_SUCCESS||num<=0||device==NULL){
        fprintf(stderr,"no device.");
        return -1;
    }
    //获得一个上下文
    cl_context_properties properties[]={
        CL_CONTEXT_PLATFORM,(cl_context_properties)*platform,0
    };
    *context=clCreateContextFromType(properties,CL_DEVICE_TYPE_GPU,NULL,NULL,&err);
    if(err!=CL_SUCCESS||context==NULL){
        fprintf(stderr,"no context.");
        return -1;
    }
    //通过上下文对指定设备构建命令队列
    *commandQueue=clCreateCommandQueue(*context, *device, 0, &err);
    if(err!=CL_SUCCESS||commandQueue==NULL){
        fprintf(stderr,"no commandQueue.");
        return -1;
    }
    //读取内核文件并转换为字符串
    FILE *kernelFile;
    kernelFile=fopen(fileName,"r");
    if(kernelFile==NULL){
        fprintf(stderr,"kernel file open failed.");
        return -1;
    }
    fseek(kernelFile, 0, SEEK_END);
    int fileLen = ftell(kernelFile);
    char *srcStr = (char *) malloc(sizeof(char) * fileLen);
    fseek(kernelFile, 0, SEEK_SET);
    fread(srcStr, fileLen, sizeof(char), kernelFile);
    fclose(kernelFile);
    srcStr[fileLen]='\0';
    //在上下文环境下编译指定内核文件的程序对象
    *program=clCreateProgramWithSource(*context, 1, (const char **)&srcStr, NULL, &err);
    if(err!=CL_SUCCESS||program==NULL){
        fprintf(stderr,"no program.");
        return -1;
    }
    err=clBuildProgram(*program, 0, NULL, NULL,NULL,NULL);
    if(err!=CL_SUCCESS){
        fprintf(stderr,"can not build program.\n");
        char buildLog[16384];
        clGetProgramBuildInfo(*program,*device,CL_PROGRAM_BUILD_LOG,sizeof(buildLog),
            buildLog,NULL);
        fprintf(stderr,buildLog);
        return -1;
    }
    return 0;
}
//资源释放
void clean(cl_context *context,cl_command_queue *commandQueue,cl_program *program,cl_kernel *kernel)
{
    if(*commandQueue!=0)
        clReleaseCommandQueue(*commandQueue);
    if(*kernel!=0)
        clReleaseKernel(*kernel);
    if(*program!=0)
        clReleaseProgram(*program);
    if(*context!=0)
        clReleaseContext(*context);
}

void setWorkItemSize(size_t n,size_t global_work_size[3],size_t local_work_size[3]){

    local_work_size[0] = BLOCK;
    local_work_size[1] = 1;
    local_work_size[2] = 1;

    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    global_work_size[0] = x*BLOCK;
    global_work_size[1] = y;
    global_work_size[2] = 1;
}

void cl_error(cl_int err,char * funName){
    if(err!=CL_SUCCESS){
        fprintf(stderr,funName);
        fprintf(stderr,": opencl error: %d, ",err);
        switch(err){
            case -1:fprintf(stderr,"CL_DEVICE_NOT_FOUND\n");break;
            case -2:fprintf(stderr,"CL_DEVICE_NOT_AVAILABLE\n");break;
            case -3:fprintf(stderr,"CL_COMPILER_NOT_AVAILABLE\n");break;
            case -4:fprintf(stderr,"CL_MEM_OBJECT_ALLOCATION_FAILURE\n");break;
            case -5:fprintf(stderr,"CL_OUT_OF_RESOURCES\n");break;
            case -6:fprintf(stderr,"CL_OUT_OF_HOST_MEMORY\n");break;
            case -7:fprintf(stderr,"CL_PROFILING_INFO_NOT_AVAILABLE\n");break;
            case -8:fprintf(stderr,"CL_MEM_COPY_OVERLAP\n");break;
            case -9:fprintf(stderr,"CL_IMAGE_FORMAT_MISMATCH\n");break;
            case -10:fprintf(stderr,"CL_IMAGE_FORMAT_NOT_SUPPORTED\n");break;
            case -11:fprintf(stderr,"CL_BUILD_PROGRAM_FAILURE\n");break;
            case -12:fprintf(stderr,"CL_MAP_FAILURE\n");break;
            case -13:fprintf(stderr,"CL_MISALIGNED_SUB_BUFFER_OFFSET\n");break;
            case -14:fprintf(stderr,"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n");break;
            case -30:fprintf(stderr,"CL_INVALID_VALUE\n");break;
            case -31:fprintf(stderr,"CL_INVALID_DEVICE_TYPE\n");break;
            case -32:fprintf(stderr,"CL_INVALID_PLATFORM\n");break;
            case -33:fprintf(stderr,"CL_INVALID_DEVICE\n");break;
            case -34:fprintf(stderr,"CL_INVALID_CONTEXT\n");break;
            case -35:fprintf(stderr,"CL_INVALID_QUEUE_PROPERTIES\n");break;
            case -36:fprintf(stderr,"CL_INVALID_COMMAND_QUEUE\n");break;
            case -37:fprintf(stderr,"CL_INVALID_HOST_PTR\n");break;
            case -38:fprintf(stderr,"CL_INVALID_MEM_OBJECT\n");break;
            case -39:fprintf(stderr,"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR\n");break;
            case -40:fprintf(stderr,"CL_INVALID_IMAGE_SIZE\n");break;
            case -41:fprintf(stderr,"CL_INVALID_SAMPLER\n");break;
            case -42:fprintf(stderr,"CL_INVALID_BINARY\n");break;
            case -43:fprintf(stderr,"CL_INVALID_BUILD_OPTIONS\n");break;
            case -44:fprintf(stderr,"CL_INVALID_PROGRAM\n");break;
            case -45:fprintf(stderr,"CL_INVALID_PROGRAM_EXECUTABLE\n");break;
            case -46:fprintf(stderr,"CL_INVALID_KERNEL_NAME\n");break;
            case -47:fprintf(stderr,"CL_INVALID_KERNEL_DEFINITION\n");break;
            case -48:fprintf(stderr,"CL_INVALID_KERNEL\n");break;
            case -49:fprintf(stderr,"CL_INVALID_ARG_INDEX\n");break;
            case -50:fprintf(stderr,"CL_INVALID_ARG_VALUE\n");break;
            case -51:fprintf(stderr,"CL_INVALID_ARG_SIZE\n");break;
            case -52:fprintf(stderr,"CL_INVALID_KERNEL_ARGS\n");break;
            case -53:fprintf(stderr,"CL_INVALID_WORK_DIMENSION\n");break;
            case -54:fprintf(stderr,"CL_INVALID_WORK_GROUP_SIZE\n");break;
            case -55:fprintf(stderr,"CL_INVALID_WORK_ITEM_SIZE\n");break;
            case -56:fprintf(stderr,"CL_INVALID_GLOBAL_OFFSET\n");break;
            case -57:fprintf(stderr,"CL_INVALID_EVENT_WAIT_LIST\n");break;
            case -58:fprintf(stderr,"CL_INVALID_EVENT\n");break;
            case -59:fprintf(stderr,"CL_INVALID_OPERATION\n");break;
            case -60:fprintf(stderr,"CL_INVALID_GL_OBJECT\n");break;
            case -61:fprintf(stderr,"CL_INVALID_BUFFER_SIZE\n");break;
            case -62:fprintf(stderr,"CL_INVALID_MIP_LEVEL\n");break;
            case -63:fprintf(stderr,"CL_INVALID_GLOBAL_WORK_SIZE\n");break;
            case -64:fprintf(stderr,"CL_INVALID_PROPERTY\n");break;
            default:fprintf(stderr,"unknown error.\n");break;
        }
        exit(-1);
    }
}

cl_mem cl_make_array(float *x, size_t n)
{
    cl_mem x_cl;
    if(x){
        x_cl=clCreateBuffer(*clContext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(float)*n,x,NULL);
    } else {
        x_cl=clCreateBuffer(*clContext,CL_MEM_READ_WRITE,sizeof(float)*n,NULL,NULL);
        fill_cl(n, 0, x_cl, 1,0);
    }
    if(!x_cl) error("Cl malloc failed\n");
    return x_cl;
}

void cl_random(cl_mem x_cl, size_t n)
{
}

float cl_compare(cl_mem x_cl,float *x, size_t n, char *s)
{
    float *tmp = calloc(n, sizeof(float));
    cl_pull_array(x_cl, tmp, n,0);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err/n));
    free(tmp);
    return err;
}

cl_mem cl_make_int_array(int *x, size_t n)
{
    cl_mem x_cl;
    if(x){
        x_cl=clCreateBuffer(*clContext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(int)*n,x,NULL);
    } else {
        x_cl=clCreateBuffer(*clContext,CL_MEM_READ_WRITE,sizeof(int)*n,NULL,NULL);
    }
    if(!x_cl) error("Cl malloc failed\n");
    return x_cl;
}

void cl_free(cl_mem x_cl)
{
    clReleaseMemObject(x_cl);
}

void cl_push_array(cl_mem x_cl, float *x, size_t n,int off1)
{
    size_t size = sizeof(float)*n;
    clEnqueueWriteBuffer(*clCommandQueue,x_cl,CL_TRUE,off1,size,x,0,NULL,NULL);
}

void cl_pull_array(cl_mem x_cl, float *x, size_t n,int off1)
{
    size_t size = sizeof(float)*n;
    clEnqueueReadBuffer(*clCommandQueue,x_cl,CL_TRUE,off1,size,x,0,NULL,NULL);
}

float cl_mag_array(cl_mem x_cl, size_t n,int off1)
{
    float *temp = calloc(n, sizeof(float));
    cl_pull_array(x_cl, temp, n,off1);
    float m = mag_array(temp, n);
    free(temp);
    return m;
}

