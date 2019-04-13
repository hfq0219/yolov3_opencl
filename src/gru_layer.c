#include "gru_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void increment_layer(layer *l, int steps)
{
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;

#ifdef GPU
    l->output_gpu += num;
    l->delta_gpu += num;
    l->x_gpu += num;
    l->x_norm_gpu += num;
#endif
#ifdef OPENCL
    clEnqueueCopyBuffer(*clCommandQueue,l->output_cl,l->output_cl,sizeof(float)*num,0,sizeof(float)*num,0,NULL,NULL);
    clEnqueueCopyBuffer(*clCommandQueue,l->delta_cl,l->delta_cl,sizeof(float)*num,0,sizeof(float)*num,0,NULL,NULL);
    clEnqueueCopyBuffer(*clCommandQueue,l->x_cl,l->x_cl,sizeof(float)*num,0,sizeof(float)*num,0,NULL,NULL);
    clEnqueueCopyBuffer(*clCommandQueue,l->x_norm_cl,l->x_norm_cl,sizeof(float)*num,0,sizeof(float)*num,0,NULL,NULL);
    /*l->output_cl += num;
    l->delta_cl += num;
    l->x_cl += num;
    l->x_norm_cl += num;*/
#endif
}

layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam)
{
    fprintf(stderr, "GRU Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    layer l = {0};
    l.batch = batch;
    l.type = GRU;
    l.steps = steps;
    l.inputs = inputs;

    l.uz = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.uz) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.uz->batch = batch;

    l.wz = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wz) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wz->batch = batch;

    l.ur = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.ur) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.ur->batch = batch;

    l.wr = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wr) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wr->batch = batch;



    l.uh = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.uh) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.uh->batch = batch;

    l.wh = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wh) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wh->batch = batch;

    l.batch_normalize = batch_normalize;


    l.outputs = outputs;
    l.output = calloc(outputs*batch*steps, sizeof(float));
    l.delta = calloc(outputs*batch*steps, sizeof(float));
    l.state = calloc(outputs*batch, sizeof(float));
    l.prev_state = calloc(outputs*batch, sizeof(float));
    l.forgot_state = calloc(outputs*batch, sizeof(float));
    l.forgot_delta = calloc(outputs*batch, sizeof(float));

    l.r_cpu = calloc(outputs*batch, sizeof(float));
    l.z_cpu = calloc(outputs*batch, sizeof(float));
    l.h_cpu = calloc(outputs*batch, sizeof(float));

    l.forward = forward_gru_layer;
    l.backward = backward_gru_layer;
    l.update = update_gru_layer;

#ifdef GPU
    l.forward_gpu = forward_gru_layer_gpu;
    l.backward_gpu = backward_gru_layer_gpu;
    l.update_gpu = update_gru_layer_gpu;

    l.forgot_state_gpu = cuda_make_array(0, batch*outputs);
    l.forgot_delta_gpu = cuda_make_array(0, batch*outputs);
    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
    l.state_gpu = cuda_make_array(0, batch*outputs);
    l.output_gpu = cuda_make_array(0, batch*outputs*steps);
    l.delta_gpu = cuda_make_array(0, batch*outputs*steps);
    l.r_gpu = cuda_make_array(0, batch*outputs);
    l.z_gpu = cuda_make_array(0, batch*outputs);
    l.h_gpu = cuda_make_array(0, batch*outputs);

#ifdef CUDNN
    cudnnSetTensor4dDescriptor(l.uz->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uz->out_c, l.uz->out_h, l.uz->out_w); 
    cudnnSetTensor4dDescriptor(l.uh->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uh->out_c, l.uh->out_h, l.uh->out_w); 
    cudnnSetTensor4dDescriptor(l.ur->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ur->out_c, l.ur->out_h, l.ur->out_w); 
    cudnnSetTensor4dDescriptor(l.wz->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wz->out_c, l.wz->out_h, l.wz->out_w); 
    cudnnSetTensor4dDescriptor(l.wh->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wh->out_c, l.wh->out_h, l.wh->out_w); 
    cudnnSetTensor4dDescriptor(l.wr->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wr->out_c, l.wr->out_h, l.wr->out_w); 
#endif
#endif
#ifdef OPENCL
    l.forward_cl = forward_gru_layer_cl;
    l.backward_cl = backward_gru_layer_cl;
    l.update_cl = update_gru_layer_cl;

    l.forgot_state_cl = cl_make_array(0, batch*outputs);
    l.forgot_delta_cl = cl_make_array(0, batch*outputs);
    l.prev_state_cl = cl_make_array(0, batch*outputs);
    l.state_cl = cl_make_array(0, batch*outputs);
    l.output_cl = cl_make_array(0, batch*outputs*steps);
    l.delta_cl = cl_make_array(0, batch*outputs*steps);
    l.r_cl = cl_make_array(0, batch*outputs);
    l.z_cl = cl_make_array(0, batch*outputs);
    l.h_cl = cl_make_array(0, batch*outputs);
#endif
    return l;
}

void update_gru_layer(layer l, update_args a)
{
    update_connected_layer(*(l.ur), a);
    update_connected_layer(*(l.uz), a);
    update_connected_layer(*(l.uh), a);
    update_connected_layer(*(l.wr), a);
    update_connected_layer(*(l.wz), a);
    update_connected_layer(*(l.wh), a);
}

void forward_gru_layer(layer l, network net)
{
    network s = net;
    s.train = net.train;
    int i;
    layer uz = *(l.uz);
    layer ur = *(l.ur);
    layer uh = *(l.uh);

    layer wz = *(l.wz);
    layer wr = *(l.wr);
    layer wh = *(l.wh);

    fill_cpu(l.outputs * l.batch * l.steps, 0, uz.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ur.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, uh.delta, 1);

    fill_cpu(l.outputs * l.batch * l.steps, 0, wz.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wr.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wh.delta, 1);
    if(net.train) {
        fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
        copy_cpu(l.outputs*l.batch, l.state, 1, l.prev_state, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input = l.state;
        forward_connected_layer(wz, s);
        forward_connected_layer(wr, s);

        s.input = net.input;
        forward_connected_layer(uz, s);
        forward_connected_layer(ur, s);
        forward_connected_layer(uh, s);


        copy_cpu(l.outputs*l.batch, uz.output, 1, l.z_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, wz.output, 1, l.z_cpu, 1);

        copy_cpu(l.outputs*l.batch, ur.output, 1, l.r_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, wr.output, 1, l.r_cpu, 1);

        activate_array(l.z_cpu, l.outputs*l.batch, LOGISTIC);
        activate_array(l.r_cpu, l.outputs*l.batch, LOGISTIC);

        copy_cpu(l.outputs*l.batch, l.state, 1, l.forgot_state, 1);
        mul_cpu(l.outputs*l.batch, l.r_cpu, 1, l.forgot_state, 1);

        s.input = l.forgot_state;
        forward_connected_layer(wh, s);

        copy_cpu(l.outputs*l.batch, uh.output, 1, l.h_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, wh.output, 1, l.h_cpu, 1);

        if(l.tanh){
            activate_array(l.h_cpu, l.outputs*l.batch, TANH);
        } else {
            activate_array(l.h_cpu, l.outputs*l.batch, LOGISTIC);
        }

        weighted_sum_cpu(l.state, l.h_cpu, l.z_cpu, l.outputs*l.batch, l.output);

        copy_cpu(l.outputs*l.batch, l.output, 1, l.state, 1);

        net.input += l.inputs*l.batch;
        l.output += l.outputs*l.batch;
        increment_layer(&uz, 1);
        increment_layer(&ur, 1);
        increment_layer(&uh, 1);

        increment_layer(&wz, 1);
        increment_layer(&wr, 1);
        increment_layer(&wh, 1);
    }
}

void backward_gru_layer(layer l, network net)
{
}

#ifdef GPU

void pull_gru_layer(layer l)
{
}

void push_gru_layer(layer l)
{
}

void update_gru_layer_gpu(layer l, update_args a)
{
    update_connected_layer_gpu(*(l.ur), a);
    update_connected_layer_gpu(*(l.uz), a);
    update_connected_layer_gpu(*(l.uh), a);
    update_connected_layer_gpu(*(l.wr), a);
    update_connected_layer_gpu(*(l.wz), a);
    update_connected_layer_gpu(*(l.wh), a);
}

void forward_gru_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer uz = *(l.uz);
    layer ur = *(l.ur);
    layer uh = *(l.uh);

    layer wz = *(l.wz);
    layer wr = *(l.wr);
    layer wh = *(l.wh);

    fill_gpu(l.outputs * l.batch * l.steps, 0, uz.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ur.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, uh.delta_gpu, 1);

    fill_gpu(l.outputs * l.batch * l.steps, 0, wz.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wr.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wh.delta_gpu, 1);
    if(net.train) {
        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.prev_state_gpu, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input_gpu = l.state_gpu;
        forward_connected_layer_gpu(wz, s);
        forward_connected_layer_gpu(wr, s);

        s.input_gpu = net.input_gpu;
        forward_connected_layer_gpu(uz, s);
        forward_connected_layer_gpu(ur, s);
        forward_connected_layer_gpu(uh, s);

        copy_gpu(l.outputs*l.batch, uz.output_gpu, 1, l.z_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);

        copy_gpu(l.outputs*l.batch, ur.output_gpu, 1, l.r_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);

        activate_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);
        activate_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);

        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
        mul_gpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);

        s.input_gpu = l.forgot_state_gpu;
        forward_connected_layer_gpu(wh, s);

        copy_gpu(l.outputs*l.batch, uh.output_gpu, 1, l.h_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);

        if(l.tanh){
            activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);
        } else {
            activate_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC);
        }

        weighted_sum_gpu(l.state_gpu, l.h_gpu, l.z_gpu, l.outputs*l.batch, l.output_gpu);
        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.state_gpu, 1);

        net.input_gpu += l.inputs*l.batch;
        l.output_gpu += l.outputs*l.batch;
        increment_layer(&uz, 1);
        increment_layer(&ur, 1);
        increment_layer(&uh, 1);

        increment_layer(&wz, 1);
        increment_layer(&wr, 1);
        increment_layer(&wh, 1);
    }
}

void backward_gru_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer uz = *(l.uz);
    layer ur = *(l.ur);
    layer uh = *(l.uh);

    layer wz = *(l.wz);
    layer wr = *(l.wr);
    layer wh = *(l.wh);

    increment_layer(&uz, l.steps - 1);
    increment_layer(&ur, l.steps - 1);
    increment_layer(&uh, l.steps - 1);

    increment_layer(&wz, l.steps - 1);
    increment_layer(&wr, l.steps - 1);
    increment_layer(&wh, l.steps - 1);

    net.input_gpu += l.inputs*l.batch*(l.steps-1);
    if(net.delta_gpu) net.delta_gpu += l.inputs*l.batch*(l.steps-1);
    l.output_gpu += l.outputs*l.batch*(l.steps-1);
    l.delta_gpu += l.outputs*l.batch*(l.steps-1);
    float *end_state = l.output_gpu;
    for (i = l.steps-1; i >= 0; --i) {
        if(i != 0) copy_gpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
        else copy_gpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
        float *prev_delta_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;

        copy_gpu(l.outputs*l.batch, uz.output_gpu, 1, l.z_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);

        copy_gpu(l.outputs*l.batch, ur.output_gpu, 1, l.r_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);

        activate_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);
        activate_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);

        copy_gpu(l.outputs*l.batch, uh.output_gpu, 1, l.h_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);

        if(l.tanh){
            activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);
        } else {
            activate_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC);
        }

        weighted_delta_gpu(l.state_gpu, l.h_gpu, l.z_gpu, prev_delta_gpu, uh.delta_gpu, uz.delta_gpu, l.outputs*l.batch, l.delta_gpu);

        if(l.tanh){
            gradient_array_gpu(l.h_gpu, l.outputs*l.batch, TANH, uh.delta_gpu);
        } else {
            gradient_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC, uh.delta_gpu);
        }

        copy_gpu(l.outputs*l.batch, uh.delta_gpu, 1, wh.delta_gpu, 1);

        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
        mul_gpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);
        fill_gpu(l.outputs*l.batch, 0, l.forgot_delta_gpu, 1);

        s.input_gpu = l.forgot_state_gpu;
        s.delta_gpu = l.forgot_delta_gpu;

        backward_connected_layer_gpu(wh, s);
        if(prev_delta_gpu) mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.r_gpu, prev_delta_gpu);
        mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.state_gpu, ur.delta_gpu);

        gradient_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC, ur.delta_gpu);
        copy_gpu(l.outputs*l.batch, ur.delta_gpu, 1, wr.delta_gpu, 1);

        gradient_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC, uz.delta_gpu);
        copy_gpu(l.outputs*l.batch, uz.delta_gpu, 1, wz.delta_gpu, 1);

        s.input_gpu = l.state_gpu;
        s.delta_gpu = prev_delta_gpu;

        backward_connected_layer_gpu(wr, s);
        backward_connected_layer_gpu(wz, s);

        s.input_gpu = net.input_gpu;
        s.delta_gpu = net.delta_gpu;

        backward_connected_layer_gpu(uh, s);
        backward_connected_layer_gpu(ur, s);
        backward_connected_layer_gpu(uz, s);


        net.input_gpu -= l.inputs*l.batch;
        if(net.delta_gpu) net.delta_gpu -= l.inputs*l.batch;
        l.output_gpu -= l.outputs*l.batch;
        l.delta_gpu -= l.outputs*l.batch;
        increment_layer(&uz, -1);
        increment_layer(&ur, -1);
        increment_layer(&uh, -1);

        increment_layer(&wz, -1);
        increment_layer(&wr, -1);
        increment_layer(&wh, -1);
    }
    copy_gpu(l.outputs*l.batch, end_state, 1, l.state_gpu, 1);
}
#endif
#ifdef OPENCL

void pull_gru_layer(layer l)
{
}

void push_gru_layer(layer l)
{
}

void update_gru_layer_cl(layer l, update_args a)
{
    update_connected_layer_cl(*(l.ur), a);
    update_connected_layer_cl(*(l.uz), a);
    update_connected_layer_cl(*(l.uh), a);
    update_connected_layer_cl(*(l.wr), a);
    update_connected_layer_cl(*(l.wz), a);
    update_connected_layer_cl(*(l.wh), a);
}

void forward_gru_layer_cl(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer uz = *(l.uz);
    layer ur = *(l.ur);
    layer uh = *(l.uh);

    layer wz = *(l.wz);
    layer wr = *(l.wr);
    layer wh = *(l.wh);

    fill_cl(l.outputs * l.batch * l.steps, 0, uz.delta_cl, 1,0);
    fill_cl(l.outputs * l.batch * l.steps, 0, ur.delta_cl, 1,0);
    fill_cl(l.outputs * l.batch * l.steps, 0, uh.delta_cl, 1,0);

    fill_cl(l.outputs * l.batch * l.steps, 0, wz.delta_cl, 1,0);
    fill_cl(l.outputs * l.batch * l.steps, 0, wr.delta_cl, 1,0);
    fill_cl(l.outputs * l.batch * l.steps, 0, wh.delta_cl, 1,0);
    if(net.train) {
        fill_cl(l.outputs * l.batch * l.steps, 0, l.delta_cl, 1,0);
        copy_cl(l.outputs*l.batch, l.state_cl, 1, l.prev_state_cl, 1,0,0);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input_cl = l.state_cl;
        forward_connected_layer_cl(wz, s);
        forward_connected_layer_cl(wr, s);

        s.input_cl = net.input_cl;
        forward_connected_layer_cl(uz, s);
        forward_connected_layer_cl(ur, s);
        forward_connected_layer_cl(uh, s);

        copy_cl(l.outputs*l.batch, uz.output_cl, 1, l.z_cl, 1,0,0);
        axpy_cl(l.outputs*l.batch, 1, wz.output_cl, 1, l.z_cl, 1,0,0);

        copy_cl(l.outputs*l.batch, ur.output_cl, 1, l.r_cl, 1,0,0);
        axpy_cl(l.outputs*l.batch, 1, wr.output_cl, 1, l.r_cl, 1,0,0);

        activate_array_cl(l.z_cl, l.outputs*l.batch, LOGISTIC,0);
        activate_array_cl(l.r_cl, l.outputs*l.batch, LOGISTIC,0);

        copy_cl(l.outputs*l.batch, l.state_cl, 1, l.forgot_state_cl, 1,0,0);
        mul_cl(l.outputs*l.batch, l.r_cl, 1, l.forgot_state_cl, 1,0,0);

        s.input_cl = l.forgot_state_cl;
        forward_connected_layer_cl(wh, s);

        copy_cl(l.outputs*l.batch, uh.output_cl, 1, l.h_cl, 1,0,0);
        axpy_cl(l.outputs*l.batch, 1, wh.output_cl, 1, l.h_cl, 1,0,0);

        if(l.tanh){
            activate_array_cl(l.h_cl, l.outputs*l.batch, TANH,0);
        } else {
            activate_array_cl(l.h_cl, l.outputs*l.batch, LOGISTIC,0);
        }

        weighted_sum_cl(l.state_cl, l.h_cl, l.z_cl, l.outputs*l.batch, l.output_cl,0,0,0,0);
        copy_cl(l.outputs*l.batch, l.output_cl, 1, l.state_cl, 1,0,0);

        //net.input_cl += l.inputs*l.batch;
        //l.output_cl += l.outputs*l.batch;
        clEnqueueCopyBuffer(*clCommandQueue,net.input_cl,net.input_cl,sizeof(float)*i*l.inputs*l.batch,0,sizeof(float)*l.inputs*l.batch,0,NULL,NULL);
        clEnqueueCopyBuffer(*clCommandQueue,l.output_cl,l.output_cl,sizeof(float)*i*l.outputs*l.batch,0,sizeof(float)*l.outputs*l.batch,0,NULL,NULL);
        
        increment_layer(&uz, 1);
        increment_layer(&ur, 1);
        increment_layer(&uh, 1);

        increment_layer(&wz, 1);
        increment_layer(&wr, 1);
        increment_layer(&wh, 1);
    }
}

void backward_gru_layer_cl(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer uz = *(l.uz);
    layer ur = *(l.ur);
    layer uh = *(l.uh);

    layer wz = *(l.wz);
    layer wr = *(l.wr);
    layer wh = *(l.wh);

    increment_layer(&uz, l.steps - 1);
    increment_layer(&ur, l.steps - 1);
    increment_layer(&uh, l.steps - 1);

    increment_layer(&wz, l.steps - 1);
    increment_layer(&wr, l.steps - 1);
    increment_layer(&wh, l.steps - 1);

    clEnqueueCopyBuffer(*clCommandQueue,net.input_cl,net.input_cl,sizeof(float)*l.inputs*l.batch*(l.steps-1),0,sizeof(float)*l.inputs*l.batch,0,NULL,NULL);

    /*net.input_cl += l.inputs*l.batch*(l.steps-1)*/;
    if(net.delta_cl) 
        clEnqueueCopyBuffer(*clCommandQueue,net.delta_cl,net.delta_cl,sizeof(float)*l.inputs*l.batch*(l.steps-1),0,sizeof(float)*l.inputs*l.batch,0,NULL,NULL);
        /*net.delta_cl += l.inputs*l.batch*(l.steps-1)*/;
    clEnqueueCopyBuffer(*clCommandQueue,l.output_cl,l.output_cl,sizeof(float)*l.outputs*l.batch*(l.steps-1),0,sizeof(float)*l.outputs*l.batch,0,NULL,NULL);
    clEnqueueCopyBuffer(*clCommandQueue,l.delta_cl,l.delta_cl,sizeof(float)*l.outputs*l.batch*(l.steps-1),0,sizeof(float)*l.outputs*l.batch,0,NULL,NULL);
    /*l.output_cl += l.outputs*l.batch*(l.steps-1);
    l.delta_cl += l.outputs*l.batch*(l.steps-1);*/
    cl_mem end_state = l.output_cl;
    for (i = l.steps-1; i >= 0; --i) {
        
        if(i != 0) copy_cl(l.outputs*l.batch, l.output_cl, 1, l.state_cl, 1,-l.outputs*l.batch,0);
        else copy_cl(l.outputs*l.batch, l.prev_state_cl, 1, l.state_cl, 1,0,0);
        cl_mem prev_delta_cl = (i == 0) ? 0 : l.delta_cl /*- l.outputs*l.batch*/;

        copy_cl(l.outputs*l.batch, uz.output_cl, 1, l.z_cl, 1,0,0);
        axpy_cl(l.outputs*l.batch, 1, wz.output_cl, 1, l.z_cl, 1,0,0);

        copy_cl(l.outputs*l.batch, ur.output_cl, 1, l.r_cl, 1,0,0);
        axpy_cl(l.outputs*l.batch, 1, wr.output_cl, 1, l.r_cl, 1,0,0);

        activate_array_cl(l.z_cl, l.outputs*l.batch, LOGISTIC,0);
        activate_array_cl(l.r_cl, l.outputs*l.batch, LOGISTIC,0);

        copy_cl(l.outputs*l.batch, uh.output_cl, 1, l.h_cl, 1,0,0);
        axpy_cl(l.outputs*l.batch, 1, wh.output_cl, 1, l.h_cl, 1,0,0);

        if(l.tanh){
            activate_array_cl(l.h_cl, l.outputs*l.batch, TANH,0);
        } else {
            activate_array_cl(l.h_cl, l.outputs*l.batch, LOGISTIC,0);
        }

        weighted_delta_cl(l.state_cl, l.h_cl, l.z_cl, prev_delta_cl, uh.delta_cl, uz.delta_cl, l.outputs*l.batch, l.delta_cl,0,0,0,0,0,0,0);

        if(l.tanh){
            gradient_array_cl(l.h_cl, l.outputs*l.batch, TANH, uh.delta_cl,0,0);
        } else {
            gradient_array_cl(l.h_cl, l.outputs*l.batch, LOGISTIC, uh.delta_cl,0,0);
        }

        copy_cl(l.outputs*l.batch, uh.delta_cl, 1, wh.delta_cl, 1,0,0);

        copy_cl(l.outputs*l.batch, l.state_cl, 1, l.forgot_state_cl, 1,0,0);
        mul_cl(l.outputs*l.batch, l.r_cl, 1, l.forgot_state_cl, 1,0,0);
        fill_cl(l.outputs*l.batch, 0, l.forgot_delta_cl, 1,0);

        s.input_cl = l.forgot_state_cl;
        s.delta_cl = l.forgot_delta_cl;

        backward_connected_layer_cl(wh, s);
        if(prev_delta_cl) mult_add_into_cl(l.outputs*l.batch, l.forgot_delta_cl, l.r_cl, prev_delta_cl,0,0,0);
        mult_add_into_cl(l.outputs*l.batch, l.forgot_delta_cl, l.state_cl, ur.delta_cl,0,0,0);

        gradient_array_cl(l.r_cl, l.outputs*l.batch, LOGISTIC, ur.delta_cl,0,0);
        copy_cl(l.outputs*l.batch, ur.delta_cl, 1, wr.delta_cl, 1,0,0);

        gradient_array_cl(l.z_cl, l.outputs*l.batch, LOGISTIC, uz.delta_cl,0,0);
        copy_cl(l.outputs*l.batch, uz.delta_cl, 1, wz.delta_cl, 1,0,0);

        s.input_cl = l.state_cl;
        s.delta_cl = prev_delta_cl;

        backward_connected_layer_cl(wr, s);
        backward_connected_layer_cl(wz, s);

        s.input_cl = net.input_cl;
        s.delta_cl = net.delta_cl;

        backward_connected_layer_cl(uh, s);
        backward_connected_layer_cl(ur, s);
        backward_connected_layer_cl(uz, s);


        /*net.input_cl -= l.inputs*l.batch*/;
        if(net.delta_cl) /*net.delta_cl -= l.inputs*l.batch*/;
        /*l.output_cl -= l.outputs*l.batch;
        l.delta_cl -= l.outputs*l.batch;*/
        increment_layer(&uz, -1);
        increment_layer(&ur, -1);
        increment_layer(&uh, -1);

        increment_layer(&wz, -1);
        increment_layer(&wr, -1);
        increment_layer(&wh, -1);
    }
    copy_cl(l.outputs*l.batch, end_state, 1, l.state_cl, 1,0,0);
}
#endif