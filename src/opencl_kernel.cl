#define BLOCK 512
//----------------------activation_kernels.cu-------------------------------//
typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
}ACTIVATION;
typedef enum{
    MULT, ADD, SUB, DIV
}BINARY_ACTIVATION;

float lhtan_activate_kernel(float x)
{
    if(x < 0) return .001f*x;
    if(x > 1) return .001f*(x-1.f) + 1.f;
    return x;
}
float lhtan_gradient_kernel(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}
float hardtan_activate_kernel(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
float linear_activate_kernel(float x){return x;}
float logistic_activate_kernel(float x){return 1.f/(1.f + exp(-x));}
float loggy_activate_kernel(float x){return 2.f/(1.f + exp(-x)) - 1;}
float relu_activate_kernel(float x){return x*(x>0);}
float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
float selu_activate_kernel(float x){return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(exp(x)-1);}
float relie_activate_kernel(float x){return (x>0) ? x : .01f*x;}
float ramp_activate_kernel(float x){return x*(x>0)+.1f*x;}
float leaky_activate_kernel(float x){return (x>0) ? x : .1f*x;}
float tanh_activate_kernel(float x){return (2.f/(1 + exp(-2*x)) - 1);}
float plse_activate_kernel(float x)
{
    if(x < -4) return .01f * (x + 4);
    if(x > 4)  return .01f * (x - 4) + 1;
    return .125f*x + .5f;
}
float stair_activate_kernel(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2);
    else return (x - n) + floor(x/2);
}
float hardtan_gradient_kernel(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
float linear_gradient_kernel(float x){return 1;}
float logistic_gradient_kernel(float x){return (1-x)*x;}
float loggy_gradient_kernel(float x)
{
    float y = (x+1)/2;
    return 2*(1-y)*y;
}
float relu_gradient_kernel(float x){return (x>0);}
float elu_gradient_kernel(float x){return (x >= 0) + (x < 0)*(x + 1);}
float selu_gradient_kernel(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
float relie_gradient_kernel(float x){return (x>0) ? 1 : .01f;}
float ramp_gradient_kernel(float x){return (x>0)+.1f;}
float leaky_gradient_kernel(float x){return (x>0) ? 1 : .1f;}
float tanh_gradient_kernel(float x){return 1-x*x;}
float plse_gradient_kernel(float x){return (x < 0 || x > 1) ? .01f : .125f;}
float stair_gradient_kernel(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}
float activate_kernel(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate_kernel(x);
        case LOGISTIC:
            return logistic_activate_kernel(x);
        case LOGGY:
            return loggy_activate_kernel(x);
        case RELU:
            return relu_activate_kernel(x);
        case ELU:
            return elu_activate_kernel(x);
        case SELU:
            return selu_activate_kernel(x);
        case RELIE:
            return relie_activate_kernel(x);
        case RAMP:
            return ramp_activate_kernel(x);
        case LEAKY:
            return leaky_activate_kernel(x);
        case TANH:
            return tanh_activate_kernel(x);
        case PLSE:
            return plse_activate_kernel(x);
        case STAIR:
            return stair_activate_kernel(x);
        case HARDTAN:
            return hardtan_activate_kernel(x);
        case LHTAN:
            return lhtan_activate_kernel(x);
    }
    return 0;
}
float gradient_kernel(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient_kernel(x);
        case LOGISTIC:
            return logistic_gradient_kernel(x);
        case LOGGY:
            return loggy_gradient_kernel(x);
        case RELU:
            return relu_gradient_kernel(x);
        case ELU:
            return elu_gradient_kernel(x);
        case SELU:
            return selu_gradient_kernel(x);
        case RELIE:
            return relie_gradient_kernel(x);
        case RAMP:
            return ramp_gradient_kernel(x);
        case LEAKY:
            return leaky_gradient_kernel(x);
        case TANH:
            return tanh_gradient_kernel(x);
        case PLSE:
            return plse_gradient_kernel(x);
        case STAIR:
            return stair_gradient_kernel(x);
        case HARDTAN:
            return hardtan_gradient_kernel(x);
        case LHTAN:
            return lhtan_gradient_kernel(x);
    }
    return 0;
}
__kernel void binary_gradient_array_opencl(__global float *x,__global float *dy, int n, int s, BINARY_ACTIVATION a,__global float *dx,
        int offset_x,int offset_dy,int offset_dx)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int id = (x_+y_*z_)*m_+n_;
    x += offset_x;
    dy += offset_dy;
    dx += offset_dx;
    int i = id % s;
    int b = id / s;
    float x1 = x[b*s + i];
    float x2 = x[b*s + s/2 + i];
    if(id < n) {
        float de = dy[id];
        dx[b*s + i] = x2*de;
        dx[b*s + s/2 + i] = x1*de; 
    }
}
__kernel void binary_activate_array_opencl(__global float *x, int n, int s, BINARY_ACTIVATION a,__global float *y,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int id = (x_+y_*z_)*m_+n_;
    x += off1;
    y += off2;
    int i = id % s;
    int b = id / s;
    float x1 = x[b*s + i];
    float x2 = x[b*s + s/2 + i];
    if(id < n) y[id] = x1*x2;
}
__kernel void activate_array_opencl(__global float *x, int n, ACTIVATION a,int off_x)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    x += off_x;
    if(i < n) x[i] = activate_kernel(x[i], a);
}

__kernel void gradient_array_opencl(__global float *x, int n, ACTIVATION a,__global float *delta,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    x += off1;
    delta += off2;
    if(i < n) delta[i] *= gradient_kernel(x[i], a);
}
//----------------------------------avgpool_layer_kernels.cu--------------------------------------//
__kernel void forward_avgpool_layer_opencl(int n, int w, int h, int c,__global float *input,__global float *output)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int id = (x_+y_*z_)*m_+n_;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    output[out_index] = 0;
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        output[out_index] += input[in_index];
    }
    output[out_index] /= w*h;
}
__kernel void backward_avgpool_layer_opencl(int n, int w, int h, int c,__global float *in_delta,__global float *out_delta)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int id = (x_+y_*z_)*m_+n_;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        in_delta[in_index] += out_delta[out_index] / (w*h);
    }
}
//--------------------------------blas_kernels.cu--------------------------------------------//
//blas.c
__kernel void scale_bias_opencl(__global float *output,__global float *biases, int n, int size,int off1,int off2)
{
    int offset = get_group_id(0)*get_local_size(0)+get_local_id(0);
    int filter = get_group_id(1);
    int batch = get_group_id(2);
    output += off1;
    biases += off2;
    if(offset < size) output[(batch*n+filter)*size + offset] *= biases[filter];
}
__kernel void backward_scale_opencl(__global float *x_norm,__global float *delta, int batch, int n, int size,__global float *scale_updates,
        int off1,int off2,int off3)
{
    __local float part[BLOCK];
    int i,b;
    int filter = get_group_id(0);
    int p = get_local_id(0);
    x_norm += off1;
    delta += off2;
    scale_updates += off3;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index]*x_norm[index] : 0;
        }
    }
    part[p] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) scale_updates[filter] += part[i];
    }
}
__kernel void add_bias_opencl(__global float *output,__global float *biases, int batch, int n, int size,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = (x_+y_*z_)*m_+n_;
    output += off1;
    biases += off2;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}
__kernel void backward_bias_conn_opencl(__global float *bias_updates,__global float *delta, int batch, int n,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = (x_+y_*z_)*m_+n_;
    bias_updates += off1;
    delta += off2;
    if (index >= n) return;
    int b;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        int i = b*n + index;
        sum += delta[i];
    }
    bias_updates[index] += sum;
}
__kernel void backward_bias_opencl(__global float *bias_updates,__global float *delta, int batch, int n, int size,int off1,int off2)
{
    __local float part[BLOCK];
    int i,b;
    int filter = get_group_id(0);
    int p = get_local_id(0);
    bias_updates += off1;
    delta += off2;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
    }
}
__kernel void adam_opencl(int N,__global float *x,__global float *m,__global float *v, float B1, float B2, float rate, float eps, int t,
        int off1,int off2,int off3)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = (x_+y_*z_)*m_+n_;
    x += off1;
    m += off2;
    v += off3;
    if (index >= N) return;

    float mhat = m[index] / (1.f - pow(B1, t));
    float vhat = v[index] / (1.f - pow(B2, t));
    
    x[index] = x[index] + rate * mhat / (sqrt(vhat) + eps);
}
__kernel void normalize_opencl(int N,__global float *x,__global float *mean,__global float *variance,
        int batch, int filters, int spatial,int off1,int off2,int off3)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = (x_+y_*z_)*m_+n_;
    x += off1;
    mean += off2;
    variance += off3;
    if(index>=N) return;
    
    int f=(index/spatial)%filters;
    x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + 0.000001f);
}
__kernel void normalize_delta_opencl(int N,__global float *x,__global float *mean,__global float *variance,
        __global float *mean_delta,__global float *variance_delta, int batch, int filters, int spatial,__global float *delta,
        int off1,int off2,int off3,int off4,int off5,int off6)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = (x_+y_*z_)*m_+n_;
    x += off1;
    mean += off2;
    variance += off3;
    mean_delta += off4;
    variance_delta += off5;
    delta += off6;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    
    delta[index] = delta[index] * 1.f/(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2.f * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
}
__kernel void  variance_delta_opencl(__global float *x,__global float *delta,__global float *mean,
        __global float *variance, int batch, int filters, int spatial,__global float *variance_delta,
        int off1,int off2,int off3,int off4,int off5)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    x += off1;
    delta += off2;
    mean += off3;
    variance += off4;
    variance_delta += off5;
    if (i >= filters) return;
    int j,k;
    variance_delta[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance_delta[i] += delta[index]*(x[index] - mean[i]);
        }
    }
    variance_delta[i] *= -.5f * pow(variance[i] + .00001f, (float)(-3.f/2.f));
}
__kernel void accumulate_opencl(__global float *x, int n, int groups,__global float *sum,int off1,int off2)
{
    int k;
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    x += off1;
    sum += off2;
    if (i >= groups) return;
    sum[i] = 0;
    for(k = 0; k < n; ++k){
        sum[i] += x[k*groups + i];
    }
}
__kernel void fast_mean_delta_opencl(__global float *delta,__global float *variance, int batch, int filters, int spatial,__global float *mean_delta,
        int off1,int off2,int off3)
{
    __local float loca[BLOCK];

    int id = get_local_id(0);
    loca[id] = 0;

    int filter = get_group_id(0);
    delta += off1;
    variance += off2;
    mean_delta += off3;
    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += BLOCK){
            int index = j*spatial*filters + filter*spatial + i + id;
            loca[id] += (i+id < spatial) ? delta[index] : 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(id == 0){
        mean_delta[filter] = 0;
        for(i = 0; i < BLOCK; ++i){
            mean_delta[filter] += loca[i];
        }
        mean_delta[filter] *= (-1.f/sqrt(variance[filter] + .00001f));
    }
}
__kernel void fast_variance_delta_opencl(__global float *x,__global float *delta,__global float *mean,__global float *variance, int batch, int filters, int spatial,__global float *variance_delta,
        int off1,int off2,int off3,int off4,int off5)
{
    __local float loca[BLOCK];

    int id = get_local_id(0);
    loca[id] = 0;

    int filter = get_group_id(0);
    x += off1;
    delta += off2;
    mean += off3;
    variance += off4;
    variance_delta += off5;
    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += BLOCK){
            int index = j*spatial*filters + filter*spatial + i + id;

            loca[id] += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < BLOCK; ++i){
            variance_delta[filter] += loca[i];
        }
        variance_delta[filter] *= -.5f * pow(variance[filter] + .00001f, (float)(-3.f/2.f));
    }
}
__kernel void mean_delta_opencl(__global float *delta,__global float *variance, int batch, int filters,
        int spatial,__global float *mean_delta,int off1,int off2,int off3)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    delta += off1;
    variance += off2;
    mean_delta += off3;
    if (i >= filters) return;
    int j,k;
    mean_delta[i] = 0;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j*filters*spatial + i*spatial + k;
            mean_delta[i] += delta[index];
        }
    }
    mean_delta[i] *= (-1.f/sqrt(variance[i] + .00001f));
}
__kernel void  mean_opencl(__global float *x, int batch, int filters, int spatial,__global float *mean,int off1,int off2)
{
    float scale = 1.f/(batch * spatial);
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    x += off1;
    mean += off2;
    if (i >= filters) return;
    int j,k;
    mean[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            mean[i] += x[index];
        }
    }
    mean[i] *= scale;
}

__kernel void variance_opencl(__global float *x,__global float *mean, int batch, int filters, 
        int spatial,__global float *variance,int off1,int off2,int off3)
{
    float scale = 1.f/(batch * spatial - 1);
    int j,k;
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    x += off1;
    mean += off2;
    variance += off3;
    if (i >= filters) return;
    variance[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance[i] += pow((x[index] - mean[i]), 2);
        }
    }
    variance[i] *= scale;
}

__kernel void reorg_opencl(int N,__global float *x, int w, int h, int c, int batch, int stride, 
        int forward,__global float *out,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    x += off1;
    out += off2;
    if(i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i/w;
    int in_h = i%h;
    i = i/h;
    int in_c = i%c;
    i = i/c;
    int b = i%batch;

    int out_c = c/(stride*stride);

    int c2 = in_c % out_c;
    int offset = in_c / out_c;
    int w2 = in_w*stride + offset % stride;
    int h2 = in_h*stride + offset / stride;
    //printf("%d\n", offset);
    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

   // printf("%d %d %d\n", w2, h2, c2);
    //printf("%d %d\n", in_index, out_index);
    //if(out_index >= N || out_index < 0) printf("bad bad bad \n");

    if(forward) out[out_index] = x[in_index];
    else out[in_index] = x[out_index];
    //if(forward) out[1] = x[1];
    //else out[0] = x[0];
}
__kernel void axpy_opencl(int N,float ALPHA,__global float *X,int OFFX,int INCX,__global float *Y,int OFFY,int INCY,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = (x_+y_*z_)*m_+n_;
    X += off1;
    Y += off2;
    if(index<N){
        Y[OFFY+index*INCY] += ALPHA*X[OFFX+index*INCX];
    }
}
__kernel void pow_opencl(int N,float ALPHA,__global float *X,int INCX,__global float *Y,int INCY,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = (x_+y_*z_)*m_+n_;
    X += off1;
    Y += off2;
    if(index<N){
        Y[index*INCY] = pow(X[index*INCX],ALPHA);
    }
}
__kernel void const_opencl(int N, float ALPHA,__global float *X, int INCX,int off1)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    X += off1;
    if(i < N) X[i*INCX] = ALPHA;
}

__kernel void constrain_opencl(int N, float ALPHA,__global float *X, int INCX,int off1)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    X += off1;
    if(i < N) X[i*INCX] = fmin(ALPHA, fmax(-ALPHA, X[i*INCX]));
}

__kernel void supp_opencl(int N, float ALPHA,__global float *X, int INCX,int off1)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    X += off1;
    if(i < N) {
        if((X[i*INCX] * X[i*INCX]) < (ALPHA * ALPHA)) X[i*INCX] = 0;
    }
}

__kernel void add_opencl(int N, float ALPHA,__global float *X, int INCX,int off1)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    X += off1;
    if(i < N) X[i*INCX] += ALPHA;
}
__kernel void scal_opencl(int N,float ALPHA,__global float *X,int INCX,int off1)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = (x_+y_*z_)*m_+n_;
    X += off1;
    if(index<N) X[index*INCX] *= ALPHA;
}
__kernel void fill_opencl(int N,float ALPHA,__global float *X,int INCX,int off1)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    X += off1;
    if(i<N) X[i*INCX] = ALPHA;
}
__kernel void copy_opencl(int N,__global float *X,int OFFX,int INCX,__global float *Y,int OFFY,int INCY)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    if(i<N) Y[i*INCY+OFFY] = X[i*INCX+OFFX];
}
__kernel void mul_opencl(int N,__global float *X,int INCX,__global float *Y,int INCY,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    X += off1;
    Y += off2;
    if(i<N) Y[i*INCY] *= X[i*INCX];
}
__kernel void l2norm_opencl(int N,__global float *x,__global float *dx, int batch, int filters, int spatial,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = (x_+y_*z_)*m_+n_;
    x += off1;
    dx += off2;
    if (index >= N) return;
    int b = index / spatial;
    int i = index % spatial;
    int f;
    float sum = 0;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        sum += pow(x[index], 2);
    }
    sum = sqrt(sum);
    if(sum == 0) sum = 1;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        x[index] /= sum;
        dx[index] = (1 - x[index]) / sum;
    }
}
__kernel void fast_mean_opencl(__global float *x, int batch, int filters, int spatial,__global float *mean,int off1,int off2)
{
    __local float loca[BLOCK];

    int id = get_local_id(0);
    loca[id] = 0;

    int filter = get_group_id(0);
    x += off1;
    mean += off2;
    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += BLOCK){
            int index = j*spatial*filters + filter*spatial + i + id;
            loca[id] += (i+id < spatial) ? x[index] : 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < BLOCK; ++i){
            mean[filter] += loca[i];
        }
        mean[filter] /= spatial * batch;
    }
}
__kernel void fast_variance_opencl(__global float *x,__global float *mean, int batch, int filters, int spatial,__global float *variance,int off1,int off2,int off3)
{
    __local float loca[BLOCK];

    int id = get_local_id(0);
    loca[id] = 0;

    int filter = get_group_id(0);
    x += off1;
    mean += off2;
    variance += off3;
    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += BLOCK){
            int index = j*spatial*filters + filter*spatial + i + id;

            loca[id] += (i+id < spatial) ? pow((x[index] - mean[filter]), 2) : 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < BLOCK; ++i){
            variance[filter] += loca[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}
__kernel void flatten_opencl(int N,__global float *x, int spatial, int layers, int batch, int forward, 
        __global float *out,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    x += off1;
    out += off2;
    if(i >= N) return;
    int in_s = i%spatial;
    i = i/spatial;
    int in_c = i%layers;
    i = i/layers;
    int b = i;

    int i1 = b*layers*spatial + in_c*spatial + in_s;
    int i2 = b*layers*spatial + in_s*layers +  in_c;

    if (forward) out[i2] = x[i1];
    else out[i1] = x[i2];
}
__kernel void mask_opencl(int n,__global float *x, float mask_num,__global float *mask, float val,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    x += off1;
    mask += off2;
    if(i < n && mask[i] == mask_num) x[i] = val;
}
__kernel void scale_mask_opencl(int n,__global float *x, float mask_num,__global float *mask, float scale,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    x += off1;
    mask += off2;
    if(i < n && mask[i] == mask_num) x[i] *= scale;
}
__kernel void shortcut_opencl(int size, int minw, int minh, int minc, int stride, int sample, int batch,
        int w1, int h1, int c1,__global float *add, int w2, int h2, int c2, float s1, float s2,__global float *out,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int id = (x_+y_*z_)*m_+n_;
    add += off1;
    out += off2;
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    out[out_index] = s1*out[out_index] + s2*add[add_index];
    //out[out_index] += add[add_index];
}
__kernel void smooth_l1_opencl(int n,__global float *pred,__global float *truth,__global float *delta,__global float *error,int off1,int off2,int off3,int off4)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    pred += off1;
    truth += off2;
    delta += off3;
    error += off4;
    if(i < n){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff > 0) ? 1 : -1;
        }
    }
}
__kernel void softmax_x_ent_opencl(int n,__global float *pred,__global float *truth,__global float *delta,__global float *error,int off1,int off2,int off3,int off4)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    pred += off1;
    truth += off2;
    delta += off3;
    error += off4;
    if(i < n){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}
__kernel void logistic_x_ent_opencl(int n,__global float *pred,__global float *truth,__global float *delta,__global float *error,int off1,int off2,int off3,int off4)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    pred += off1;
    truth += off2;
    delta += off3;
    error += off4;
    if(i < n){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p+.0000001) - (1-t)*log(1-p+.0000001);
        delta[i] = t-p;
    }
}
__kernel void l2_opencl(int n,__global float *pred,__global float *truth,__global float *delta,__global float *error,int off1,int off2,int off3,int off4)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    pred += off1;
    truth += off2;
    delta += off3;
    error += off4;
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff; //I know this is technically wrong, deal with it.
        delta[i] = diff;
    }
}
__kernel void l1_opencl(int n,__global float *pred,__global float *truth,__global float *delta,__global float *error,int off1,int off2,int off3,int off4)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    pred += off1;
    truth += off2;
    delta += off3;
    error += off4;
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = (diff > 0) ? 1 : -1;
    }
}
__kernel void wgan_opencl(int n,__global float *pred,__global float *truth,__global float *delta,__global float *error,int off1,int off2,int off3,int off4)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    pred += off1;
    truth += off2;
    delta += off3;
    error += off4;
    if(i < n){
        error[i] = truth[i] ? -pred[i] : pred[i];
        delta[i] = (truth[i] > 0) ? 1 : -1;
    }
}
__kernel void weighted_sum_opencl(int n,__global float *a,__global float *b,__global float *s,__global float *c,int off1,int off2,int off3,int off4)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    a += off1;
    b += off2;
    s += off3;
    c += off4;
    if(i < n){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}
__kernel void deinter_opencl(int NX,__global float *X, int NY,__global float *Y, int B,__global float *OUT,int off1,int off2,int off3)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    X += off1;
    Y += off2;
    OUT += off3;
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            if(X) X[b*NX + j] += OUT[i];
        } else {
            if(Y) Y[b*NY + j - NX] += OUT[i];
        }
    }
}
__kernel void inter_opencl(int NX,__global float *X, int NY,__global float *Y, int B,__global float *OUT,int off1,int off2,int off3)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    X += off1;
    Y += off2;
    OUT += off3;
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            OUT[i] = X[b*NX + j];
        } else {
            OUT[i] = Y[b*NY + j - NX];
        }
    }
}
__kernel void weighted_delta_opencl(int n,__global float *a,__global float *b,__global float *s,
        __global float *da,__global float *db,__global float *ds,__global float *dc,
        int off1,int off2,int off3,int off4,int off5,int off6,int off7)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    a += off1;
    b += off2;
    s += off3;
    da += off4;
    db += off5;
    ds += off6;
    dc += off7;
    if(i < n){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}
__kernel void mult_add_into_opencl(int n,__global float *a,__global float *b,__global float *c,int off1,int off2,int off3)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    a += off1;
    b += off2;
    c += off3;
    if(i < n){
        c[i] += a[i]*b[i];
    }
}
void softmax_device(__global float *input, int n, float temp, int stride,__global float *output)
{
    int i;
    float sum = 0;
    float largest = -9999.0;
    for(i = 0; i < n; ++i){
        int val = input[i*stride];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}
__kernel void softmax_tree_opencl(__global float *input, int spatial, int batch, int stride, float temp,__global float *output, 
        int groups,__global int *group_size,__global int *group_offset,int off1,int off2,int off3,int off4)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int id = (x_+y_*z_)*m_+n_;
    input += off1;
    output += off2;
    group_size += off3;
    group_offset += off4;
    if (id >= spatial*batch*groups) return;
    int s = id % spatial;
    id = id / spatial;
    int g = id % groups;
    int b = id / groups;
    int goff = group_offset[g]*spatial;
    int boff = b*stride;
    softmax_device(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);
}
__kernel void softmax_opencl(__global float *input, int n, int batch, int batch_offset, int groups, 
        int group_offset, int stride, float temp,__global float *output,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int id = (x_+y_*z_)*m_+n_;
    input += off1;
    output += off2;
    if (id >= batch*groups) return;
    int b = id / groups;
    int g = id % groups;
    softmax_device(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
}
inline void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, 
                             prevVal.intVal, newVal.intVal) 
                             != prevVal.intVal);
}
__kernel void upsample_opencl(int N,__global float *x, int w, int h, int c, int batch, int stride, 
        int forward, float scale,__global float *out,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    x += off1;
    out += off2;
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;

    if(forward) out[out_index] += scale * x[in_index];
    else AtomicAdd(x+in_index, scale * out[out_index]);
}
/////////////--------------------col2im.c-----------------------------------//////////////////////
__kernel void col2im_opencl(int n,__global float *data_col,int height,int width,
        int ksize,int pad,int stride,int height_col,int width_col,__global float *data_im,int offset_data,int offset_im)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = x_*m_+n_;
    data_col += offset_data;
    data_im += offset_im;
    for(; index < n; index += m_*z_){
        float val = 0;
        int w = index % width + pad;
        int h = (index / width) % height + pad;
        int c = index / (width * height);
        // compute the start and end of the output
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);
        // equivalent implementation
        int offset =(c * ksize * ksize + h * ksize + w) * height_col * width_col;
        int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        int coeff_w_col = (1 - stride * height_col * width_col);
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col){
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col){
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[index] += val;
    }
}
/////////////-----------------------convolutional_kernels.cu-----------------------------------////////////
__kernel void binarize_opencl(__global float *x, int n,__global float *binary,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int i = (x_+y_*z_)*m_+n_;
    x += off1;
    binary += off2;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}
__kernel void binarize_input_opencl(__global float *input, int n, int size,__global float *binary,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int s = (x_+y_*z_)*m_+n_;
    input += off1;
    binary += off2;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += fabs(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}
__kernel void binarize_weights_opencl(__global float *weights, int n, int size,__global float *binary,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int f = (x_+y_*z_)*m_+n_;
    weights += off1;
    binary += off2;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += fabs(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}
__kernel void smooth_opencl(__global float *x, int n, int w, int h, int c, int size, float rate,__global float *delta,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int id = (x_+y_*z_)*m_+n_;
    x += off1;
    delta += off2;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.f);
    int h_offset = -(size/2.f);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

////////////////////////---------------------gemm.c-------------------------//////////////////////
__kernel void gemm_nn_opencl(int M,int N,int K,float ALPHA,__global float *weight,
        int lda,__global float *input,int ldb,__global float *output,int ldc,
        int offset_a,int offset_b,int offset_c)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = (x_+y_*z_)*m_+n_;
    weight += offset_a;
    input += offset_b;
    output += offset_c;
    if(index<M*ldc){
        int row = index / ldc;
        int col = index % ldc;
        for(int i = 0; i < K; i++){
            output[index] += ALPHA * weight[row*lda+i] * input[i*ldb+col];
        }
    }
}
__kernel void gemm_nt_opencl(int M,int N,int K,float ALPHA,__global float *weight,
        int lda,__global float *input,int ldb,__global float *output,int ldc,
        int offset_a,int offset_b,int offset_c)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = (x_+y_*z_)*m_+n_;
    weight += offset_a;
    input += offset_b;
    output += offset_c;
    if(index<M*ldc){
        int row = index / ldc;
        int col = index % ldc;
        float sum=0;
        for(int i = 0; i < K; i++){
            sum += ALPHA*weight[row*lda+i]*input[col*ldb+i];
        }
        output[index] += sum;
    }
}
__kernel void gemm_tn_opencl(int M,int N,int K,float ALPHA,__global float *weight,
        int lda,__global float *input,int ldb,__global float *output,int ldc,
        int offset_a,int offset_b,int offset_c)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = (x_+y_*z_)*m_+n_;
    weight += offset_a;
    input += offset_b;
    output += offset_c;
    if(index<M*ldc){
        int row = index / ldc;
        int col = index % ldc;
        for(int i = 0; i < K; i++){
            float A_PART = ALPHA*weight[i*lda+row];
            output[row*ldc+col] += A_PART*input[i*ldb+col];
        }
    }
}
__kernel void gemm_tt_opencl(int M, int N, int K, float ALPHA, 
        __global float *A, int lda, 
        __global float *B, int ldb,
        __global float *C, int ldc,
        int offset_a,int offset_b,int offset_c)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = (x_+y_*z_)*m_+n_;
    A += offset_a;
    B += offset_b;
    C += offset_c;
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            float sum = 0;
            for(int k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}
////////////////////-------------------im2col_kernels.cu--------------------------------////////////////////
__kernel void im2col_opencl(int n,__global float *data_im,int height,int width,
        int ksize,int pad,int stride,int height_col,int width_col,__global float *data_col,int offset_im,int offset_data)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int index = x_*m_+n_;
    data_im += offset_im;
    data_col += offset_data;
    for(; index < n; index += m_*z_){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        __global float *data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        __global float *data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}
//----------------------------------------crop_layer_kernels.cu------------------------------------------//
float get_pixel_kernel(__global float *image, int w, int h, int x, int y, int c)
{
    if(x < 0 || x >= w || y < 0 || y >= h) return 0;
    return image[x + w*(y + c*h)];
}

float3 make_float3(float a,float b,float c){
    float3 result;
    result.x=a;
    result.y=b;
    result.z=c;
    return result;
}

float3 rgb_to_hsv_kernel(float3 rgb)
{
    float r = rgb.x;
    float g = rgb.y; 
    float b = rgb.z;

    float h, s, v;
    float max = (r > g) ? ( (r > b) ? r : b) : ( (g > b) ? g : b);
    float min = (r < g) ? ( (r < b) ? r : b) : ( (g < b) ? g : b);
    float delta = max - min;
    v = max;
    if(max == 0){
        s = 0;
        h = -1;
    }else{
        s = delta/max;
        if(r == max){
            h = (g - b) / delta;
        } else if (g == max) {
            h = 2 + (b - r) / delta;
        } else {
            h = 4 + (r - g) / delta;
        }
        if (h < 0) h += 6;
    }
    return make_float3(h,s,v);
}

float3 hsv_to_rgb_kernel(float3 hsv)
{
    float h = hsv.x;
    float s = hsv.y; 
    float v = hsv.z;

    float r, g, b;
    float f, p, q, t;

    if (s == 0) {
        r = g = b = v;
    } else {
        int index = (int) floor(h);
        f = h - index;
        p = v*(1-s);
        q = v*(1-s*f);
        t = v*(1-s*(1-f));
        if(index == 0){
            r = v; g = t; b = p;
        } else if(index == 1){
            r = q; g = v; b = p;
        } else if(index == 2){
            r = p; g = v; b = t;
        } else if(index == 3){
            r = p; g = q; b = v;
        } else if(index == 4){
            r = t; g = p; b = v;
        } else {
            r = v; g = p; b = q;
        }
    }
    r = (r < 0) ? 0 : ((r > 1) ? 1 : r);
    g = (g < 0) ? 0 : ((g > 1) ? 1 : g);
    b = (b < 0) ? 0 : ((b > 1) ? 1 : b);
    return make_float3(r,g,b);
}

float bilinear_interpolate_kernel(__global float *image, int w, int h, float x, float y, int c)
{
    int ix = (int) floor(x);
    int iy = (int) floor(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * get_pixel_kernel(image, w, h, ix, iy, c) + 
        dy     * (1-dx) * get_pixel_kernel(image, w, h, ix, iy+1, c) + 
        (1-dy) *   dx   * get_pixel_kernel(image, w, h, ix+1, iy, c) +
        dy     *   dx   * get_pixel_kernel(image, w, h, ix+1, iy+1, c);
    return val;
}

__kernel void levels_image_opencl(__global float *image,__global float *rand, int batch, int w, int h, int train, float saturation, float exposure, 
        float translate, float scale, float shift,int off1,int off2)
{
    int size = batch * w * h;
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int id = (x_+y_*z_)*m_+n_;
    image += off1;
    rand += off2;
    if(id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;
    id /= h;
    float rshift = rand[0];
    float gshift = rand[1];
    float bshift = rand[2];
    float r0 = rand[8*id + 0];
    float r1 = rand[8*id + 1];
    float r2 = rand[8*id + 2];
    float r3 = rand[8*id + 3];

    saturation = r0*(saturation - 1) + 1;
    saturation = (r1 > .5f) ? 1.f/saturation : saturation;
    exposure = r2*(exposure - 1) + 1;
    exposure = (r3 > .5f) ? 1.f/exposure : exposure;

    size_t offset = id * h * w * 3;
    image += offset;
    float r = image[x + w*(y + h*0)];
    float g = image[x + w*(y + h*1)];
    float b = image[x + w*(y + h*2)];
    float3 rgb = make_float3(r,g,b);
    if(train){
        float3 hsv = rgb_to_hsv_kernel(rgb);
        hsv.y *= saturation;
        hsv.z *= exposure;
        rgb = hsv_to_rgb_kernel(hsv);
    } else {
        shift = 0;
    }
    image[x + w*(y + h*0)] = rgb.x*scale + translate + (rshift - .5f)*shift;
    image[x + w*(y + h*1)] = rgb.y*scale + translate + (gshift - .5f)*shift;
    image[x + w*(y + h*2)] = rgb.z*scale + translate + (bshift - .5f)*shift;
}

__kernel void forward_crop_layer_opencl(__global float *input,__global float *rand, int size, int c, int h, int w, int crop_height, int crop_width, int train, int flip, float angle,__global float *output)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int id = (x_+y_*z_)*m_+n_;
    if(id >= size) return;

    float cx = w/2.f;
    float cy = h/2.f;

    int count = id;
    int j = id % crop_width;
    id /= crop_width;
    int i = id % crop_height;
    id /= crop_height;
    int k = id % c;
    id /= c;
    int b = id;

    float r4 = rand[8*b + 4];
    float r5 = rand[8*b + 5];
    float r6 = rand[8*b + 6];
    float r7 = rand[8*b + 7];

    float dw = (w - crop_width)*r4;
    float dh = (h - crop_height)*r5;
    flip = (flip && (r6 > .5f));
    angle = 2*angle*r7 - angle;
    if(!train){
        dw = (w - crop_width)/2.f;
        dh = (h - crop_height)/2.f;
        flip = 0;
        angle = 0;
    }

    input += w*h*c*b;

    float x = (flip) ? w - dw - j - 1 : j + dw;    
    float y = i + dh;

    float rx = cos(angle)*(x-cx) - sin(angle)*(y-cy) + cx;
    float ry = sin(angle)*(x-cx) + cos(angle)*(y-cy) + cy;

    output[count] = bilinear_interpolate_kernel(input, w, h, rx, ry, k);
}
//-------------------------------------dropout_layer_kernels.cu-----------------------------------//
__kernel void yoloswag420blazeit360noscope(__global float *input, int size,__global float *rand, float prob, float scale,int off1,int off2)
{
    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int id = (x_+y_*z_)*m_+n_;
    input += off1;
    rand += off2;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}
//--------------------------------------maxpool_layer_kernels.cu----------------------------------//
__kernel void forward_maxpool_layer_opencl(int n, int in_h, int in_w, int in_c, int stride, int size, int pad,__global float *input,__global float *output,__global int *indexes)
{
    int h = (in_h + pad - size)/stride + 1;
    int w = (in_w + pad - size)/stride + 1;
    int c = in_c;

    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int id = (x_+y_*z_)*m_+n_;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad/2;
    int h_offset = -pad/2;

    int out_index = j + w*(i + h*(k + c*b));
    float max = -9999.0;
    int max_i = -1;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride + l;
            int cur_w = w_offset + j*stride + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -9999.0;
            max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }
    output[out_index] = max;
    indexes[out_index] = max_i;
}

__kernel void backward_maxpool_layer_opencl(int n, int in_h, int in_w, int in_c, int stride, int size, int pad,__global float *delta,__global float *prev_delta,__global int *indexes)
{
    int h = (in_h + pad - size)/stride + 1;
    int w = (in_w + pad - size)/stride + 1;
    int c = in_c;
    int area = (size-1)/stride;

    int x_=get_group_id(0),y_=get_group_id(1),z_=get_num_groups(0);
    int m_=get_local_size(0),n_=get_local_id(0);

    int id = (x_+y_*z_)*m_+n_;
    if(id >= n) return;

    int index = id;
    int j = id % in_w;
    id /= in_w;
    int i = id % in_h;
    id /= in_h;
    int k = id % in_c;
    id /= in_c;
    int b = id;

    int w_offset = -pad/2;
    int h_offset = -pad/2;

    float d = 0;
    int l, m;
    for(l = -area; l < area+1; ++l){
        for(m = -area; m < area+1; ++m){
            int out_w = (j-w_offset)/stride + m;
            int out_h = (i-h_offset)/stride + l;
            int out_index = out_w + w*(out_h + h*(k + c*b));
            int valid = (out_w >= 0 && out_w < w &&
                     out_h >= 0 && out_h < h);
            d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
        }
    }
    prev_delta[index] += d;
}