#ifndef BLAS_H
#define BLAS_H
#include "darknet.h"

void flatten(float *x, int size, int layers, int batch, int forward);
void pm(int M, int N, float *A);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void test_blas();

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void mult_add_into_cpu(int N, float *X, float *Y, float *Z);

void const_cpu(int N, float ALPHA, float *X, int INCX);
void constrain_gpu(int N, float ALPHA, float * X, int INCX);
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);

int test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);

void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);
void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial);

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_sum_cpu(float *a, float *b, float *s, int num, float *c);
void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc);

void softmax(float *input, int n, float temp, int stride, float *output);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

#ifdef GPU
#include "cuda.h"
#include "tree.h"

void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
void add_gpu(int N, float ALPHA, float * X, int INCX);
void supp_gpu(int N, float ALPHA, float * X, int INCX);
void mask_gpu(int N, float * X, float mask_num, float * mask, float val);
void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale);
void const_gpu(int N, float ALPHA, float *X, int INCX);
void pow_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_gpu(int N, float *X, int INCX, float *Y, int INCY);

void mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void l2normalize_gpu(float *x, float *dx, int batch, int filters, int spatial);

void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);

void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);

void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);

void logistic_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);
void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);
void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_gpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
void wgan_gpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc);
void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c);
void mult_add_into_gpu(int num, float *a, float *b, float *c);
void inter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void deinter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);

void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t);

void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out);
void softmax_tree(float *input, int spatial, int batch, int stride, float temp, float *output, tree hier);
void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

#endif
#ifdef OPENCL
#include "tree.h"
void constrain_cl(int N, float ALPHA, cl_mem X, int INCX,int off1);
void axpy_cl(int N, float ALPHA, cl_mem X, int INCX, cl_mem Y, int INCY,int OFFX,int OFFY);
void axpy_cl_offset(int N, float ALPHA,cl_mem X, int OFFX, int INCX,cl_mem Y, int OFFY, int INCY);
void copy_cl(int N, cl_mem X, int INCX, cl_mem Y, int INCY,int OFFX,int OFFY);
void copy_cl_offset(int N,cl_mem X, int OFFX, int INCX,cl_mem Y, int OFFY, int INCY);
void add_cl(int N, float ALPHA, cl_mem X, int INCX,int off1);
void supp_cl(int N, float ALPHA, cl_mem X, int INCX,int off1);
void mask_cl(int N, cl_mem X, float mask_num, cl_mem mask, float val,int off1,int off2);
void scale_mask_cl(int N, cl_mem X, float mask_num, cl_mem mask, float scale,int off1,int off2);
void const_cl(int N, float ALPHA, cl_mem X, int INCX,int off1);
void pow_cl(int N, float ALPHA, cl_mem X, int INCX, cl_mem Y, int INCY,int off1,int off2);
void mul_cl(int N, cl_mem X, int INCX, cl_mem Y, int INCY,int off1,int off2);

void mean_cl(cl_mem x, int batch, int filters, int spatial, cl_mem mean,int off1,int off2);
void variance_cl(cl_mem x, cl_mem mean, int batch, int filters, int spatial, cl_mem variance,int off1,int off2,int off3);
void normalize_cl(cl_mem x, cl_mem mean, cl_mem variance, int batch, int filters, int spatial,int off1,int off2,int off3);
void l2normalize_cl(cl_mem x, cl_mem dx, int batch, int filters, int spatial,int off1,int off2);

void normalize_delta_cl(cl_mem x, cl_mem mean, cl_mem variance, cl_mem mean_delta, cl_mem variance_delta, int batch, int filters, int spatial, cl_mem delta,int off1,int off2,int off3,int off4,int off5,int off6);

void fast_mean_delta_cl(cl_mem delta, cl_mem variance, int batch, int filters, int spatial, cl_mem mean_delta,int off1,int off2,int off3);
void fast_variance_delta_cl(cl_mem x, cl_mem delta, cl_mem mean, cl_mem variance, int batch, int filters, int spatial, cl_mem variance_delta,int off1,int off2,int off3,int off4,int off5);

void fast_variance_cl(cl_mem x, cl_mem mean, int batch, int filters, int spatial, cl_mem variance,int off1,int off2,int off3);
void fast_mean_cl(cl_mem x, int batch, int filters, int spatial, cl_mem mean,int off1,int off2);
void shortcut_cl(int batch, int w1, int h1, int c1, cl_mem add, int w2, int h2, int c2, float s1, float s2, cl_mem out,int off1,int off2);
void scale_bias_cl(cl_mem output, cl_mem biases, int batch, int n, int size,int off1,int off2);
void backward_scale_cl(cl_mem x_norm, cl_mem delta, int batch, int n, int size, cl_mem scale_updates,int off1,int off2,int off3);
void scale_bias_cl(cl_mem output, cl_mem biases, int batch, int n, int size,int off1,int off2);
void add_bias_cl(cl_mem output, cl_mem biases, int batch, int n, int size,int off1,int off2);
void backward_bias_cl(cl_mem bias_updates, cl_mem delta, int batch, int n, int size,int off1,int off2);

void logistic_x_ent_cl(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error,int off1,int off2,int off3,int off4);
void softmax_x_ent_cl(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error,int off1,int off2,int off3,int off4);
void smooth_l1_cl(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error,int off1,int off2,int off3,int off4);
void l2_cl(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error,int off1,int off2,int off3,int off4);
void l1_cl(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error,int off1,int off2,int off3,int off4);
void wgan_cl(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error,int off1,int off2,int off3,int off4);
void weighted_delta_cl(cl_mem a, cl_mem b, cl_mem s, cl_mem da, cl_mem db, cl_mem ds, int num, cl_mem dc,int off1,int off2,int off3,int off4,int off5,int off6,int off7);
void weighted_sum_cl(cl_mem a, cl_mem b, cl_mem s, int num, cl_mem c,int off1,int off2,int off3,int off4);
void mult_add_into_cl(int num, cl_mem a, cl_mem b, cl_mem c,int off1,int off2,int off3);
void inter_cl(int NX, cl_mem X, int NY, cl_mem Y, int B, cl_mem OUT,int off1,int off2,int off3);
void deinter_cl(int NX, cl_mem X, int NY, cl_mem Y, int B, cl_mem OUT,int off1,int off2,int off3);

void reorg_cl(cl_mem x, int w, int h, int c, int batch, int stride, int forward, cl_mem out,int off1,int off2);

void softmax_cl(cl_mem input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem output,int off1,int off2);
void adam_update_cl(cl_mem w, cl_mem d, cl_mem m, cl_mem v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t,int off1,int off2,int off3,int off4);
void adam_cl(int n, cl_mem x, cl_mem m, cl_mem v, float B1, float B2, float rate, float eps, int t,int off1,int off2,int off3);

void flatten_cl(cl_mem x, int spatial, int layers, int batch, int forward, cl_mem out,int off1,int off2);
void softmax_tree(cl_mem input, int spatial, int batch, int stride, float temp,cl_mem output, tree hier,int off1,int off2);
void upsample_cl(cl_mem in, int w, int h, int c, int batch, int stride, int forward, float scale, cl_mem out,int off1,int off2);
#endif
#endif
