#ifndef METAL_MATH_H
#define METAL_MATH_H

/* Freestanding substitutes for libm. The kernel has no C library. */

float bm_floor(float x);
float bm_exp(float x);
float bm_log(float x);
float bm_sqrt(float x);
float bm_sin(float x);
float bm_cos(float x);
float bm_tanh(float x);
float bm_sigmoid(float x);
float bm_fabs(float x);

#endif
