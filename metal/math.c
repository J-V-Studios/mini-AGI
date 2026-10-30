#include "math.h"

float bm_fabs(float x) { return x < 0.f ? -x : x; }

float bm_floor(float x) {
    int i = (int)x;
    if ((float)i > x) i--;
    return (float)i;
}

float bm_sqrt(float x) {
    if (!(x > 0.f)) return 0.f;
    return __builtin_sqrtf(x);
}

float bm_exp(float x) {
    union { float f; unsigned u; } u;
    int ni, e;
    float n, r, p;
    if (x > 80.f) return 1e30f;
    if (x < -80.f) return 0.f;
    n = bm_floor(x * 1.4426950408889634f + 0.5f);
    r = x - n * 0.6931471805599453f;
    p = 1.f + r * (1.f + r * (0.5f + r * (0.1666666716f + r * (
        0.0416666679f + r * (0.0083333338f + r * (0.0013888889f +
        r * 0.0001984127f))))));
    u.f = p;
    e = (int)((u.u >> 23) & 255);
    ni = (int)n;
    e += ni;
    if (e <= 0) return 0.f;
    if (e >= 255) return 1e30f;
    u.u = (u.u & 0x807fffffu) | ((unsigned)e << 23);
    return u.f;
}

float bm_log(float x) {
    union { float f; unsigned u; } u;
    int e;
    float m, z, z2, ln;
    if (!(x > 0.f)) return -80.f;
    u.f = x;
    e = (int)((u.u >> 23) & 255) - 127;
    u.u = (u.u & 0x007fffffu) | (127u << 23);
    m = u.f;
    z = (m - 1.f) / (m + 1.f);
    z2 = z * z;
    ln = 2.f * z * (1.f + z2 * (0.3333333433f + z2 * (0.2f + z2 * (
        0.1428571492f + z2 * (0.1111111119f + z2 * 0.0909090936f)))));
    return ln + (float)e * 0.6931471805599453f;
}

static float wrap_pi(float x) {
    const float pi2 = 6.283185307179586f;
    const float pi = 3.141592653589793f;
    x = x - bm_floor(x / pi2) * pi2;
    if (x > pi) x -= pi2;
    return x;
}

float bm_sin(float x) {
    float x2, y;
    x = wrap_pi(x);
    x2 = x * x;
    y = x * (1.f + x2 * (-0.1666666716f + x2 * (0.0083333338f + x2 * (
        -0.0001984127f + x2 * 0.0000027557f))));
    return y;
}

float bm_cos(float x) {
    return bm_sin(x + 1.5707963267948966f);
}

float bm_sigmoid(float x) {
    if (x > 20.f) return 1.f;
    if (x < -20.f) return 0.f;
    return 1.f / (1.f + bm_exp(-x));
}

float bm_tanh(float x) {
    if (x > 10.f) return 1.f;
    if (x < -10.f) return -1.f;
    return 2.f * bm_sigmoid(2.f * x) - 1.f;
}
