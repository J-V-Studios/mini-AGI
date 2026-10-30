#include "plasticity.h"
#include "math.h"

enum {
    P_FLOOR_OK = 1
};

static const float LAM = 0.97f;
static const float LAM_FAST = 0.85f;
static const float EFFECT = 45.f;
static const float MIN_EFF = 12.f;
static const float NUDGE = 0.0008f;
static const float NUDGE_DOWN = 0.004f;
static const float T_W_DOWN = 6.f;
static const float T_MID = 2.2f;
static const float T_W = 0.75f;
static const float MAX_MOVE = 0.02f;
static const float FLOOR = 0.01f;
static const float CEIL = 1.f;

void plast_init(Plasticity *p) {
    p->w = p->x = p->y = p->xx = p->xy = p->yy = p->w2 = 0.f;
    p->fw = p->fx = p->fy = p->fxx = p->fxy = p->fyy = p->fw2 = 0.f;
    p->i = 0.f;
    p->scale = 1.f;
    p->last_t = 0.f;
    p->last_e = 0.f;
    p->step = 0;
    p->seen_at = -100000;
}

float plast_nudge(float t) {
    int up = t >= T_MID;
    float gain = up ? NUDGE : NUDGE_DOWN;
    float width = up ? T_W : T_W_DOWN;
    float f = bm_exp(gain * bm_tanh((t - T_MID) / width));
    if (f < 1.f - MAX_MOVE) f = 1.f - MAX_MOVE;
    if (f > 1.f + MAX_MOVE) f = 1.f + MAX_MOVE;
    return f;
}

float plast_factor(const Plasticity *p) {
    float w = 1.f;
    if (p->step < 100) w = (float)(p->step + 1) / 100.f;
    return p->scale * w;
}

void plast_tick(Plasticity *p) { p->step++; }

static void fit_of(float w, float x, float y, float xx, float xy, float yy,
                   float w2, float *t, float *neff, float *e) {
    float sxx, sxy, syy, slope, resid, sigma, n;
    *t = 0.f;
    *neff = 0.f;
    *e = 0.f;
    if (w <= 2.f) return;
    n = w * w / (w2 > 1e-12f ? w2 : 1e-12f);
    *neff = n;
    sxx = xx - x * x / w;
    sxy = xy - x * y / w;
    syy = yy - y * y / w;
    if (sxx <= 0.f || n <= 2.f) return;
    slope = sxy / sxx;
    resid = syy - slope * sxy;
    if (resid < 0.f) resid = 0.f;
    sigma = bm_sqrt(resid / (n - 2.f));
    if (!(sigma > 0.f)) return;
    *t = -slope / (sigma / bm_sqrt(sxx));
    *e = -slope / sigma;
    if (*t > 50.f) *t = 50.f;
    if (*t < -50.f) *t = -50.f;
}

static void accum(float *w, float *x, float *y, float *xx, float *xy,
                  float *yy, float *w2, float lam, float xv, float yv) {
    *w *= lam; *x *= lam; *y *= lam; *xx *= lam; *xy *= lam; *yy *= lam;
    *w2 *= lam * lam;
    *w += 1.f;
    *w2 += 1.f;
    *x += xv;
    *y += yv;
    *xx += xv * xv;
    *xy += xv * yv;
    *yy += yv * yv;
}

float plast_observe(Plasticity *p, float heldout) {
    float ts, ns, es, nf, ef, dummy, t;
    if (p->step > 0 && p->step - p->seen_at < 32) return p->scale;
    p->seen_at = p->step;
    p->i += 1.f;
    accum(&p->w, &p->x, &p->y, &p->xx, &p->xy, &p->yy, &p->w2,
          LAM, p->i, heldout);
    accum(&p->fw, &p->fx, &p->fy, &p->fxx, &p->fxy, &p->fyy, &p->fw2,
          LAM_FAST, p->i, heldout);
    fit_of(p->w, p->x, p->y, p->xx, p->xy, p->yy, p->w2, &ts, &ns, &es);
    fit_of(p->fw, p->fx, p->fy, p->fxx, p->fxy, p->fyy, p->fw2,
           &dummy, &nf, &ef);
    t = ts < EFFECT * es ? ts : EFFECT * es;
    if (nf >= MIN_EFF) {
        float fe = EFFECT * ef;
        if (fe < t) t = fe;
    }
    if (t > 50.f) t = 50.f;
    if (t < -50.f) t = -50.f;
    p->last_t = t;
    p->last_e = es;
    if (ns >= MIN_EFF) {
        p->scale *= plast_nudge(t);
        if (p->scale < FLOOR) p->scale = FLOOR;
        if (p->scale > CEIL) p->scale = CEIL;
    }
    (void)P_FLOOR_OK;
    return p->scale;
}
