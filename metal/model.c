#include "model.h"

#include "math.h"
#include "mem.h"
#include "plasticity.h"

#define LR_BASE 3.0e-4f
#define TRUNK_MULT 0.1f
#define WD 0.1f
#define CLIP 1.0f
#define PONDER_BETA 0.01f
#define HALT_PRIOR 0.072f
#define HALT_THRESH 0.9f
#define BALANCE 3.5e-5f
#define BIRTH_GATE 0.001f
#define ADAPT_STRENGTH 2.5f
#define ADAPT_DECAY 0.88f
#define ROPE_THETA 10000.f
#define RMS_EPS 1.0e-6f
#define ADAM_B1 0.9f
#define ADAM_B2 0.999f
#define ADAM_EPS 1.0e-8f
#define USAGE_STEPS 1000.f
#define SURVIVAL_CHARS 100000000
#define GROW_EVERY 2000000
#define MAX_GAP 0.4f
#define MAX_SLOTS 40

enum { K_TRUNK = 0, K_ROW = 1, K_EXPERT = 2 };

typedef struct {
    float *w, *g, *m, *v;
    int n;
    int kind;
    float lr_mult;
} Slot;

typedef struct {
    float x[TMAX * D];
    float xn[TMAX * D];
    float rstd1[TMAX];
    float q[N_HEAD * TMAX * HEAD_DIM];
    float k[N_HEAD * TMAX * HEAD_DIM];
    float v[N_HEAD * TMAX * HEAD_DIM];
    float prob[N_HEAD * TMAX * TMAX];
    float att[TMAX * D];
    float res1[TMAX * D];
    float xn2[TMAX * D];
    float rstd2[TMAX];
    float pre1[TMAX * D_FF];
    float pre3[TMAX * D_FF];
    float hid[TMAX * D_FF];
    float y[TMAX * D];
} DenseAct;

typedef struct {
    float x[TMAX * D];
    float xn[TMAX * D];
    float rstd1[TMAX];
    float q[N_HEAD * TMAX * HEAD_DIM];
    float k[N_HEAD * TMAX * HEAD_DIM];
    float v[N_HEAD * TMAX * HEAD_DIM];
    float prob[N_HEAD * TMAX * TMAX];
    float att[TMAX * D];
    float res1[TMAX * D];
    float xn2[TMAX * D];
    float rstd2[TMAX];
    int nres;
    int res[RESIDENT];
    int nmix;
    int idk[TMAX * RESIDENT];
    float wk[TMAX * RESIDENT];
    float pk[TMAX * RESIDENT];
    float rprob[TMAX * RESIDENT];
    float psum[TMAX];
    float e1[TMAX * RESIDENT * EXPERT_FF];
    float e3[TMAX * RESIDENT * EXPERT_FF];
    float eh[TMAX * RESIDENT * EXPERT_FF];
    float ey[TMAX * RESIDENT * D];
    float y[TMAX * D];
    int balanced;
} RecurAct;

typedef struct {
    float *ln1, *qkv, *proj, *ln2, *w1, *w3, *w2;
    float *ln1g, *qkvg, *projg, *ln2g, *w1g, *w3g, *w2g;
} DenseW;

struct Model {
    Slot slots[MAX_SLOTS];
    int nslots;
    DenseW pre[N_PRELUDE];
    DenseW rec;
    float *embed, *embed_g;
    float *router, *router_g;
    float *depth, *depth_g;
    float *gate, *gate_g;
    float *ew1, *ew3, *ew2;
    float *ew1g, *ew3g, *ew2g;
    float *adapter, *adapter_g;
    float *lnf, *lnfg;
    float *halt_w, *halt_g;
    float *halt_b, *halt_bg;
    int n_experts;
    int slots_e[RESIDENT];
    float vote[MAX_EXPERTS];
    float recent[MAX_EXPERTS];
    float recent_view[MAX_EXPERTS];
    float share_used[MAX_EXPERTS];
    float rate[MAX_EXPERTS];
    int last_seen[MAX_EXPERTS];
    int born_chars[MAX_EXPERTS];
    int adm[RESIDENT];
    int n_adm;
    int step;
    int chars;
    int segments;
    int cursor;
    int grow_at;
    float last_loss;
    float last_held;
    float last_steps;
    int have_held;
    uint32_t rng;
    Plasticity plast;
    DenseAct *pa[N_PRELUDE];
    RecurAct *ra[MAX_STEPS];
    float *tokemb;
    float *stream;
    float *hbefore[MAX_STEPS];
    float *hafter[MAX_STEPS];
    float *logits[MAX_STEPS];
    float *ynorm[MAX_STEPS];
    float *rstdf[MAX_STEPS];
    float *lam[MAX_STEPS];
    float *ps[MAX_STEPS];
    float *cs;
    float *ce[MAX_STEPS];
    float *hraw[MAX_STEPS];
    unsigned char *active[MAX_STEPS];
    int curT;
    int training;
    int ids[TMAX + 1];
    float loss;
    float bal_loss;
};

static float rope_cos[CTX * (HEAD_DIM / 2)];
static float rope_sin[CTX * (HEAD_DIM / 2)];
static float *buf_qkv, *buf_a, *buf_b, *buf_c, *buf_ff, *buf_f1, *buf_f2;
static float *buf_h0, *buf_h1, *buf_h2, *buf_h3, *buf_hx, *buf_big;
static float *scr_q, *scr_k, *scr_v;
static int g_freeze, g_soft;
static int mem_ready;
static int tr_ids[640];
static int tr_n;
static int ho_ids[160];
static int ho_n;

static const char *SPEC[9] = {
    "<think>", "</think>", "<user>", "</user>", "<bot>", "</bot>",
    "<g>", "</g>", "<|endoftext|>"
};

static const char *TRAIN_TEXT =
    "the cat sat. the cat sat. the cat sat. the cat sat.\n"
    "the cat sat on the mat. the cat sat on the mat.\n"
    "the cat sat. the cat sat. the cat sat.\n"
    "add 2 + 2 = 4. add 2 + 2 = 4. add 2 + 2 = 4.\n"
    "add 1 + 1 = 2. add 1 + 1 = 2. add 1 + 1 = 2.\n"
    "def add(a, b):\n    return a + b\n"
    "<user>\nhi\n</user>\n<bot>\nhello\n</bot>\n"
    "once upon a time a cat sat on the mat.\n"
    "the cat sat. the cat sat. the cat sat. the cat sat.\n";

static const char *HELD_TEXT =
    "the dog ran. the dog ran. the dog ran. the dog ran fast.\n";

static int slen(const char *s) {
    int n = 0;
    while (s[n]) n++;
    return n;
}

static int starts(const char *s, const char *p) {
    while (*p) {
        if (*s != *p) return 0;
        s++;
        p++;
    }
    return 1;
}

int tok_encode(const char *s, int *out, int cap) {
    int n = 0;
    while (*s && n < cap) {
        int k, hit = -1, hlen = 0;
        for (k = 0; k < 9; k++) {
            int L = slen(SPEC[k]);
            if (L > hlen && starts(s, SPEC[k])) {
                hit = k;
                hlen = L;
            }
        }
        if (hit >= 0) {
            out[n++] = 256 + hit;
            s += hlen;
        } else {
            out[n++] = (unsigned char)*s++;
        }
    }
    return n;
}

int tok_decode(const int *ids, int n, char *out, int cap) {
    int u = 0, i;
    if (cap <= 0) return 0;
    for (i = 0; i < n && u + 1 < cap; i++) {
        int id = ids[i];
        if (id >= 256 && id < 265) {
            const char *p = SPEC[id - 256];
            while (*p && u + 1 < cap) out[u++] = *p++;
        } else if (id >= 32 && id < 127) {
            out[u++] = (char)id;
        } else if (id == '\n' || id == '\t') {
            out[u++] = (char)id;
        } else if (id >= 0 && id < 256) {
            out[u++] = '.';
        }
    }
    out[u] = 0;
    return u;
}

static void zero(float *p, int n) {
    int i;
    for (i = 0; i < n; i++) p[i] = 0.f;
}

static float dot(const float *a, const float *b, int n) {
    float s = 0.f;
    int i;
    for (i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float rndu(Model *m) {
    m->rng ^= m->rng << 13;
    m->rng ^= m->rng >> 17;
    m->rng ^= m->rng << 5;
    return (m->rng + 1u) * (1.f / 4294967296.f);
}

static float rndn(Model *m) {
    float u = rndu(m);
    float v = rndu(m);
    if (u < 1e-7f) u = 1e-7f;
    return bm_sqrt(-2.f * bm_log(u)) * bm_cos(6.28318530718f * v);
}

static void fill_n(Model *m, float *p, int n, float std) {
    int i;
    for (i = 0; i < n; i++) p[i] = rndn(m) * std;
}

static float *param(Model *m, int n, int kind, float lr_mult) {
    Slot *s = &m->slots[m->nslots++];
    s->n = n;
    s->kind = kind;
    s->lr_mult = lr_mult;
    s->w = kalloc((size_t)n * sizeof(float));
    s->g = kalloc((size_t)n * sizeof(float));
    s->m = kalloc((size_t)n * sizeof(float));
    s->v = kalloc((size_t)n * sizeof(float));
    return s->w;
}

static void build_rope(void) {
    int pos, i;
    float half = (float)HEAD_DIM;
    for (pos = 0; pos < CTX; pos++) {
        for (i = 0; i < HEAD_DIM / 2; i++) {
            float inv = bm_exp(-bm_log(ROPE_THETA) * ((float)(2 * i) / half));
            float ang = (float)pos * inv;
            rope_cos[pos * (HEAD_DIM / 2) + i] = bm_cos(ang);
            rope_sin[pos * (HEAD_DIM / 2) + i] = bm_sin(ang);
        }
    }
}

static void linear(const float *x, const float *W, float *y,
                   int T, int In, int Out) {
    int t, o, i;
    for (t = 0; t < T; t++) {
        for (o = 0; o < Out; o++) {
            float s = 0.f;
            const float *w = W + o * In;
            const float *xx = x + t * In;
            for (i = 0; i < In; i++) s += xx[i] * w[i];
            y[t * Out + o] = s;
        }
    }
}

static void linear_bwd(const float *x, const float *W, const float *dy,
                       float *dx, float *dW, int T, int In, int Out) {
    int t, o, i;
    for (t = 0; t < T; t++) {
        for (o = 0; o < Out; o++) {
            float g = dy[t * Out + o];
            const float *w = W + o * In;
            float *dw = dW + o * In;
            float *dxx = dx + t * In;
            const float *xx = x + t * In;
            for (i = 0; i < In; i++) {
                dw[i] += g * xx[i];
                dxx[i] += g * w[i];
            }
        }
    }
}

static void rmsnorm(const float *w, const float *x, float *y, float *rstd,
                    int T) {
    int t, d;
    for (t = 0; t < T; t++) {
        float ss = 0.f;
        for (d = 0; d < D; d++) {
            float v = x[t * D + d];
            ss += v * v;
        }
        ss = ss / (float)D + RMS_EPS;
        rstd[t] = 1.f / bm_sqrt(ss);
        for (d = 0; d < D; d++)
            y[t * D + d] = x[t * D + d] * rstd[t] * w[d];
    }
}

static void rmsnorm_bwd(const float *w, const float *x, const float *rstd,
                        const float *dy, float *dx, float *dw, int T) {
    int t, d;
    for (t = 0; t < T; t++) {
        float dxhat[D];
        float mean = 0.f;
        float r = rstd[t];
        for (d = 0; d < D; d++) {
            dxhat[d] = dy[t * D + d] * w[d];
            dw[d] += dy[t * D + d] * x[t * D + d] * r;
            mean += dxhat[d] * x[t * D + d];
        }
        mean /= (float)D;
        for (d = 0; d < D; d++)
            dx[t * D + d] += r * (dxhat[d] - x[t * D + d] * r * r * mean);
    }
}

static void rope_inplace(float *q, int T) {
    int h, t, i;
    for (h = 0; h < N_HEAD; h++) {
        for (t = 0; t < T; t++) {
            float *qq = q + (h * T + t) * HEAD_DIM;
            for (i = 0; i < HEAD_DIM / 2; i++) {
                float c = rope_cos[t * (HEAD_DIM / 2) + i];
                float s = rope_sin[t * (HEAD_DIM / 2) + i];
                float x1 = qq[2 * i];
                float x2 = qq[2 * i + 1];
                qq[2 * i] = x1 * c - x2 * s;
                qq[2 * i + 1] = x1 * s + x2 * c;
            }
        }
    }
}

static void rope_bwd(float *dq, int T) {
    int h, t, i;
    for (h = 0; h < N_HEAD; h++) {
        for (t = 0; t < T; t++) {
            float *qq = dq + (h * T + t) * HEAD_DIM;
            for (i = 0; i < HEAD_DIM / 2; i++) {
                float c = rope_cos[t * (HEAD_DIM / 2) + i];
                float s = rope_sin[t * (HEAD_DIM / 2) + i];
                float g1 = qq[2 * i];
                float g2 = qq[2 * i + 1];
                qq[2 * i] = g1 * c + g2 * s;
                qq[2 * i + 1] = -g1 * s + g2 * c;
            }
        }
    }
}

static void attention_fwd(const float *xn, const float *Wqkv, const float *Wproj,
                          float *q, float *k, float *v, float *prob,
                          float *att, float *out, int T) {
    int t, h, d, s;
    float scale = 1.f / bm_sqrt((float)HEAD_DIM);
    linear(xn, Wqkv, buf_qkv, T, D, 3 * D);
    for (t = 0; t < T; t++) {
        for (h = 0; h < N_HEAD; h++) {
            for (d = 0; d < HEAD_DIM; d++) {
                q[(h * T + t) * HEAD_DIM + d] =
                    buf_qkv[t * 3 * D + h * HEAD_DIM + d];
                k[(h * T + t) * HEAD_DIM + d] =
                    buf_qkv[t * 3 * D + D + h * HEAD_DIM + d];
                v[(h * T + t) * HEAD_DIM + d] =
                    buf_qkv[t * 3 * D + 2 * D + h * HEAD_DIM + d];
            }
        }
    }
    rope_inplace(q, T);
    rope_inplace(k, T);
    for (h = 0; h < N_HEAD; h++) {
        for (t = 0; t < T; t++) {
            float sc[TMAX];
            float mx = -1e30f, sum = 0.f;
            const float *qt = q + (h * T + t) * HEAD_DIM;
            for (s = 0; s <= t; s++) {
                sc[s] = dot(qt, k + (h * T + s) * HEAD_DIM, HEAD_DIM) * scale;
                if (sc[s] > mx) mx = sc[s];
            }
            for (s = 0; s <= t; s++) {
                float e = bm_exp(sc[s] - mx);
                prob[(h * T + t) * TMAX + s] = e;
                sum += e;
            }
            if (sum < 1e-12f) sum = 1e-12f;
            for (s = 0; s <= t; s++) prob[(h * T + t) * TMAX + s] /= sum;
            for (s = t + 1; s < TMAX; s++) prob[(h * T + t) * TMAX + s] = 0.f;
        }
    }
    zero(att, T * D);
    for (h = 0; h < N_HEAD; h++) {
        for (t = 0; t < T; t++) {
            for (d = 0; d < HEAD_DIM; d++) {
                float o = 0.f;
                for (s = 0; s <= t; s++)
                    o += prob[(h * T + t) * TMAX + s] *
                         v[(h * T + s) * HEAD_DIM + d];
                att[t * D + h * HEAD_DIM + d] = o;
            }
        }
    }
    linear(att, Wproj, out, T, D, D);
}

static void attention_bwd(const float *xn, const float *Wqkv, const float *Wproj,
                          const float *q, const float *k, const float *v,
                          const float *prob, const float *att,
                          const float *dout, float *dxn,
                          float *dWqkv, float *dWproj, int T) {
    int t, h, d, s;
    float scale = 1.f / bm_sqrt((float)HEAD_DIM);
    zero(buf_a, T * D);
    linear_bwd(att, Wproj, dout, buf_a, dWproj, T, D, D);
    zero(scr_q, N_HEAD * T * HEAD_DIM);
    zero(scr_k, N_HEAD * T * HEAD_DIM);
    zero(scr_v, N_HEAD * T * HEAD_DIM);
    for (h = 0; h < N_HEAD; h++) {
        for (t = 0; t < T; t++) {
            float dp[TMAX], pr[TMAX], dz[TMAX];
            float dotp = 0.f;
            for (s = 0; s < T; s++) {
                pr[s] = prob[(h * T + t) * TMAX + s];
                dp[s] = 0.f;
                for (d = 0; d < HEAD_DIM; d++)
                    dp[s] += buf_a[t * D + h * HEAD_DIM + d] *
                             v[(h * T + s) * HEAD_DIM + d];
                dotp += pr[s] * dp[s];
            }
            for (s = 0; s < T; s++)
                dz[s] = pr[s] * (dp[s] - dotp) * scale;
            for (d = 0; d < HEAD_DIM; d++) {
                float gv = 0.f;
                for (s = 0; s <= t; s++)
                    gv += prob[(h * T + t) * TMAX + s] *
                          buf_a[t * D + h * HEAD_DIM + d];
                scr_v[(h * T + t) * HEAD_DIM + d] += 0.f;
                (void)gv;
            }
            for (s = 0; s <= t; s++) {
                for (d = 0; d < HEAD_DIM; d++) {
                    float go = buf_a[t * D + h * HEAD_DIM + d];
                    scr_v[(h * T + s) * HEAD_DIM + d] +=
                        prob[(h * T + t) * TMAX + s] * go;
                    scr_q[(h * T + t) * HEAD_DIM + d] +=
                        dz[s] * k[(h * T + s) * HEAD_DIM + d];
                    scr_k[(h * T + s) * HEAD_DIM + d] +=
                        dz[s] * q[(h * T + t) * HEAD_DIM + d];
                }
            }
        }
    }
    rope_bwd(scr_q, T);
    rope_bwd(scr_k, T);
    zero(buf_qkv, T * 3 * D);
    for (t = 0; t < T; t++) {
        for (h = 0; h < N_HEAD; h++) {
            for (d = 0; d < HEAD_DIM; d++) {
                buf_qkv[t * 3 * D + h * HEAD_DIM + d] =
                    scr_q[(h * T + t) * HEAD_DIM + d];
                buf_qkv[t * 3 * D + D + h * HEAD_DIM + d] =
                    scr_k[(h * T + t) * HEAD_DIM + d];
                buf_qkv[t * 3 * D + 2 * D + h * HEAD_DIM + d] =
                    scr_v[(h * T + t) * HEAD_DIM + d];
            }
        }
    }
    zero(dxn, T * D);
    linear_bwd(xn, Wqkv, buf_qkv, dxn, dWqkv, T, D, 3 * D);
}

static void swiglu_fwd(const float *x, const float *W1, const float *W3,
                       const float *W2, float *pre1, float *pre3, float *hid,
                       float *y, int T, int FF) {
    int t, f;
    linear(x, W1, pre1, T, D, FF);
    linear(x, W3, pre3, T, D, FF);
    for (t = 0; t < T; t++) {
        for (f = 0; f < FF; f++) {
            float z = pre1[t * FF + f];
            float sig = bm_sigmoid(z);
            hid[t * FF + f] = (z * sig) * pre3[t * FF + f];
        }
    }
    linear(hid, W2, y, T, FF, D);
}

static void swiglu_bwd(const float *x, const float *W1, const float *W3,
                       const float *W2, const float *pre1, const float *pre3,
                       const float *hid, const float *dy, float *dx,
                       float *dW1, float *dW3, float *dW2, int T, int FF) {
    int t, f;
    zero(buf_ff, T * FF);
    linear_bwd(hid, W2, dy, buf_ff, dW2, T, FF, D);
    zero(buf_f1, T * FF);
    zero(buf_f2, T * FF);
    for (t = 0; t < T; t++) {
        for (f = 0; f < FF; f++) {
            float z = pre1[t * FF + f];
            float sig = bm_sigmoid(z);
            float dh = buf_ff[t * FF + f];
            float dsilu = sig * (1.f + z * (1.f - sig));
            buf_f2[t * FF + f] = dh * (z * sig);
            buf_f1[t * FF + f] = dh * pre3[t * FF + f] * dsilu;
        }
    }
    linear_bwd(x, W1, buf_f1, dx, dW1, T, D, FF);
    linear_bwd(x, W3, buf_f2, dx, dW3, T, D, FF);
}

static void dense_fwd(const DenseW *w, DenseAct *a, const float *x, int T) {
    int t, d;
    for (t = 0; t < T; t++)
        for (d = 0; d < D; d++) a->x[t * D + d] = x[t * D + d];
    rmsnorm(w->ln1, a->x, a->xn, a->rstd1, T);
    attention_fwd(a->xn, w->qkv, w->proj, a->q, a->k, a->v, a->prob,
                  a->att, buf_a, T);
    for (t = 0; t < T; t++)
        for (d = 0; d < D; d++)
            a->res1[t * D + d] = a->x[t * D + d] + buf_a[t * D + d];
    rmsnorm(w->ln2, a->res1, a->xn2, a->rstd2, T);
    swiglu_fwd(a->xn2, w->w1, w->w3, w->w2, a->pre1, a->pre3, a->hid,
               buf_b, T, D_FF);
    for (t = 0; t < T; t++)
        for (d = 0; d < D; d++)
            a->y[t * D + d] = a->res1[t * D + d] + buf_b[t * D + d];
}

static void dense_bwd(const DenseW *w, const DenseAct *a, const float *dy,
                      float *dx, int T) {
    int t, d;
    for (t = 0; t < T; t++)
        for (d = 0; d < D; d++) buf_h0[t * D + d] = dy[t * D + d];
    zero(buf_h1, T * D);
    swiglu_bwd(a->xn2, w->w1, w->w3, w->w2, a->pre1, a->pre3, a->hid,
               buf_h0, buf_h1, w->w1g, w->w3g, w->w2g, T, D_FF);
    rmsnorm_bwd(w->ln2, a->res1, a->rstd2, buf_h1, buf_h0, w->ln2g, T);
    zero(buf_h1, T * D);
    attention_bwd(a->xn, w->qkv, w->proj, a->q, a->k, a->v, a->prob, a->att,
                  buf_h0, buf_h1, w->qkvg, w->projg, T);
    zero(dx, T * D);
    for (t = 0; t < T; t++)
        for (d = 0; d < D; d++) dx[t * D + d] = buf_h0[t * D + d];
    rmsnorm_bwd(w->ln1, a->x, a->rstd1, buf_h1, dx, w->ln1g, T);
}

static int e_off_ff(int e, int f) { return (e * EXPERT_FF + f) * D; }
static int e_off_w2(int e, int d) { return (e * D + d) * EXPERT_FF; }

static void expert_fwd_one(const Model *m, int e, const float *x,
                           float *pre1, float *pre3, float *hid, float *y) {
    int f, d;
    for (f = 0; f < EXPERT_FF; f++) {
        pre1[f] = dot(x, m->ew1 + e_off_ff(e, f), D);
        pre3[f] = dot(x, m->ew3 + e_off_ff(e, f), D);
        hid[f] = (pre1[f] * bm_sigmoid(pre1[f])) * pre3[f];
    }
    for (d = 0; d < D; d++)
        y[d] = dot(hid, m->ew2 + e_off_w2(e, d), EXPERT_FF);
}

static void expert_bwd_one(Model *m, int e, const float *x, const float *pre1,
                           const float *pre3, const float *hid, const float *dy,
                           float *dx) {
    int f, d;
    float dhid[EXPERT_FF];
    float dp1[EXPERT_FF], dp3[EXPERT_FF];
    zero(dhid, EXPERT_FF);
    for (d = 0; d < D; d++) {
        float g = dy[d];
        const float *w2 = m->ew2 + e_off_w2(e, d);
        float *gw2 = m->ew2g + e_off_w2(e, d);
        for (f = 0; f < EXPERT_FF; f++) {
            gw2[f] += g * hid[f];
            dhid[f] += g * w2[f];
        }
    }
    for (f = 0; f < EXPERT_FF; f++) {
        float z = pre1[f];
        float sig = bm_sigmoid(z);
        float dsilu = sig * (1.f + z * (1.f - sig));
        dp3[f] = dhid[f] * (z * sig);
        dp1[f] = dhid[f] * pre3[f] * dsilu;
    }
    for (f = 0; f < EXPERT_FF; f++) {
        float *gw1 = m->ew1g + e_off_ff(e, f);
        float *gw3 = m->ew3g + e_off_ff(e, f);
        const float *w1 = m->ew1 + e_off_ff(e, f);
        const float *w3 = m->ew3 + e_off_ff(e, f);
        for (d = 0; d < D; d++) {
            gw1[d] += dp1[f] * x[d];
            gw3[d] += dp3[f] * x[d];
            dx[d] += dp1[f] * w1[d] + dp3[f] * w3[d];
        }
    }
}

static void softmax_row(const float *z, float *p, int n) {
    int i;
    float mx = z[0], sum = 0.f;
    for (i = 1; i < n; i++) if (z[i] > mx) mx = z[i];
    for (i = 0; i < n; i++) {
        p[i] = bm_exp(z[i] - mx);
        sum += p[i];
    }
    if (sum < 1e-12f) sum = 1e-12f;
    for (i = 0; i < n; i++) p[i] /= sum;
}

static void share_now(Model *m, float *share) {
    int e;
    float s = 0.f;
    for (e = 0; e < m->n_experts; e++) s += m->recent_view[e];
    if (s <= 0.f) {
        for (e = 0; e < m->n_experts; e++) share[e] = 1.f / (float)m->n_experts;
    } else {
        for (e = 0; e < m->n_experts; e++) share[e] = m->recent_view[e] / s;
    }
    for (; e < MAX_EXPERTS; e++) share[e] = 0.f;
}

static void admit(Model *m, const float *mass) {
    int e, i, s, nw = 0;
    int order[MAX_EXPERTS];
    int want[RESIDENT];
    int neu[RESIDENT];
    int taken[MAX_EXPERTS];
    if (g_freeze) {
        m->n_adm = 0;
        for (s = 0; s < RESIDENT; s++) {
            if (m->slots_e[s] >= 0) m->adm[m->n_adm++] = m->slots_e[s];
        }
        return;
    }
    for (e = 0; e < m->n_experts; e++) m->vote[e] += mass[e];
    for (e = 0; e < m->n_experts; e++) order[e] = e;
    for (i = 0; i < m->n_experts; i++) {
        for (s = i + 1; s < m->n_experts; s++) {
            if (m->vote[order[s]] > m->vote[order[i]]) {
                int tmp = order[i];
                order[i] = order[s];
                order[s] = tmp;
            }
        }
    }
    for (i = 0; i < m->n_experts && nw < RESIDENT; i++) want[nw++] = order[i];
    for (e = 0; e < MAX_EXPERTS; e++) taken[e] = 0;
    for (s = 0; s < RESIDENT; s++) neu[s] = -1;
    for (s = 0; s < RESIDENT; s++) {
        int ee = m->slots_e[s];
        int keep = 0;
        for (i = 0; i < nw; i++) if (want[i] == ee) keep = 1;
        if (ee >= 0 && keep) {
            neu[s] = ee;
            taken[ee] = 1;
        }
    }
    i = 0;
    for (s = 0; s < RESIDENT; s++) {
        if (neu[s] >= 0) continue;
        while (i < nw && taken[want[i]]) i++;
        if (i < nw) {
            neu[s] = want[i];
            taken[want[i]] = 1;
            i++;
        }
    }
    for (s = 0; s < RESIDENT; s++) m->slots_e[s] = neu[s];
    m->n_adm = 0;
    for (i = 0; i < nw; i++) {
        m->adm[m->n_adm++] = want[i];
        m->recent_view[want[i]] += 1.f / USAGE_STEPS;
    }
}

static void moe_fwd(Model *m, RecurAct *a, const unsigned char *active,
                    int T, int do_admit, int do_balance) {
    int t, e, j, k, d, s;
    float mass[MAX_EXPERTS];
    zero(mass, MAX_EXPERTS);
    if (do_admit && !g_freeze) {
        for (t = 0; t < T; t++) {
            float z[MAX_EXPERTS], p[MAX_EXPERTS];
            float h[D];
            if (!active[t]) continue;
            for (d = 0; d < D; d++) h[d] = a->xn2[t * D + d] + m->depth[d];
            for (e = 0; e < m->n_experts; e++)
                z[e] = dot(m->router + e * D, h, D);
            softmax_row(z, p, m->n_experts);
            for (k = 0; k < TOP_K; k++) {
                int best = 0;
                for (e = 1; e < m->n_experts; e++)
                    if (p[e] > p[best]) best = e;
                mass[best] += p[best];
                p[best] = -1.f;
            }
        }
        admit(m, mass);
    } else if (g_freeze && do_admit) {
        admit(m, mass);
    }
    a->nres = 0;
    for (s = 0; s < RESIDENT; s++) {
        if (m->slots_e[s] >= 0 && m->slots_e[s] < m->n_experts)
            a->res[a->nres++] = m->slots_e[s];
    }
    a->balanced = 0;
    if (do_balance && m->training && a->nres > 0) {
        float share[MAX_EXPERTS];
        float P[MAX_EXPERTS];
        float acc = 0.f;
        share_now(m, share);
        for (e = 0; e < MAX_EXPERTS; e++) {
            m->share_used[e] = share[e];
            P[e] = 0.f;
        }
        for (t = 0; t < T; t++) {
            float z[MAX_EXPERTS], p[MAX_EXPERTS], h[D];
            for (d = 0; d < D; d++) h[d] = a->xn2[t * D + d] + m->depth[d];
            for (e = 0; e < m->n_experts; e++)
                z[e] = dot(m->router + e * D, h, D);
            softmax_row(z, p, m->n_experts);
            for (e = 0; e < m->n_experts; e++) P[e] += p[e];
        }
        for (e = 0; e < m->n_experts; e++) {
            P[e] /= (float)T;
            acc += share[e] * P[e];
        }
        m->bal_loss = BALANCE * ((float)m->n_experts * acc - 1.f);
        a->balanced = 1;
    }
    a->nmix = 0;
    for (t = 0; t < T; t++) {
        float z[RESIDENT], p[RESIDENT], h[D];
        int n = a->nres;
        int mix;
        for (d = 0; d < D; d++) a->y[t * D + d] = 0.f;
        a->psum[t] = 1.f;
        if (!active[t] || n <= 0) continue;
        for (d = 0; d < D; d++) h[d] = a->xn2[t * D + d] + m->depth[d];
        for (j = 0; j < n; j++)
            z[j] = dot(m->router + a->res[j] * D, h, D);
        softmax_row(z, p, n);
        for (j = 0; j < n; j++) a->rprob[t * RESIDENT + j] = p[j];
        if (g_soft) {
            mix = n;
            for (k = 0; k < mix; k++) {
                a->idk[t * RESIDENT + k] = k;
                a->pk[t * RESIDENT + k] = p[k];
                a->wk[t * RESIDENT + k] = p[k] * m->gate[a->res[k]];
            }
            a->psum[t] = 1.f;
        } else {
            float raw[TOP_K];
            int idx[TOP_K];
            float ps = 0.f;
            float pp[RESIDENT];
            for (j = 0; j < n; j++) pp[j] = p[j];
            mix = TOP_K < n ? TOP_K : n;
            for (k = 0; k < mix; k++) {
                int best = 0;
                for (j = 1; j < n; j++) if (pp[j] > pp[best]) best = j;
                idx[k] = best;
                raw[k] = pp[best];
                ps += pp[best];
                pp[best] = -1.f;
            }
            if (ps < 1e-12f) ps = 1e-12f;
            a->psum[t] = ps;
            for (k = 0; k < mix; k++) {
                int slot = idx[k];
                a->idk[t * RESIDENT + k] = slot;
                a->pk[t * RESIDENT + k] = raw[k];
                a->wk[t * RESIDENT + k] =
                    (raw[k] / ps) * m->gate[a->res[slot]];
            }
        }
        a->nmix = mix;
        for (k = 0; k < mix; k++) {
            int slot = a->idk[t * RESIDENT + k];
            int ee = a->res[slot];
            float *pre1 = a->e1 + (t * RESIDENT + k) * EXPERT_FF;
            float *pre3 = a->e3 + (t * RESIDENT + k) * EXPERT_FF;
            float *hid = a->eh + (t * RESIDENT + k) * EXPERT_FF;
            float *ey = a->ey + (t * RESIDENT + k) * D;
            const float *xx = a->xn2 + t * D;
            expert_fwd_one(m, ee, xx, pre1, pre3, hid, ey);
            for (d = 0; d < D; d++)
                a->y[t * D + d] += a->wk[t * RESIDENT + k] * ey[d];
        }
    }
}

static void moe_bwd(Model *m, const RecurAct *a, const unsigned char *active,
                    const float *dy, float *dx, int T) {
    int t, k, d, j, e;
    zero(dx, T * D);
    for (t = 0; t < T; t++) {
        int mix = a->nmix;
        float dprob[RESIDENT];
        float h[D];
        if (!active[t] || a->nres <= 0 || mix <= 0) continue;
        for (j = 0; j < RESIDENT; j++) dprob[j] = 0.f;
        for (d = 0; d < D; d++) h[d] = a->xn2[t * D + d] + m->depth[d];
        for (k = 0; k < mix; k++) {
            int slot = a->idk[t * RESIDENT + k];
            int ee = a->res[slot];
            const float *ey = a->ey + (t * RESIDENT + k) * D;
            const float *xx = a->xn2 + t * D;
            const float *pre1 = a->e1 + (t * RESIDENT + k) * EXPERT_FF;
            const float *pre3 = a->e3 + (t * RESIDENT + k) * EXPERT_FF;
            const float *hid = a->eh + (t * RESIDENT + k) * EXPERT_FF;
            float dey[D];
            float w = a->wk[t * RESIDENT + k];
            float dotted = 0.f;
            for (d = 0; d < D; d++) {
                dey[d] = dy[t * D + d] * w;
                dotted += dy[t * D + d] * ey[d];
            }
            expert_bwd_one(m, ee, xx, pre1, pre3, hid, dey, dx + t * D);
            if (g_soft) {
                float pk = a->pk[t * RESIDENT + k];
                m->gate_g[ee] += dotted * pk;
                dprob[slot] += dotted * m->gate[ee];
            } else {
                float raw = a->pk[t * RESIDENT + k];
                float ps = a->psum[t];
                float un = raw / ps;
                float dun = dotted * m->gate[ee];
                m->gate_g[ee] += dotted * un;
                /* renorm backward folded below after the k loop */
                dprob[slot] += dun / ps;
                /* the d(1/s) term needs every component; accumulate into a side channel */
                (void)un;
            }
        }
        if (!g_soft) {
            float ps = a->psum[t];
            float ds = 0.f;
            int mix2 = mix;
            for (k = 0; k < mix2; k++) {
                float raw = a->pk[t * RESIDENT + k];
                float dun = 0.f;
                /* recompute dun = dotted * gate from the gate grad piece we want.
                   Easier: recompute dotted. */
                int slot = a->idk[t * RESIDENT + k];
                const float *ey = a->ey + (t * RESIDENT + k) * D;
                float dotted = 0.f;
                for (d = 0; d < D; d++) dotted += dy[t * D + d] * ey[d];
                dun = dotted * m->gate[a->res[slot]];
                ds += dun * (-raw / (ps * ps));
            }
            for (k = 0; k < mix2; k++) {
                int slot = a->idk[t * RESIDENT + k];
                dprob[slot] += ds;
            }
        }
        {
            float pr[RESIDENT], dp[RESIDENT], dotp = 0.f, zstub;
            int n = a->nres;
            for (j = 0; j < n; j++) {
                pr[j] = a->rprob[t * RESIDENT + j];
                dp[j] = dprob[j];
                dotp += pr[j] * dp[j];
            }
            for (j = 0; j < n; j++) {
                float dz = pr[j] * (dp[j] - dotp);
                int ee = a->res[j];
                for (d = 0; d < D; d++) {
                    m->router_g[ee * D + d] += dz * h[d];
                    dx[t * D + d] += dz * m->router[ee * D + d];
                    m->depth_g[d] += dz * m->router[ee * D + d];
                }
            }
            (void)zstub;
        }
    }
    if (a->balanced) {
        float P[MAX_EXPERTS];
        float dP[MAX_EXPERTS];
        for (e = 0; e < m->n_experts; e++) {
            P[e] = 0.f;
            dP[e] = BALANCE * (float)m->n_experts * m->share_used[e];
        }
        for (t = 0; t < T; t++) {
            float z[MAX_EXPERTS], p[MAX_EXPERTS], h[D];
            float dotp = 0.f;
            for (d = 0; d < D; d++) h[d] = a->xn2[t * D + d] + m->depth[d];
            for (e = 0; e < m->n_experts; e++)
                z[e] = dot(m->router + e * D, h, D);
            softmax_row(z, p, m->n_experts);
            for (e = 0; e < m->n_experts; e++) dotp += p[e] * (dP[e] / (float)T);
            for (e = 0; e < m->n_experts; e++) {
                float dz = p[e] * (dP[e] / (float)T - dotp);
                for (d = 0; d < D; d++)
                    m->router_g[e * D + d] += dz * h[d];
            }
            (void)P;
        }
    }
}

static void recur_fwd(Model *m, const DenseW *w, RecurAct *a, const float *x,
                      const unsigned char *active, int T,
                      int do_admit, int do_balance) {
    int t, d;
    for (t = 0; t < T; t++)
        for (d = 0; d < D; d++) a->x[t * D + d] = x[t * D + d];
    rmsnorm(w->ln1, a->x, a->xn, a->rstd1, T);
    attention_fwd(a->xn, w->qkv, w->proj, a->q, a->k, a->v, a->prob,
                  a->att, buf_a, T);
    for (t = 0; t < T; t++)
        for (d = 0; d < D; d++)
            a->res1[t * D + d] = a->x[t * D + d] + buf_a[t * D + d];
    rmsnorm(w->ln2, a->res1, a->xn2, a->rstd2, T);
    moe_fwd(m, a, active, T, do_admit, do_balance);
    for (t = 0; t < T; t++)
        for (d = 0; d < D; d++)
            a->y[t * D + d] = a->res1[t * D + d] + a->y[t * D + d];
}

static void recur_bwd(Model *m, const DenseW *w, const RecurAct *a,
                      const unsigned char *active, const float *dy,
                      float *dx, int T) {
    int t, d;
    for (t = 0; t < T; t++)
        for (d = 0; d < D; d++) buf_h0[t * D + d] = dy[t * D + d];
    zero(buf_h1, T * D);
    moe_bwd(m, a, active, buf_h0, buf_h1, T);
    rmsnorm_bwd(w->ln2, a->res1, a->rstd2, buf_h1, buf_h0, w->ln2g, T);
    zero(buf_h1, T * D);
    attention_bwd(a->xn, w->qkv, w->proj, a->q, a->k, a->v, a->prob, a->att,
                  buf_h0, buf_h1, w->qkvg, w->projg, T);
    zero(dx, T * D);
    for (t = 0; t < T; t++)
        for (d = 0; d < D; d++) dx[t * D + d] = buf_h0[t * D + d];
    rmsnorm_bwd(w->ln1, a->x, a->rstd1, buf_h1, dx, w->ln1g, T);
}

static float ce_one(const float *logits, int target, float *drow, float scale) {
    float p[VOCAB];
    float mx = logits[0], sum = 0.f, pt;
    int i;
    for (i = 1; i < VOCAB; i++) if (logits[i] > mx) mx = logits[i];
    for (i = 0; i < VOCAB; i++) {
        p[i] = bm_exp(logits[i] - mx);
        sum += p[i];
    }
    if (sum < 1e-12f) sum = 1e-12f;
    pt = p[target] / sum;
    if (pt < 1e-12f) pt = 1e-12f;
    if (drow) {
        for (i = 0; i < VOCAB; i++) drow[i] = p[i] / sum * scale;
        if (target >= 0 && target < VOCAB) drow[target] -= scale;
    }
    return -bm_log(pt);
}

static void adapter_fwd(const float *A, const float *h, const float *x,
                        float *y, int T) {
    int t, d, i;
    for (t = 0; t < T; t++) {
        for (d = 0; d < D; d++) {
            float s = 0.f;
            const float *w = A + d * (2 * D);
            for (i = 0; i < D; i++) s += h[t * D + i] * w[i];
            for (i = 0; i < D; i++) s += x[t * D + i] * w[D + i];
            y[t * D + d] = s;
        }
    }
}

static void adapter_bwd(const float *A, const float *h, const float *x,
                        const float *dy, float *dh, float *dx, float *dA,
                        int T) {
    int t, d, i;
    for (t = 0; t < T; t++) {
        for (d = 0; d < D; d++) {
            float g = dy[t * D + d];
            float *dw = dA + d * (2 * D);
            const float *w = A + d * (2 * D);
            for (i = 0; i < D; i++) {
                dw[i] += g * h[t * D + i];
                dh[t * D + i] += g * w[i];
                dw[D + i] += g * x[t * D + i];
                dx[t * D + i] += g * w[D + i];
            }
        }
    }
}

static void bind_dense(Model *m, DenseW *w, int moe_instead) {
    float std_out = 0.02f / bm_sqrt(2.f * (float)(N_PRELUDE + 1));
    w->ln1 = param(m, D, K_TRUNK, TRUNK_MULT);
    w->qkv = param(m, 3 * D * D, K_TRUNK, TRUNK_MULT);
    w->proj = param(m, D * D, K_TRUNK, TRUNK_MULT);
    w->ln2 = param(m, D, K_TRUNK, TRUNK_MULT);
    {
        int i;
        Slot *s = &m->slots[m->nslots - 4];
        w->ln1g = s[0].g;
        w->qkvg = s[1].g;
        w->projg = s[2].g;
        w->ln2g = s[3].g;
        for (i = 0; i < D; i++) w->ln1[i] = 1.f;
        for (i = 0; i < D; i++) w->ln2[i] = 1.f;
    }
    fill_n(m, w->qkv, 3 * D * D, 0.02f);
    fill_n(m, w->proj, D * D, std_out);
    if (!moe_instead) {
        w->w1 = param(m, D_FF * D, K_TRUNK, TRUNK_MULT);
        w->w3 = param(m, D_FF * D, K_TRUNK, TRUNK_MULT);
        w->w2 = param(m, D * D_FF, K_TRUNK, TRUNK_MULT);
        w->w1g = m->slots[m->nslots - 3].g;
        w->w3g = m->slots[m->nslots - 2].g;
        w->w2g = m->slots[m->nslots - 1].g;
        fill_n(m, w->w1, D_FF * D, 0.02f);
        fill_n(m, w->w3, D_FF * D, 0.02f);
        fill_n(m, w->w2, D * D_FF, std_out);
    }
}

static void zero_expert_mom(Model *m, int e) {
    int s, per, i;
    for (s = 0; s < m->nslots; s++) {
        Slot *sl = &m->slots[s];
        if (sl->kind == K_TRUNK) continue;
        per = sl->n / MAX_EXPERTS;
        for (i = 0; i < per; i++) {
            sl->m[e * per + i] = 0.f;
            sl->v[e * per + i] = 0.f;
            sl->g[e * per + i] = 0.f;
        }
    }
}

static void forward(Model *m, const int *tok, int T, int training) {
    int n, t, d, e;
    unsigned char halted[TMAX];
    unsigned char alive[TMAX];
    m->curT = T;
    m->training = training;
    m->bal_loss = 0.f;
    m->loss = 0.f;
    {
        int lim = training ? T + 1 : T;
        for (t = 0; t < lim; t++) {
            int id = tok[t];
            if (id < 0 || id >= VOCAB) id = 0;
            m->ids[t] = id;
        }
    }
    for (e = 0; e < m->n_experts; e++)
        m->recent_view[e] = m->recent[e] * (1.f - 1.f / USAGE_STEPS);
    for (e = 0; e < MAX_EXPERTS; e++) m->vote[e] = 0.f;
    for (t = 0; t < T; t++) {
        int id = tok[t];
        if (id < 0 || id >= VOCAB) id = 0;
        for (d = 0; d < D; d++)
            m->tokemb[t * D + d] = m->embed[id * D + d];
        halted[t] = 0;
    }
    dense_fwd(&m->pre[0], m->pa[0], m->tokemb, T);
    dense_fwd(&m->pre[1], m->pa[1], m->pa[0]->y, T);
    for (t = 0; t < T; t++)
        for (d = 0; d < D; d++)
            m->stream[t * D + d] = m->pa[1]->y[t * D + d];
    for (t = 0; t < T; t++) m->cs[t] = 1.f;
    for (n = 0; n < MAX_STEPS; n++) {
        float prior_stub;
        for (t = 0; t < T; t++) {
            alive[t] = (unsigned char)!halted[t];
            m->active[n][t] = alive[t];
            for (d = 0; d < D; d++) {
                if (n == 0) m->hbefore[n][t * D + d] = 0.f;
            }
        }
        if (n > 0) {
            for (t = 0; t < T; t++)
                for (d = 0; d < D; d++)
                    m->hbefore[n][t * D + d] = m->hafter[n - 1][t * D + d];
        }
        adapter_fwd(m->adapter, m->hbefore[n], m->stream, buf_a, T);
        recur_fwd(m, &m->rec, m->ra[n], buf_a, alive, T,
                  n == 0, n == 0 && training);
        for (t = 0; t < T; t++) {
            for (d = 0; d < D; d++) {
                float hv = alive[t] ? m->ra[n]->y[t * D + d]
                                    : m->hbefore[n][t * D + d];
                m->hafter[n][t * D + d] = hv;
            }
        }
        rmsnorm(m->lnf, m->hafter[n], m->ynorm[n], m->rstdf[n], T);
        linear(m->ynorm[n], m->embed, m->logits[n], T, D, VOCAB);
        for (t = 0; t < T; t++) {
            float raw = m->halt_b[0] + dot(m->halt_w, m->ynorm[n] + t * D, D);
            float lam;
            m->hraw[n][t] = raw;
            if (n == MAX_STEPS - 1) lam = 1.f;
            else if (n < MIN_STEPS - 1) lam = 0.f;
            else lam = bm_sigmoid(raw);
            m->lam[n][t] = lam;
            m->ps[n][t] = m->cs[n * TMAX + t] * lam;
            m->cs[(n + 1) * TMAX + t] = m->cs[n * TMAX + t] * (1.f - lam);
            if (!training) {
                /* filled after the loop's halt rule, stored in logits mix later */
            } else {
                int tgt = tok[t + 1];
                if (tgt < 0 || tgt >= VOCAB) tgt = 0;
                m->ce[n][t] = ce_one(m->logits[n] + t * VOCAB, tgt, 0, 0.f);
            }
            if ((1.f - m->cs[(n + 1) * TMAX + t]) >= HALT_THRESH)
                halted[t] = 1;
        }
        (void)prior_stub;
    }
    if (training) {
        float prior[MAX_STEPS], ps = 0.f, invT = 1.f / (float)T;
        float lsum = 0.f, ksum = 0.f, stepmean = 0.f;
        float keep = 1.f - HALT_PRIOR;
        prior[0] = HALT_PRIOR;
        ps = prior[0];
        for (n = 1; n < MAX_STEPS; n++) {
            prior[n] = prior[n - 1] * keep;
            ps += prior[n];
        }
        for (n = 0; n < MAX_STEPS; n++) prior[n] /= ps;
        for (t = 0; t < T; t++) {
            float kl = 0.f;
            for (n = 0; n < MAX_STEPS; n++) {
                float p = m->ps[n][t];
                float pc = p > 1e-8f ? p : 1e-8f;
                lsum += p * m->ce[n][t];
                kl += pc * (bm_log(pc) - bm_log(prior[n]));
                stepmean += p * (float)(n + 1);
            }
            ksum += kl;
        }
        m->loss = lsum * invT + PONDER_BETA * ksum * invT + m->bal_loss;
        m->last_steps = stepmean * invT;
    }
}

static void backward(Model *m) {
    int n, t, d, b;
    int T = m->curT;
    float invT = 1.f / (float)T;
    float prior[MAX_STEPS], psum = 0.f, keep = 1.f - HALT_PRIOR;
    float dc_next[TMAX];
    float drawv[MAX_STEPS][TMAX];

    prior[0] = HALT_PRIOR;
    psum = prior[0];
    for (n = 1; n < MAX_STEPS; n++) {
        prior[n] = prior[n - 1] * keep;
        psum += prior[n];
    }
    for (n = 0; n < MAX_STEPS; n++) prior[n] /= psum;
    for (b = 0; b < m->nslots; b++) zero(m->slots[b].g, m->slots[b].n);

    for (t = 0; t < T; t++) dc_next[t] = 0.f;
    for (n = MAX_STEPS - 1; n >= 0; n--) {
        float dc[TMAX];
        int forced = (n == MAX_STEPS - 1) || (n < MIN_STEPS - 1);
        for (t = 0; t < T; t++) {
            float p = m->ps[n][t];
            float c0 = m->cs[n * TMAX + t];
            float lam = m->lam[n][t];
            float pc = p > 1e-8f ? p : 1e-8f;
            float dpd = m->ce[n][t] * invT;
            float dlam;
            dpd += PONDER_BETA * (bm_log(pc) - bm_log(prior[n]) + 1.f) * invT;
            dlam = dpd * c0 - dc_next[t] * c0;
            dc[t] = dpd * lam + dc_next[t] * (1.f - lam);
            drawv[n][t] = forced ? 0.f : dlam * lam * (1.f - lam);
        }
        for (t = 0; t < T; t++) dc_next[t] = dc[t];
    }

    zero(buf_h2, T * D);
    zero(buf_h3, T * D);
    for (n = MAX_STEPS - 1; n >= 0; n--) {
        zero(buf_a, T * D);
        for (t = 0; t < T; t++) {
            float scale = m->ps[n][t] * invT;
            ce_one(m->logits[n] + t * VOCAB, m->ids[t + 1],
                   buf_big + t * VOCAB, scale);
        }
        linear_bwd(m->ynorm[n], m->embed, buf_big, buf_a, m->embed_g, T, D, VOCAB);
        for (t = 0; t < T; t++) {
            float draw = drawv[n][t];
            m->halt_bg[0] += draw;
            for (d = 0; d < D; d++) {
                m->halt_g[d] += draw * m->ynorm[n][t * D + d];
                buf_a[t * D + d] += draw * m->halt_w[d];
            }
        }
        zero(buf_b, T * D);
        rmsnorm_bwd(m->lnf, m->hafter[n], m->rstdf[n], buf_a, buf_b, m->lnfg, T);
        for (t = 0; t < T; t++)
            for (d = 0; d < D; d++) buf_b[t * D + d] += buf_h3[t * D + d];

        zero(buf_h0, T * D);
        zero(buf_c, T * D);
        for (t = 0; t < T; t++) {
            if (m->active[n][t]) {
                for (d = 0; d < D; d++) buf_h0[t * D + d] = buf_b[t * D + d];
            } else {
                for (d = 0; d < D; d++) buf_c[t * D + d] = buf_b[t * D + d];
            }
        }
        recur_bwd(m, &m->rec, m->ra[n], m->active[n], buf_h0, buf_hx, T);
        adapter_bwd(m->adapter, m->hbefore[n], m->stream, buf_hx,
                    buf_c, buf_h2, m->adapter_g, T);
        for (t = 0; t < T; t++)
            for (d = 0; d < D; d++) buf_h3[t * D + d] = buf_c[t * D + d];
    }

    dense_bwd(&m->pre[1], m->pa[1], buf_h2, buf_hx, T);
    dense_bwd(&m->pre[0], m->pa[0], buf_hx, buf_h2, T);
    for (t = 0; t < T; t++) {
        int id = m->ids[t];
        for (d = 0; d < D; d++)
            m->embed_g[id * D + d] += buf_h2[t * D + d];
    }
}

static void relax_rates(Model *m) {
    int e;
    float k = bm_exp(bm_log(0.5f) / 15000.f);
    for (e = 0; e < m->n_experts; e++) {
        if (m->rate[e] > 1.f) {
            float l = bm_log(m->rate[e]) * k;
            m->rate[e] = bm_exp(l);
            if (m->rate[e] < 1.0001f) m->rate[e] = 1.f;
        }
    }
}

static void adam(Model *m) {
    int s, g, i, ng, per;
    float norm2 = 0.f, scale, b1c, b2c, tt;
    m->step++;
    tt = (float)m->step;
    b1c = 1.f - bm_exp(tt * bm_log(ADAM_B1));
    b2c = 1.f - bm_exp(tt * bm_log(ADAM_B2));
    if (b1c < 1e-8f) b1c = 1e-8f;
    if (b2c < 1e-8f) b2c = 1e-8f;
    for (s = 0; s < m->nslots; s++) {
        Slot *sl = &m->slots[s];
        if (sl->kind == K_TRUNK) {
            ng = 1;
            per = sl->n;
        } else {
            per = sl->n / MAX_EXPERTS;
            ng = m->n_experts;
        }
        for (g = 0; g < ng; g++) {
            for (i = 0; i < per; i++) {
                float v = sl->g[g * per + i];
                norm2 += v * v;
            }
        }
    }
    scale = 1.f;
    if (norm2 > 0.f) {
        float nrm = bm_sqrt(norm2);
        if (nrm > CLIP) scale = CLIP / nrm;
    }
    for (s = 0; s < m->nslots; s++) {
        Slot *sl = &m->slots[s];
        if (sl->kind == K_TRUNK) {
            ng = 1;
            per = sl->n;
        } else {
            per = sl->n / MAX_EXPERTS;
            ng = m->n_experts;
        }
        for (g = 0; g < ng; g++) {
            float lr = LR_BASE * plast_factor(&m->plast) * sl->lr_mult;
            if (sl->kind != K_TRUNK) lr *= m->rate[g];
            for (i = 0; i < per; i++) {
                int o = g * per + i;
                float gr = sl->g[o] * scale;
                float mhat, vhat;
                sl->m[o] = ADAM_B1 * sl->m[o] + (1.f - ADAM_B1) * gr;
                sl->v[o] = ADAM_B2 * sl->v[o] + (1.f - ADAM_B2) * gr * gr;
                mhat = sl->m[o] / b1c;
                vhat = sl->v[o] / b2c;
                sl->w[o] -= lr * (mhat / (bm_sqrt(vhat) + ADAM_EPS) + WD * sl->w[o]);
            }
        }
    }
    relax_rates(m);
}

static void commit_stats(Model *m) {
    int e, i;
    for (e = 0; e < m->n_experts; e++) m->recent[e] = m->recent_view[e];
    for (i = 0; i < m->n_adm; i++) {
        int e2 = m->adm[i];
        if (e2 >= 0 && e2 < m->n_experts) m->last_seen[e2] = m->chars;
    }
}

float model_nll(Model *m, const int *tok, int n) {
    int T = n - 1;
    if (T < 1 || T > TMAX) return -1.f;
    forward(m, tok, T, 1);
    return m->loss;
}

float model_train(Model *m, const int *tok, int n) {
    int T = n - 1;
    float loss;
    if (T < 1 || T > TMAX) return -1.f;
    plast_tick(&m->plast);
    forward(m, tok, T, 1);
    loss = m->loss;
    backward(m);
    adam(m);
    m->chars += T;
    m->segments += 1;
    commit_stats(m);
    m->last_loss = loss;
    return loss;
}

float model_lesson(Model *m) {
    int n = TMAX + 1;
    float L;
    if (tr_n < n) return -1.f;
    if (m->cursor + n > tr_n) m->cursor = 0;
    L = model_train(m, tr_ids + m->cursor, n);
    m->cursor += TMAX;
    if (m->step > 0 && (m->step % 32) == 0) {
        m->last_held = model_heldout(m);
        m->have_held = 1;
        plast_observe(&m->plast, m->last_held);
    }
    return L;
}

float model_heldout(Model *m) {
    if (ho_n < TMAX + 1) return -1.f;
    return model_nll(m, ho_ids, TMAX + 1);
}

static int pick_next(const float *logits, const int *prev, int nprev) {
    float score[VOCAB];
    float tr[VOCAB];
    int i, best, n, a;
    for (i = 0; i < VOCAB; i++) score[i] = logits[i];
    n = nprev < 64 ? nprev : 64;
    if (n > 0) {
        zero(tr, VOCAB);
        for (a = 0; a < n; a++) {
            int id = prev[nprev - n + a];
            int age = n - 1 - a;
            float w = 1.f;
            int k;
            for (k = 0; k < age; k++) w *= ADAPT_DECAY;
            if (id >= 0 && id < VOCAB) tr[id] += w;
        }
        for (i = 0; i < VOCAB; i++) score[i] -= ADAPT_STRENGTH * tr[i];
    }
    best = 0;
    for (i = 1; i < VOCAB; i++) if (score[i] > score[best]) best = i;
    return best;
}

int model_generate(Model *m, const int *prompt, int n, int *out, int ngen) {
    int seq[CTX + 2];
    int len, g, i;
    if (n > CTX) {
        prompt += n - CTX;
        n = CTX;
    }
    if (n < 1) {
        seq[0] = ' ';
        len = 1;
    } else {
        for (i = 0; i < n; i++) seq[i] = prompt[i];
        len = n;
    }
    for (g = 0; g < ngen; g++) {
        int window = len > TMAX ? TMAX : len;
        int start = len - window;
        int nxt;
        forward(m, seq + start, window, 0);
        /* inference keeps the step-0 logits unless a later step halts.
           Recompute the halted choice here from the saved schedule. */
        {
            int t = window - 1;
            int chosen = 0;
            unsigned char halted = 0;
            for (i = 0; i < MAX_STEPS; i++) {
                int crossed = (1.f - m->cs[(i + 1) * TMAX + t]) >= HALT_THRESH;
                if (i == 0 || (!halted && crossed)) chosen = i;
                if (!halted && crossed) halted = 1;
            }
            nxt = pick_next(m->logits[chosen] + t * VOCAB, seq, len);
        }
        if (len < CTX) seq[len++] = nxt;
        else {
            for (i = 1; i < CTX; i++) seq[i - 1] = seq[i];
            seq[CTX - 1] = nxt;
            len = CTX;
        }
        if (g < ngen) out[g] = nxt;
    }
    return ngen;
}

static void swap_range(float *p, int per, int i, int j) {
    int k;
    for (k = 0; k < per; k++) {
        float tmp = p[i * per + k];
        p[i * per + k] = p[j * per + k];
        p[j * per + k] = tmp;
    }
}

static void swap_expert(Model *m, int i, int j) {
    int s, sl;
    if (i == j) return;
    for (sl = 0; sl < m->nslots; sl++) {
        Slot *S = &m->slots[sl];
        int per;
        if (S->kind == K_TRUNK) continue;
        per = S->n / MAX_EXPERTS;
        swap_range(S->w, per, i, j);
        swap_range(S->g, per, i, j);
        swap_range(S->m, per, i, j);
        swap_range(S->v, per, i, j);
    }
    {
        float tr = m->rate[i];
        float rc = m->recent[i];
        int ls = m->last_seen[i];
        int bc = m->born_chars[i];
        m->rate[i] = m->rate[j];
        m->recent[i] = m->recent[j];
        m->last_seen[i] = m->last_seen[j];
        m->born_chars[i] = m->born_chars[j];
        m->rate[j] = tr;
        m->recent[j] = rc;
        m->last_seen[j] = ls;
        m->born_chars[j] = bc;
    }
    for (s = 0; s < RESIDENT; s++) {
        if (m->slots_e[s] == i) m->slots_e[s] = -3;
        else if (m->slots_e[s] == j) m->slots_e[s] = -4;
    }
    for (s = 0; s < RESIDENT; s++) {
        if (m->slots_e[s] == -3) m->slots_e[s] = j;
        else if (m->slots_e[s] == -4) m->slots_e[s] = i;
    }
}

static void drop_last(Model *m) {
    int last, s;
    if (m->n_experts <= INIT_EXPERTS) return;
    last = m->n_experts - 1;
    for (s = 0; s < RESIDENT; s++)
        if (m->slots_e[s] == last) m->slots_e[s] = -1;
    m->n_experts--;
}

static int dying_count(const Model *m) {
    int e, n = 0;
    for (e = 0; e < m->n_experts; e++) {
        int age, life;
        if (e < INIT_EXPERTS && m->born_chars[e] == 0) continue;
        life = m->chars - m->born_chars[e];
        if (life < SURVIVAL_CHARS) continue;
        age = m->chars - m->last_seen[e];
        if (age > (int)(0.65f * (float)SURVIVAL_CHARS)) n++;
    }
    return n;
}

static int in_flight(const Model *m) {
    int e, n = 0;
    for (e = 0; e < m->n_experts; e++) {
        if (m->born_chars[e] <= 0) continue;
        if (m->chars - m->born_chars[e] < SURVIVAL_CHARS) n++;
    }
    return n;
}

static int copy_why(char *why, int n, const char *s) {
    int i = 0;
    if (!why || n <= 0) return 0;
    while (s[i] && i + 1 < n) {
        why[i] = s[i];
        i++;
    }
    why[i] = 0;
    return i;
}

static void birth(Model *m) {
    int e = m->n_experts;
    int npar = m->n_experts >= 3 ? 3 : m->n_experts;
    int par[3];
    int counts[3];
    int u, d, pi;
    float inv = 1.f / (float)EXPERT_FF;
    par[0] = 0;
    par[1] = 1 % m->n_experts;
    par[2] = 2 % m->n_experts;
    if (m->n_adm > 0) par[0] = m->adm[0];
    counts[0] = counts[1] = counts[2] = 0;
    for (u = 0; u < EXPERT_FF; u++) {
        int src;
        pi = (int)(rndu(m) * (float)npar);
        if (pi >= npar) pi = npar - 1;
        if (pi < 0) pi = 0;
        src = par[pi];
        counts[pi]++;
        for (d = 0; d < D; d++) {
            m->ew1[e_off_ff(e, u) + d] = m->ew1[e_off_ff(src, u) + d];
            m->ew3[e_off_ff(e, u) + d] = m->ew3[e_off_ff(src, u) + d];
            m->ew2[e_off_w2(e, d) + u] = m->ew2[e_off_w2(src, d) + u];
        }
    }
    for (d = 0; d < D; d++) m->router[e * D + d] = 0.f;
    for (pi = 0; pi < npar; pi++) {
        float sh = (float)counts[pi] * inv;
        int src = par[pi];
        for (d = 0; d < D; d++)
            m->router[e * D + d] += sh * m->router[src * D + d];
    }
    m->gate[e] = BIRTH_GATE;
    m->rate[e] = 1.f / (m->plast.scale > 1e-6f ? m->plast.scale : 1e-6f);
    if (m->rate[e] < 1.f) m->rate[e] = 1.f;
    m->born_chars[e] = m->chars > 0 ? m->chars : 1;
    m->last_seen[e] = m->chars;
    m->recent[e] = 0.f;
    zero_expert_mom(m, e);
    m->n_experts++;
}

int model_grow(Model *m, int force, char *why, int whylen) {
    int dying, flight;
    float gap;
    if (m->n_experts >= MAX_EXPERTS) {
        copy_why(why, whylen, "no room: the expert table is full");
        return 0;
    }
    if (!force) {
        if (m->chars - m->grow_at < GROW_EVERY) {
            copy_why(why, whylen, "not yet: growth waits on the character clock");
            return 0;
        }
        dying = dying_count(m);
        if ((float)dying / (float)m->n_experts > 0.35f) {
            copy_why(why, whylen, "capacity already added is not being asked for");
            return 0;
        }
        flight = in_flight(m);
        if (flight + 1 > 50) {
            copy_why(why, whylen, "too many experts are still inside their trial");
            return 0;
        }
        if (m->have_held && (m->last_held - m->last_loss) >= MAX_GAP) {
            gap = m->last_held - m->last_loss;
            (void)gap;
            copy_why(why, whylen, "train and held-out have separated");
            return 0;
        }
    }
    birth(m);
    m->grow_at = m->chars;
    copy_why(why, whylen, "grew: room, and the pool still has a use for another expert");
    return 1;
}

int model_prune(Model *m, int force) {
    int e, removed = 0;
    if (force) {
        if (m->n_experts > INIT_EXPERTS) {
            drop_last(m);
            return 1;
        }
        return 0;
    }
    for (e = m->n_experts - 1; e >= INIT_EXPERTS; e--) {
        int life = m->chars - m->born_chars[e];
        int age = m->chars - m->last_seen[e];
        if (life > SURVIVAL_CHARS && age > SURVIVAL_CHARS) {
            swap_expert(m, e, m->n_experts - 1);
            drop_last(m);
            removed++;
        }
    }
    return removed;
}

static int app(char *b, int n, int u, const char *s) {
    while (*s && u + 1 < n) b[u++] = *s++;
    if (u < n) b[u] = 0;
    return u;
}

static int app_i(char *b, int n, int u, int v) {
    char tmp[16];
    int k = 0;
    if (v < 0) {
        u = app(b, n, u, "-");
        v = -v;
    }
    if (v == 0) tmp[k++] = '0';
    while (v && k < 16) {
        tmp[k++] = (char)('0' + (v % 10));
        v /= 10;
    }
    while (k > 0 && u + 1 < n) b[u++] = tmp[--k];
    if (u < n) b[u] = 0;
    return u;
}

static int app_f(char *b, int n, int u, float v, int prec) {
    int ip, i, dig;
    if (!(v == v)) return app(b, n, u, "nan");
    if (v < 0.f) {
        u = app(b, n, u, "-");
        v = -v;
    }
    ip = (int)v;
    u = app_i(b, n, u, ip);
    u = app(b, n, u, ".");
    v -= (float)ip;
    for (i = 0; i < prec; i++) {
        char s[2];
        v *= 10.f;
        dig = (int)v;
        if (dig < 0) dig = 0;
        if (dig > 9) dig = 9;
        s[0] = (char)('0' + dig);
        s[1] = 0;
        u = app(b, n, u, s);
        v -= (float)dig;
    }
    return u;
}

void model_stat(const Model *m, char *buf, int n) {
    int u = 0, s, params = 0, res = 0;
    for (s = 0; s < m->nslots; s++) params += m->slots[s].n;
    for (s = 0; s < RESIDENT; s++) if (m->slots_e[s] >= 0) res++;
    u = app(buf, n, u, "step ");
    u = app_i(buf, n, u, m->step);
    u = app(buf, n, u, "  chars ");
    u = app_i(buf, n, u, m->chars);
    u = app(buf, n, u, "  loss ");
    u = app_f(buf, n, u, m->last_loss, 4);
    u = app(buf, n, u, "  held ");
    if (m->have_held) u = app_f(buf, n, u, m->last_held, 4);
    else u = app(buf, n, u, "-");
    u = app(buf, n, u, "  depth ");
    u = app_f(buf, n, u, m->last_steps, 2);
    u = app(buf, n, u, "  experts ");
    u = app_i(buf, n, u, m->n_experts);
    u = app(buf, n, u, "/");
    u = app_i(buf, n, u, res);
    u = app(buf, n, u, "  params ");
    u = app_i(buf, n, u, params);
    u = app(buf, n, u, "  lr x");
    u = app_f(buf, n, u, plast_factor(&m->plast), 4);
}

static uint32_t blob_floats(const Model *m) {
    uint32_t n = 0;
    int s;
    for (s = 0; s < m->nslots; s++) n += (uint32_t)m->slots[s].n * 3u;
    return n;
}

int model_sector_count(const Model *m) {
    uint32_t bytes = blob_floats(m) * 4u;
    return 1 + (int)((bytes + 511u) / 512u);
}

static int flat_at(const Model *m, uint32_t fi, float **out, int write) {
    int s;
    for (s = 0; s < m->nslots; s++) {
        const Slot *sl = &m->slots[s];
        uint32_t span = (uint32_t)sl->n;
        float *bases[3];
        int k;
        bases[0] = sl->w;
        bases[1] = sl->m;
        bases[2] = sl->v;
        for (k = 0; k < 3; k++) {
            if (fi < span) {
                *out = bases[k] + fi;
                (void)write;
                return 1;
            }
            fi -= span;
        }
    }
    return 0;
}

void model_export_sector(const Model *m, unsigned index, void *dst512) {
    unsigned char *dst = dst512;
    int i;
    if (index == 0) {
        unsigned char *p = dst;
        uint32_t words[8];
        for (i = 0; i < 512; i++) dst[i] = 0;
        p[0] = 'M'; p[1] = 'A'; p[2] = 'G'; p[3] = 'I';
        words[0] = 1;
        words[1] = (uint32_t)m->nslots;
        words[2] = (uint32_t)m->step;
        words[3] = (uint32_t)m->n_experts;
        words[4] = (uint32_t)m->chars;
        words[5] = (uint32_t)m->cursor;
        words[6] = (uint32_t)m->segments;
        words[7] = blob_floats(m);
        for (i = 0; i < 8; i++) {
            unsigned v = words[i];
            p[4 + i * 4 + 0] = (unsigned char)(v & 255);
            p[4 + i * 4 + 1] = (unsigned char)((v >> 8) & 255);
            p[4 + i * 4 + 2] = (unsigned char)((v >> 16) & 255);
            p[4 + i * 4 + 3] = (unsigned char)((v >> 24) & 255);
        }
        {
            union { float f; unsigned u; } u;
            u.f = m->plast.scale;
            p[36] = (unsigned char)(u.u & 255);
            p[37] = (unsigned char)((u.u >> 8) & 255);
            p[38] = (unsigned char)((u.u >> 16) & 255);
            p[39] = (unsigned char)((u.u >> 24) & 255);
        }
        for (i = 0; i < RESIDENT; i++) {
            int v = m->slots_e[i];
            p[40 + i] = (unsigned char)(v < 0 ? 255 : v);
        }
        return;
    }
    {
        uint32_t base = (index - 1u) * 128u;
        for (i = 0; i < 128; i++) {
            float *fp = 0;
            union { float f; unsigned u; } u;
            u.f = 0.f;
            if (flat_at(m, base + (uint32_t)i, &fp, 0)) u.f = *fp;
            dst[i * 4 + 0] = (unsigned char)(u.u & 255);
            dst[i * 4 + 1] = (unsigned char)((u.u >> 8) & 255);
            dst[i * 4 + 2] = (unsigned char)((u.u >> 16) & 255);
            dst[i * 4 + 3] = (unsigned char)((u.u >> 24) & 255);
        }
    }
}

int model_import_sector(Model *m, unsigned index, const void *src512) {
    const unsigned char *src = src512;
    int i;
    if (index == 0) {
        uint32_t ver, nslots, step, nexp, chars, cursor, segs;
        if (src[0] != 'M' || src[1] != 'A' || src[2] != 'G' || src[3] != 'I')
            return 0;
        ver = (uint32_t)src[4] | ((uint32_t)src[5] << 8) |
              ((uint32_t)src[6] << 16) | ((uint32_t)src[7] << 24);
        nslots = (uint32_t)src[8] | ((uint32_t)src[9] << 8) |
                 ((uint32_t)src[10] << 16) | ((uint32_t)src[11] << 24);
        if (ver != 1 || (int)nslots != m->nslots) return 0;
        step = (uint32_t)src[12] | ((uint32_t)src[13] << 8) |
               ((uint32_t)src[14] << 16) | ((uint32_t)src[15] << 24);
        nexp = (uint32_t)src[16] | ((uint32_t)src[17] << 8) |
               ((uint32_t)src[18] << 16) | ((uint32_t)src[19] << 24);
        chars = (uint32_t)src[20] | ((uint32_t)src[21] << 8) |
                ((uint32_t)src[22] << 16) | ((uint32_t)src[23] << 24);
        cursor = (uint32_t)src[24] | ((uint32_t)src[25] << 8) |
                 ((uint32_t)src[26] << 16) | ((uint32_t)src[27] << 24);
        segs = (uint32_t)src[28] | ((uint32_t)src[29] << 8) |
               ((uint32_t)src[30] << 16) | ((uint32_t)src[31] << 24);
        m->step = (int)step;
        m->n_experts = (int)nexp;
        m->chars = (int)chars;
        m->cursor = (int)cursor;
        m->segments = (int)segs;
        m->plast.step = (int)step;
        {
            union { float f; unsigned u; } u;
            u.u = (unsigned)src[36] | ((unsigned)src[37] << 8) |
                  ((unsigned)src[38] << 16) | ((unsigned)src[39] << 24);
            m->plast.scale = u.f;
        }
        for (i = 0; i < RESIDENT; i++) {
            int v = src[40 + i];
            m->slots_e[i] = (v == 255) ? -1 : v;
        }
        return 1;
    }
    {
        uint32_t base = (index - 1u) * 128u;
        for (i = 0; i < 128; i++) {
            float *fp = 0;
            union { float f; unsigned u; } u;
            u.u = (unsigned)src[i * 4] | ((unsigned)src[i * 4 + 1] << 8) |
                  ((unsigned)src[i * 4 + 2] << 16) | ((unsigned)src[i * 4 + 3] << 24);
            if (flat_at(m, base + (uint32_t)i, &fp, 1)) *fp = u.f;
        }
    }
    return 1;
}

static float *alloc_f(int n) {
    return kalloc((size_t)n * sizeof(float));
}

Model *model_create(uint32_t seed) {
    Model *m;
    int i, e, d;
    float std_out;
    if (!mem_ready) {
        mem_init();
        mem_ready = 1;
    }
    mem_reset();
    g_freeze = 0;
    g_soft = 0;
    m = kalloc(sizeof(Model));
    m->rng = seed ? seed : 1u;
    plast_init(&m->plast);
    build_rope();
    std_out = 0.02f / bm_sqrt(2.f * (float)(N_PRELUDE + 1));

    m->embed = param(m, VOCAB * D, K_TRUNK, TRUNK_MULT);
    m->embed_g = m->slots[m->nslots - 1].g;
    fill_n(m, m->embed, VOCAB * D, 0.02f);

    bind_dense(m, &m->pre[0], 0);
    bind_dense(m, &m->pre[1], 0);
    bind_dense(m, &m->rec, 1);

    m->router = param(m, MAX_EXPERTS * D, K_ROW, 1.f);
    m->router_g = m->slots[m->nslots - 1].g;
    fill_n(m, m->router, INIT_EXPERTS * D, 0.01f);
    m->depth = param(m, D, K_TRUNK, TRUNK_MULT);
    m->depth_g = m->slots[m->nslots - 1].g;
    m->gate = param(m, MAX_EXPERTS, K_ROW, 1.f);
    m->gate_g = m->slots[m->nslots - 1].g;
    for (e = 0; e < INIT_EXPERTS; e++) m->gate[e] = 1.f;

    m->ew1 = param(m, MAX_EXPERTS * EXPERT_FF * D, K_EXPERT, 1.f);
    m->ew1g = m->slots[m->nslots - 1].g;
    m->ew3 = param(m, MAX_EXPERTS * EXPERT_FF * D, K_EXPERT, 1.f);
    m->ew3g = m->slots[m->nslots - 1].g;
    m->ew2 = param(m, MAX_EXPERTS * D * EXPERT_FF, K_EXPERT, 1.f);
    m->ew2g = m->slots[m->nslots - 1].g;
    for (e = 0; e < INIT_EXPERTS; e++) {
        fill_n(m, m->ew1 + e * EXPERT_FF * D, EXPERT_FF * D, 0.02f);
        fill_n(m, m->ew3 + e * EXPERT_FF * D, EXPERT_FF * D, 0.02f);
        fill_n(m, m->ew2 + e * D * EXPERT_FF, D * EXPERT_FF, std_out);
    }

    m->adapter = param(m, D * 2 * D, K_TRUNK, TRUNK_MULT);
    m->adapter_g = m->slots[m->nslots - 1].g;
    for (d = 0; d < D; d++) {
        m->adapter[d * (2 * D) + d] = 1.f;
        m->adapter[d * (2 * D) + D + d] = 1.f;
    }
    m->lnf = param(m, D, K_TRUNK, TRUNK_MULT);
    m->lnfg = m->slots[m->nslots - 1].g;
    for (d = 0; d < D; d++) m->lnf[d] = 1.f;
    m->halt_w = param(m, D, K_TRUNK, TRUNK_MULT);
    m->halt_g = m->slots[m->nslots - 1].g;
    fill_n(m, m->halt_w, D, 0.02f);
    for (d = 0; d < D; d++) m->halt_w[d] *= 0.01f;
    m->halt_b = param(m, 1, K_TRUNK, TRUNK_MULT);
    m->halt_bg = m->slots[m->nslots - 1].g;
    m->halt_b[0] = -2.f;

    m->n_experts = INIT_EXPERTS;
    for (i = 0; i < RESIDENT; i++) m->slots_e[i] = -1;
    for (e = 0; e < MAX_EXPERTS; e++) m->rate[e] = 1.f;

    buf_qkv = alloc_f(TMAX * 3 * D);
    buf_a = alloc_f(TMAX * D_FF);
    buf_b = alloc_f(TMAX * D_FF);
    buf_c = alloc_f(TMAX * D_FF);
    buf_ff = alloc_f(TMAX * D_FF);
    buf_f1 = alloc_f(TMAX * D_FF);
    buf_f2 = alloc_f(TMAX * D_FF);
    buf_h0 = alloc_f(TMAX * D);
    buf_h1 = alloc_f(TMAX * D);
    buf_h2 = alloc_f(TMAX * D);
    buf_h3 = alloc_f(TMAX * D);
    buf_hx = alloc_f(TMAX * D);
    buf_big = alloc_f(TMAX * VOCAB);
    scr_q = alloc_f(N_HEAD * TMAX * HEAD_DIM);
    scr_k = alloc_f(N_HEAD * TMAX * HEAD_DIM);
    scr_v = alloc_f(N_HEAD * TMAX * HEAD_DIM);

    for (i = 0; i < N_PRELUDE; i++) m->pa[i] = kalloc(sizeof(DenseAct));
    for (i = 0; i < MAX_STEPS; i++) {
        m->ra[i] = kalloc(sizeof(RecurAct));
        m->hbefore[i] = alloc_f(TMAX * D);
        m->hafter[i] = alloc_f(TMAX * D);
        m->logits[i] = alloc_f(TMAX * VOCAB);
        m->ynorm[i] = alloc_f(TMAX * D);
        m->rstdf[i] = alloc_f(TMAX);
        m->lam[i] = alloc_f(TMAX);
        m->ps[i] = alloc_f(TMAX);
        m->ce[i] = alloc_f(TMAX);
        m->hraw[i] = alloc_f(TMAX);
        m->active[i] = kalloc(TMAX);
    }
    m->cs = alloc_f((MAX_STEPS + 1) * TMAX);
    m->tokemb = alloc_f(TMAX * D);
    m->stream = alloc_f(TMAX * D);

    tr_n = tok_encode(TRAIN_TEXT, tr_ids, 640);
    ho_n = tok_encode(HELD_TEXT, ho_ids, 160);
    return m;
}

#ifdef HOST
#include <stdio.h>

static int fail(const char *msg) {
    fprintf(stderr, "FAIL %s\n", msg);
    return 1;
}

static int check_math(void) {
    float e = bm_exp(1.f);
    float l = bm_log(2.f);
    float s = bm_sin(1.57079632679f);
    float n = plast_nudge(2.2f);
    if (bm_fabs(e - 2.7182818f) > 2e-5f) return fail("exp");
    if (bm_fabs(l - 0.693147f) > 2e-5f) return fail("log");
    if (bm_fabs(s - 1.f) > 2e-5f) return fail("sin");
    if (bm_fabs(n - 1.f) > 1e-5f) return fail("nudge");
    return 0;
}

static int fd_bad(float analytic, float plus, float minus, float eps) {
    float fd = (plus - minus) / (2.f * eps);
    float err = bm_fabs(analytic - fd);
    float den = bm_fabs(fd) > bm_fabs(analytic) ? bm_fabs(fd) : bm_fabs(analytic);
    /* A gradient under about 1e-3 moves the loss by less than a float32 ulp
       across a 1e-3 probe, so only the sign and the rough size are meaningful. */
    if (den < 1e-3f) return err > 5e-4f;
    return err / den > 0.25f;
}

int model_selftest(void) {
    Model *m;
    float first, last, g, w, eps, Lp, Lm;
    int i, bad = 0;
    char why[96];
    char line[160];
    int snap_slots[RESIDENT];
    if (check_math()) return 1;
    m = model_create(1);
    printf("corpus train %d held %d\n", tr_n, ho_n);
    if (tr_n < TMAX + 1 || ho_n < TMAX + 1) return fail("corpus short");
    first = model_nll(m, tr_ids, TMAX + 1);
    printf("init nll %.4f\n", first);
    if (!(first > 4.5f && first < 7.5f)) return fail("init nll");

    g_freeze = 1;
    g_soft = 1;
    for (i = 0; i < RESIDENT; i++) m->slots_e[i] = i;
    for (i = 0; i < RESIDENT; i++) snap_slots[i] = m->slots_e[i];
    forward(m, tr_ids, TMAX, 1);
    backward(m);
    g = m->halt_bg[0];
    w = m->halt_b[0];
    eps = 1e-3f;
    m->halt_b[0] = w + eps;
    Lp = model_nll(m, tr_ids, TMAX + 1);
    for (i = 0; i < RESIDENT; i++) m->slots_e[i] = snap_slots[i];
    m->halt_b[0] = w - eps;
    Lm = model_nll(m, tr_ids, TMAX + 1);
    for (i = 0; i < RESIDENT; i++) m->slots_e[i] = snap_slots[i];
    m->halt_b[0] = w;
    printf("halt_b analytic %.6g fd %.6g\n", g, (Lp - Lm) / (2.f * eps));
    if (fd_bad(g, Lp, Lm, eps)) bad = 1;

    g = m->embed_g[7];
    w = m->embed[7];
    m->embed[7] = w + eps;
    Lp = model_nll(m, tr_ids, TMAX + 1);
    for (i = 0; i < RESIDENT; i++) m->slots_e[i] = snap_slots[i];
    m->embed[7] = w - eps;
    Lm = model_nll(m, tr_ids, TMAX + 1);
    for (i = 0; i < RESIDENT; i++) m->slots_e[i] = snap_slots[i];
    m->embed[7] = w;
    printf("embed analytic %.6g fd %.6g\n", g, (Lp - Lm) / (2.f * eps));
    if (fd_bad(g, Lp, Lm, eps)) bad = 1;

    g = m->ew1g[0];
    w = m->ew1[0];
    m->ew1[0] = w + eps;
    Lp = model_nll(m, tr_ids, TMAX + 1);
    for (i = 0; i < RESIDENT; i++) m->slots_e[i] = snap_slots[i];
    m->ew1[0] = w - eps;
    Lm = model_nll(m, tr_ids, TMAX + 1);
    for (i = 0; i < RESIDENT; i++) m->slots_e[i] = snap_slots[i];
    m->ew1[0] = w;
    printf("expert analytic %.6g fd %.6g\n", g, (Lp - Lm) / (2.f * eps));
    if (fd_bad(g, Lp, Lm, eps)) bad = 1;

    g = m->adapter_g[0];
    w = m->adapter[0];
    m->adapter[0] = w + eps;
    Lp = model_nll(m, tr_ids, TMAX + 1);
    for (i = 0; i < RESIDENT; i++) m->slots_e[i] = snap_slots[i];
    m->adapter[0] = w - eps;
    Lm = model_nll(m, tr_ids, TMAX + 1);
    for (i = 0; i < RESIDENT; i++) m->slots_e[i] = snap_slots[i];
    m->adapter[0] = w;
    printf("adapter analytic %.6g fd %.6g\n", g, (Lp - Lm) / (2.f * eps));
    if (fd_bad(g, Lp, Lm, eps)) bad = 1;
    if (bad) return fail("gradient");

    g_freeze = 0;
    g_soft = 0;
    m = model_create(1);
    first = model_train(m, tr_ids, TMAX + 1);
    last = first;
    for (i = 1; i < 80; i++) {
        int off = (i * TMAX) % (tr_n - TMAX - 1);
        last = model_train(m, tr_ids + off, TMAX + 1);
        if ((i % 20) == 0) printf("step %d loss %.4f\n", m->step, last);
    }
    printf("train %.4f -> %.4f\n", first, last);
    if (!(last < first - 0.05f)) return fail("loss did not fall");

    {
        int p[4];
        int o1[16], o2[16];
        p[0] = 't'; p[1] = 'h'; p[2] = 'e'; p[3] = ' ';
        model_generate(m, p, 4, o1, 16);
        model_generate(m, p, 4, o2, 16);
        for (i = 0; i < 16; i++) if (o1[i] != o2[i]) return fail("generate");
    }
    if (!model_grow(m, 1, why, 96)) return fail("grow");
    printf("grow: %s (%d experts)\n", why, m->n_experts);
    if (m->n_experts != INIT_EXPERTS + 1) return fail("grow count");
    if (model_prune(m, 1) != 1) return fail("prune");
    if (m->n_experts != INIT_EXPERTS) return fail("prune count");
    if (!model_grow(m, 0, why, 96)) {
        printf("grow brake: %s\n", why);
    } else return fail("brake should hold");

    {
        int ns = model_sector_count(m);
        unsigned char *img = kalloc((size_t)ns * 512);
        float before = model_nll(m, tr_ids, TMAX + 1);
        for (i = 0; i < ns; i++) model_export_sector(m, (unsigned)i, img + i * 512);
        m->embed[0] += 1.f;
        for (i = 0; i < ns; i++) {
            if (!model_import_sector(m, (unsigned)i, img + i * 512))
                return fail("import");
        }
        last = model_nll(m, tr_ids, TMAX + 1);
        printf("store %.4f -> %.4f\n", before, last);
        if (bm_fabs(before - last) > 1e-4f) return fail("store");
    }
    model_stat(m, line, 160);
    printf("%s\n", line);
    printf("ok\n");
    return 0;
}
#endif
