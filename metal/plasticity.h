#ifndef METAL_PLASTICITY_H
#define METAL_PLASTICITY_H

/* The learning-rate governor from minagi/plasticity.py.
   Two exponentially weighted fits of held-out against time. The rate eases
   down unless held-out is clearly still falling, and never moves more than
   two percent in one evaluation. */

typedef struct Plasticity {
    float w, x, y, xx, xy, yy, w2;       /* slow fit, lam 0.97 */
    float fw, fx, fy, fxx, fxy, fyy, fw2; /* fast fit, lam 0.85 */
    float i;
    float scale;
    float last_t;
    float last_e;
    int step;
    int seen_at;
} Plasticity;

void plast_init(Plasticity *p);
float plast_nudge(float t);
float plast_factor(const Plasticity *p);
/* One held-out reading. Returns the new scale. */
float plast_observe(Plasticity *p, float heldout);
void plast_tick(Plasticity *p);

#endif
