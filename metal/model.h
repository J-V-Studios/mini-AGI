#ifndef METAL_MODEL_H
#define METAL_MODEL_H

#include <stddef.h>
#include <stdint.h>

/* Same machine as minagi/, at a width that fits in the machine's own RAM
   with the optimiser state beside the weights. config.yaml names the full
   model; these are the same fields. */

enum {
    VOCAB = 265,
    D = 32,
    N_HEAD = 4,
    HEAD_DIM = 8,          /* D / N_HEAD */
    D_FF = 80,
    N_PRELUDE = 2,
    MAX_STEPS = 4,
    MIN_STEPS = 1,
    MAX_EXPERTS = 12,
    INIT_EXPERTS = 8,
    EXPERT_FF = 32,
    TOP_K = 2,
    RESIDENT = 4,
    CTX = 64,
    TMAX = 16
};

typedef struct Model Model;

Model *model_create(uint32_t seed);
float model_train(Model *m, const int *tok, int n);
float model_nll(Model *m, const int *tok, int n);
float model_lesson(Model *m);
float model_heldout(Model *m);
int model_generate(Model *m, const int *prompt, int n, int *out, int ngen);

/* 0 = a brake held. force skips the brakes and still refuses a full pool.
   why receives a short reason. */
int model_grow(Model *m, int force, char *why, int whylen);
int model_prune(Model *m, int force);
void model_stat(const Model *m, char *buf, int n);

int tok_encode(const char *s, int *out, int cap);
int tok_decode(const int *ids, int n, char *out, int cap);

/* Round-trip the weights, Adam moments and the pool clocks through a flat
   image. sector 0 is the header. Returns 0 on a bad header. */
int model_sector_count(const Model *m);
void model_export_sector(const Model *m, unsigned index, void *dst512);
int model_import_sector(Model *m, unsigned index, const void *src512);

#ifdef HOST
int model_selftest(void);
#endif

#endif
