/* Qwen1.5-MoE backend scaffold for colibri.
 *
 * Milestone 0: model-specific config parsing and routed-expert identity
 * conventions. The full resident forward path will follow the OLMoE backend's
 * shape, but Qwen uses its own tensor names, tokenizer/config, and optional
 * shared experts.
 */
#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif
#include "speedax_forecast.h"
#include "st.h"

typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim;
    int n_experts, topk, inter, shared_inter, vocab;
    float theta, eps;
    int norm_topk;
} QwenCfg;

typedef struct {
    int dense_missing;
    int expert_missing;
    int expert_scale_missing;
    int expert_tensors;
    int64_t dense_bytes;
    int64_t expert_bytes;
    int64_t scale_bytes;
} QwenLayoutStats;

typedef struct {
    float *in_ln, *post_ln;
    float *q, *k, *v, *o;
    float *qb, *kb, *vb;
    float *gate;
    float *shared_gate, *shared_up, *shared_down, *shared_gate_score;
} QwenLayer;

typedef struct {
    int eid;
    int8_t *g, *u, *d;
    float *gs, *us, *ds;
    uint64_t used;
} QwenSlot;

typedef struct {
    QwenSlot *slots;
    int n, cap;
} QwenLayerCache;

typedef struct {
    QwenCfg c;
    shards S;
    float *embed, *lm_head, *final_norm;
    QwenLayer *L;
    QwenLayerCache *cache;
    uint64_t clock, hits, miss;
    uint64_t prefetch_calls, forecast_items, forecast_skipped_cached, prefetch_enqueued, prefetch_dropped;
    float **K, **V;
    int max_t;
    double dense_load_s;
} QwenModel;

static int g_speedax_prefetch = 0;
static int g_speedax_pilot = 0;
static int g_speedax_pilot_k = 0;

typedef struct {
    int layer, expert;
} QwenPrefetchJob;

typedef struct {
    QwenModel *m;
    QwenPrefetchJob q[4096];
    unsigned r, w;
    int started;
    pthread_t th;
    pthread_mutex_t mx;
    pthread_cond_t cv;
} QwenPrefetchQueue;

static QwenPrefetchQueue g_qwen_pf;

static int json_int_default(jval *root, const char *key, int fallback) {
    jval *v = json_get(root, key);
    return v ? (int)v->num : fallback;
}

static float json_float_default(jval *root, const char *key, float fallback) {
    jval *v = json_get(root, key);
    return v ? (float)v->num : fallback;
}

static int json_bool_default(jval *root, const char *key, int fallback) {
    jval *v = json_get(root, key);
    return (v && v->t == J_BOOL) ? v->boolean : fallback;
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double rss_gb(void) {
#if defined(__APPLE__) || defined(__linux__)
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / (1024.0 * 1024.0);
#else
    return 0.0;
#endif
}

static float *falloc(int64_t n) {
    float *p = malloc((size_t)n * sizeof(float));
    if (!p) { fprintf(stderr, "OOM allocating %lld floats\n", (long long)n); exit(1); }
    return p;
}

static int8_t *ialloc(int64_t n) {
    int8_t *p = malloc((size_t)n);
    if (!p) { fprintf(stderr, "OOM allocating %lld bytes\n", (long long)n); exit(1); }
    return p;
}

static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

static void add_bias(float *y, const float *b, int S, int O) {
    if (!b) return;
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) y[(int64_t)s * O + o] += b[o];
}

static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0;
    for (int i = 0; i < D; i++) ms += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f;
    for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0.f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static float silu(float x) {
    return x / (1.f + expf(-x));
}

static float sigmoidf_q(float x) {
    return 1.f / (1.f + expf(-x));
}

static void qwen_load_cfg(QwenCfg *c, const char *snap) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    if (!buf) { fprintf(stderr, "OOM reading %s\n", path); exit(1); }
    if (fread(buf, 1, n, f) != (size_t)n) {}
    buf[n] = 0;
    fclose(f);

    char *arena = NULL;
    jval *root = json_parse(buf, &arena);
    c->hidden = json_int_default(root, "hidden_size", 0);
    c->n_layers = json_int_default(root, "num_hidden_layers", 0);
    c->n_heads = json_int_default(root, "num_attention_heads", 0);
    c->n_kv_heads = json_int_default(root, "num_key_value_heads", c->n_heads);
    c->n_experts = json_int_default(root, "num_experts", 0);
    if (!c->n_experts) c->n_experts = json_int_default(root, "num_experts_per_layer", 0);
    c->topk = json_int_default(root, "num_experts_per_tok", 0);
    c->inter = json_int_default(root, "moe_intermediate_size", 0);
    if (!c->inter) c->inter = json_int_default(root, "intermediate_size", 0);
    c->shared_inter = json_int_default(root, "shared_expert_intermediate_size", 0);
    c->vocab = json_int_default(root, "vocab_size", 0);
    c->head_dim = json_int_default(root, "head_dim", c->n_heads ? c->hidden / c->n_heads : 0);
    c->theta = json_float_default(root, "rope_theta", 1000000.0f);
    c->eps = json_float_default(root, "rms_norm_eps", 1e-6f);
    c->norm_topk = json_bool_default(root, "norm_topk_prob", 0);

    free(buf);
    free(arena);
}

static int qwen_validate_cfg(const QwenCfg *c) {
    return c->hidden > 0 && c->n_layers > 0 && c->n_heads > 0 &&
           c->n_kv_heads > 0 && c->n_experts > 0 && c->topk > 0 &&
           c->inter > 0 && c->vocab > 0 && c->head_dim > 0 &&
           c->n_heads % c->n_kv_heads == 0;
}

static void qwen_print_cfg(const QwenCfg *c) {
    printf("== Qwen MoE backend scaffold ==\n");
    printf("layers=%d hidden=%d heads=%d kv_heads=%d head_dim=%d vocab=%d\n",
           c->n_layers, c->hidden, c->n_heads, c->n_kv_heads, c->head_dim, c->vocab);
    printf("experts/layer=%d topk=%d moe_intermediate=%d shared_intermediate=%d norm_topk=%d\n",
           c->n_experts, c->topk, c->inter, c->shared_inter, c->norm_topk);
    printf("expert object key: (layer_id, expert_id); tensor names use "
           "model.layers.<layer>.mlp.experts.<expert>.*.weight\n");
}

static int qwen_require(shards *S, const char *name, const char *kind, QwenLayoutStats *st) {
    int64_t nbytes = st_nbytes(S, name);
    if (nbytes < 0) {
        fprintf(stderr, "missing %s tensor: %s\n", kind, name);
        return 0;
    }
    if (!strcmp(kind, "expert")) {
        st->expert_tensors++;
        st->expert_bytes += nbytes;
    } else if (!strcmp(kind, "scale")) {
        st->scale_bytes += nbytes;
    } else {
        st->dense_bytes += nbytes;
    }
    return 1;
}

static int qwen_check_layout(const QwenCfg *cfg, const char *snap, QwenLayoutStats *stats) {
    shards S;
    st_init(&S, snap);
    if (S.n == 0) {
        printf("no safetensors found; config-only scaffold check\n");
        return 1;
    }

    const char *top[] = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    };
    for (int i = 0; i < (int)(sizeof(top) / sizeof(top[0])); i++) {
        if (!qwen_require(&S, top[i], "dense", stats)) stats->dense_missing++;
    }

    char name[512];
    for (int l = 0; l < cfg->n_layers; l++) {
        const char *layer_dense[] = {
            "model.layers.%d.input_layernorm.weight",
            "model.layers.%d.post_attention_layernorm.weight",
            "model.layers.%d.self_attn.q_proj.weight",
            "model.layers.%d.self_attn.k_proj.weight",
            "model.layers.%d.self_attn.v_proj.weight",
            "model.layers.%d.self_attn.o_proj.weight",
            "model.layers.%d.mlp.gate.weight",
        };
        for (int i = 0; i < (int)(sizeof(layer_dense) / sizeof(layer_dense[0])); i++) {
            snprintf(name, sizeof(name), layer_dense[i], l);
            if (!qwen_require(&S, name, "dense", stats)) stats->dense_missing++;
        }

        const char *optional_dense[] = {
            "model.layers.%d.self_attn.q_proj.bias",
            "model.layers.%d.self_attn.k_proj.bias",
            "model.layers.%d.self_attn.v_proj.bias",
            "model.layers.%d.mlp.shared_expert.gate_proj.weight",
            "model.layers.%d.mlp.shared_expert.up_proj.weight",
            "model.layers.%d.mlp.shared_expert.down_proj.weight",
            "model.layers.%d.mlp.shared_expert_gate.weight",
        };
        for (int i = 0; i < (int)(sizeof(optional_dense) / sizeof(optional_dense[0])); i++) {
            snprintf(name, sizeof(name), optional_dense[i], l);
            if (st_has(&S, name)) qwen_require(&S, name, "dense", stats);
        }

        for (int e = 0; e < cfg->n_experts; e++) {
            const char *proj[] = {"gate_proj", "up_proj", "down_proj"};
            for (int p = 0; p < 3; p++) {
                snprintf(name, sizeof(name),
                         "model.layers.%d.mlp.experts.%d.%s.weight", l, e, proj[p]);
                if (!qwen_require(&S, name, "expert", stats)) stats->expert_missing++;
                snprintf(name, sizeof(name),
                         "model.layers.%d.mlp.experts.%d.%s.weight.qs", l, e, proj[p]);
                if (!qwen_require(&S, name, "scale", stats)) stats->expert_scale_missing++;
            }
        }
    }

    printf("layout tensors=%d dense=%.2f MB experts=%.2f MB scales=%.2f MB expert_tensors=%d\n",
           S.n,
           stats->dense_bytes / 1048576.0,
           stats->expert_bytes / 1048576.0,
           stats->scale_bytes / 1048576.0,
           stats->expert_tensors);
    return stats->dense_missing == 0 && stats->expert_missing == 0 && stats->expert_scale_missing == 0;
}

static float *qwen_load_t(QwenModel *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);
    return p;
}

static float *qwen_load_optional_t(QwenModel *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) return NULL;
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);
    return p;
}

static void qwen_model_init(QwenModel *m, const char *snap, int cap) {
    memset(m, 0, sizeof(*m));
    qwen_load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    QwenCfg *c = &m->c;
    double t0 = now_s();
    m->embed = qwen_load_t(m, "model.embed_tokens.weight");
    m->lm_head = qwen_load_t(m, "lm_head.weight");
    m->final_norm = qwen_load_t(m, "model.norm.weight");
    m->L = calloc(c->n_layers, sizeof(QwenLayer));
    if (!m->L) { fprintf(stderr, "OOM layers\n"); exit(1); }
    char nm[512];
    for (int i = 0; i < c->n_layers; i++) {
        QwenLayer *l = &m->L[i];
        #define LD(field, suffix) snprintf(nm, sizeof(nm), "model.layers.%d." suffix, i); l->field = qwen_load_t(m, nm)
        #define LDO(field, suffix) snprintf(nm, sizeof(nm), "model.layers.%d." suffix, i); l->field = qwen_load_optional_t(m, nm)
        LD(in_ln, "input_layernorm.weight");
        LD(post_ln, "post_attention_layernorm.weight");
        LD(q, "self_attn.q_proj.weight");
        LD(k, "self_attn.k_proj.weight");
        LD(v, "self_attn.v_proj.weight");
        LD(o, "self_attn.o_proj.weight");
        LDO(qb, "self_attn.q_proj.bias");
        LDO(kb, "self_attn.k_proj.bias");
        LDO(vb, "self_attn.v_proj.bias");
        LD(gate, "mlp.gate.weight");
        LDO(shared_gate, "mlp.shared_expert.gate_proj.weight");
        LDO(shared_up, "mlp.shared_expert.up_proj.weight");
        LDO(shared_down, "mlp.shared_expert.down_proj.weight");
        LDO(shared_gate_score, "mlp.shared_expert_gate.weight");
        #undef LD
        #undef LDO
    }
    m->cache = calloc(c->n_layers, sizeof(QwenLayerCache));
    if (!m->cache) { fprintf(stderr, "OOM expert caches\n"); exit(1); }
    for (int i = 0; i < c->n_layers; i++) {
        m->cache[i].cap = cap;
        m->cache[i].slots = calloc(cap, sizeof(QwenSlot));
        if (cap && !m->cache[i].slots) { fprintf(stderr, "OOM expert cache slots\n"); exit(1); }
    }
    m->dense_load_s = now_s() - t0;
}

static void read_expert_quant(QwenModel *m, const char *name, int8_t *q, float *scale) {
    int64_t nb = st_nbytes(&m->S, name);
    if (nb < 0) { fprintf(stderr, "missing expert tensor: %s\n", name); exit(1); }
    st_read_raw(&m->S, name, q, 1);
    char qs[640];
    snprintf(qs, sizeof(qs), "%s.qs", name);
    st_read_f32(&m->S, qs, scale, 1);
}

static int qwen_expert_cached(QwenModel *m, int layer, int eid) {
    QwenLayerCache *lc = &m->cache[layer];
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) return 1;
    return 0;
}

static void qwen_expert_prefetch_raw(QwenModel *m, int layer, int eid) {
    char nm[512];
    const char *proj[] = {"gate_proj", "up_proj", "down_proj"};
    for (int p = 0; p < 3; p++) {
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.%s.weight", layer, eid, proj[p]);
        st_prefetch(&m->S, nm);
        char qs[640];
        snprintf(qs, sizeof(qs), "%s.qs", nm);
        st_prefetch(&m->S, qs);
    }
}

static void *qwen_prefetch_worker(void *arg) {
    QwenPrefetchQueue *q = arg;
    for (;;) {
        pthread_mutex_lock(&q->mx);
        while (q->r == q->w) pthread_cond_wait(&q->cv, &q->mx);
        QwenPrefetchJob job = q->q[q->r & 4095u];
        q->r++;
        pthread_mutex_unlock(&q->mx);
        qwen_expert_prefetch_raw(q->m, job.layer, job.expert);
    }
    return NULL;
}

static void qwen_prefetch_queue_start(QwenModel *m) {
    if (g_qwen_pf.started) return;
    memset(&g_qwen_pf, 0, sizeof(g_qwen_pf));
    g_qwen_pf.m = m;
    pthread_mutex_init(&g_qwen_pf.mx, NULL);
    pthread_cond_init(&g_qwen_pf.cv, NULL);
    if (pthread_create(&g_qwen_pf.th, NULL, qwen_prefetch_worker, &g_qwen_pf) != 0) {
        perror("pthread_create qwen prefetch");
        return;
    }
    pthread_detach(g_qwen_pf.th);
    g_qwen_pf.started = 1;
}

static void qwen_expert_prefetch(QwenModel *m, int layer, int eid) {
    if (layer < 0 || layer >= m->c.n_layers || eid < 0 || eid >= m->c.n_experts) return;
    if (qwen_expert_cached(m, layer, eid)) {
        m->forecast_skipped_cached++;
        return;
    }
    qwen_prefetch_queue_start(m);
    if (!g_qwen_pf.started) {
        qwen_expert_prefetch_raw(m, layer, eid);
        m->prefetch_calls++;
        return;
    }
    pthread_mutex_lock(&g_qwen_pf.mx);
    if (g_qwen_pf.w - g_qwen_pf.r >= 4096u) {
        m->prefetch_dropped++;
        pthread_mutex_unlock(&g_qwen_pf.mx);
        return;
    }
    g_qwen_pf.q[g_qwen_pf.w & 4095u].layer = layer;
    g_qwen_pf.q[g_qwen_pf.w & 4095u].expert = eid;
    g_qwen_pf.w++;
    m->prefetch_enqueued++;
    m->prefetch_calls++;
    pthread_cond_signal(&g_qwen_pf.cv);
    pthread_mutex_unlock(&g_qwen_pf.mx);
}

static void qwen_prefetch_forecast(QwenModel *m, const forecast_item *items, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!forecast_item_valid(&items[i])) continue;
        qwen_expert_prefetch(m, (int)items[i].layer, (int)items[i].expert);
    }
}

static void expert_get(QwenModel *m, int layer, int eid, QwenSlot **out) {
    QwenLayerCache *lc = &m->cache[layer];
    for (int i = 0; i < lc->n; i++) {
        if (lc->slots[i].eid == eid) {
            m->hits++;
            lc->slots[i].used = ++m->clock;
            *out = &lc->slots[i];
            return;
        }
    }
    m->miss++;
    QwenCfg *c = &m->c;
    int64_t ng = (int64_t)c->inter * c->hidden;
    int64_t nd = (int64_t)c->hidden * c->inter;
    QwenSlot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        s->g = ialloc(ng);
        s->u = ialloc(ng);
        s->d = ialloc(nd);
        s->gs = falloc(c->inter);
        s->us = falloc(c->inter);
        s->ds = falloc(c->hidden);
    } else {
        int lru = 0;
        for (int i = 1; i < lc->n; i++) if (lc->slots[i].used < lc->slots[lru].used) lru = i;
        s = &lc->slots[lru];
    }
    char nm[512];
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.gate_proj.weight", layer, eid);
    read_expert_quant(m, nm, s->g, s->gs);
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.up_proj.weight", layer, eid);
    read_expert_quant(m, nm, s->u, s->us);
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.down_proj.weight", layer, eid);
    read_expert_quant(m, nm, s->d, s->ds);
    s->eid = eid;
    s->used = ++m->clock;
    *out = s;
}

static void rope_head(float *x, int pos, const QwenCfg *c) {
    int half = c->head_dim / 2;
    for (int j = 0; j < half; j++) {
        float inv = powf(c->theta, -2.0f * j / c->head_dim);
        float ang = pos * inv;
        float cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j + half];
        x[j] = a * cs - b * sn;
        x[j + half] = b * cs + a * sn;
    }
}

static void qwen_attention(QwenModel *m, QwenLayer *l, int layer, float *x, int S, int pos_base, float *out) {
    QwenCfg *c = &m->c;
    int H = c->n_heads, KVH = c->n_kv_heads, hd = c->head_dim, D = c->hidden;
    int KD = KVH * hd;
    float *q = falloc((int64_t)S * D);
    float *k = falloc((int64_t)S * KD);
    float *v = falloc((int64_t)S * KD);
    matmul(q, x, l->q, S, D, D);
    matmul(k, x, l->k, S, D, KD);
    matmul(v, x, l->v, S, D, KD);
    add_bias(q, l->qb, S, D);
    add_bias(k, l->kb, S, KD);
    add_bias(v, l->vb, S, KD);

    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        for (int hh = 0; hh < H; hh++) rope_head(q + (int64_t)s * D + hh * hd, pos, c);
        for (int hh = 0; hh < KVH; hh++) rope_head(k + (int64_t)s * KD + hh * hd, pos, c);
    }

    for (int s = 0; s < S; s++) {
        int t = pos_base + s;
        for (int hh = 0; hh < KVH; hh++) {
            memcpy(m->K[layer] + ((int64_t)hh * m->max_t + t) * hd,
                   k + (int64_t)s * KD + hh * hd, hd * sizeof(float));
            memcpy(m->V[layer] + ((int64_t)hh * m->max_t + t) * hd,
                   v + (int64_t)s * KD + hh * hd, hd * sizeof(float));
        }
    }

    float scale = 1.f / sqrtf((float)hd);
    float *ctx = falloc((int64_t)S * D);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) {
        for (int s = 0; s < S; s++) {
            int qpos = pos_base + s;
            int kvh = hh / (H / KVH);
            const float *qv = q + (int64_t)s * D + hh * hd;
            float *scores = malloc((size_t)(qpos + 1) * sizeof(float));
            if (!scores) { fprintf(stderr, "OOM attention scores\n"); exit(1); }
            for (int t = 0; t <= qpos; t++) {
                const float *kv = m->K[layer] + ((int64_t)kvh * m->max_t + t) * hd;
                float acc = 0.f;
                for (int d = 0; d < hd; d++) acc += qv[d] * kv[d];
                scores[t] = acc * scale;
            }
            softmax_row(scores, qpos + 1);
            float *cx = ctx + (int64_t)s * D + hh * hd;
            for (int d = 0; d < hd; d++) cx[d] = 0.f;
            for (int t = 0; t <= qpos; t++) {
                const float *vv = m->V[layer] + ((int64_t)kvh * m->max_t + t) * hd;
                float a = scores[t];
                for (int d = 0; d < hd; d++) cx[d] += a * vv[d];
            }
            free(scores);
        }
    }
    matmul(out, ctx, l->o, S, D, D);
    free(q);
    free(k);
    free(v);
    free(ctx);
}

static void qwen_shared_expert(QwenModel *m, QwenLayer *l, const float *x, float *out) {
    QwenCfg *c = &m->c;
    int D = c->hidden, I = c->shared_inter;
    if (I <= 0 || !l->shared_gate || !l->shared_up || !l->shared_down || !l->shared_gate_score) {
        memset(out, 0, (size_t)D * sizeof(float));
        return;
    }
    float *g = falloc(I), *u = falloc(I), *h = falloc(I);
    matmul(g, x, l->shared_gate, 1, D, I);
    matmul(u, x, l->shared_up, 1, D, I);
    for (int i = 0; i < I; i++) h[i] = silu(g[i]) * u[i];
    matmul(out, h, l->shared_down, 1, I, D);
    float score = 0.f;
    for (int d = 0; d < D; d++) score += x[d] * l->shared_gate_score[d];
    score = sigmoidf_q(score);
    for (int d = 0; d < D; d++) out[d] *= score;
    free(g);
    free(u);
    free(h);
}

static void qwen_moe(QwenModel *m, QwenLayer *l, int layer, float *x, int S, float *out) {
    QwenCfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, K = c->topk, I = c->inter;
    float *logits = falloc((int64_t)S * E);
    matmul(logits, x, l->gate, S, D, E);
    memset(out, 0, (size_t)S * D * sizeof(float));
    float *g = falloc(I), *u = falloc(I), *mid = falloc(I), *hh = falloc(D), *shared = falloc(D);
    int *all_idx = malloc((size_t)S * K * sizeof(int));
    float *all_val = malloc((size_t)S * K * sizeof(float));
    if (!all_idx || !all_val) { fprintf(stderr, "OOM qwen route scratch\n"); exit(1); }
    for (int s = 0; s < S; s++) {
        float *pr = logits + (int64_t)s * E;
        softmax_row(pr, E);
        if (K > 128) { fprintf(stderr, "topk too large for qwen_moe scratch: %d\n", K); exit(1); }
        for (int kk = 0; kk < K; kk++) {
            int best = -1;
            float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int taken = 0;
                for (int j = 0; j < kk; j++) if (all_idx[(int64_t)s * K + j] == e) { taken = 1; break; }
                if (!taken && pr[e] > bv) { bv = pr[e]; best = e; }
            }
            all_idx[(int64_t)s * K + kk] = best;
            all_val[(int64_t)s * K + kk] = bv;
        }
        if (c->norm_topk) {
            float sm = 0.f;
            for (int kk = 0; kk < K; kk++) sm += all_val[(int64_t)s * K + kk];
            for (int kk = 0; kk < K; kk++) all_val[(int64_t)s * K + kk] /= sm;
        }
    }
    if (g_speedax_prefetch) {
        forecast_item *items = malloc((size_t)S * K * sizeof(forecast_item));
        if (!items) { fprintf(stderr, "OOM qwen forecast\n"); exit(1); }
        size_t n = 0;
        for (int s = 0; s < S; s++) {
            for (int kk = 0; kk < K; kk++) {
                int eid = all_idx[(int64_t)s * K + kk];
                int seen = 0;
                for (size_t j = 0; j < n; j++) {
                    if (items[j].layer == (uint16_t)layer && items[j].expert == (uint16_t)eid) {
                        seen = 1;
                        break;
                    }
                }
                if (!seen) {
                    n = forecast_push(items, n, (size_t)S * K, (uint16_t)layer, (uint16_t)eid,
                                      (uint32_t)(layer + 1), all_val[(int64_t)s * K + kk]);
                }
            }
        }
        m->forecast_items += n;
        qwen_prefetch_forecast(m, items, n);
        free(items);
    }
    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s * D;
        float *os = out + (int64_t)s * D;
        for (int kk = 0; kk < K; kk++) {
            QwenSlot *e;
            expert_get(m, layer, all_idx[(int64_t)s * K + kk], &e);
            matmul_q(g, xs, e->g, e->gs, D, I);
            matmul_q(u, xs, e->u, e->us, D, I);
            for (int i = 0; i < I; i++) mid[i] = silu(g[i]) * u[i];
            matmul_q(hh, mid, e->d, e->ds, I, D);
            for (int d = 0; d < D; d++) os[d] += all_val[(int64_t)s * K + kk] * hh[d];
        }
        qwen_shared_expert(m, l, xs, shared);
        for (int d = 0; d < D; d++) os[d] += shared[d];
    }
    free(logits);
    free(g);
    free(u);
    free(mid);
    free(hh);
    free(shared);
    free(all_idx);
    free(all_val);
}

static void qwen_pilot_prefetch_next(QwenModel *m, int layer, const float *x, int S) {
    QwenCfg *c = &m->c;
    if (layer < 0 || layer >= c->n_layers) return;
    int D = c->hidden, E = c->n_experts;
    int K = g_speedax_pilot_k > 0 && g_speedax_pilot_k < c->topk ? g_speedax_pilot_k : c->topk;
    QwenLayer *l = &m->L[layer];
    float *nrm = falloc(D);
    float *scores = falloc(E);
    forecast_item *items = malloc((size_t)S * K * sizeof(forecast_item));
    if (!items) { fprintf(stderr, "OOM qwen pilot forecast\n"); exit(1); }
    size_t n = 0;
    for (int s = 0; s < S; s++) {
        rmsnorm_row(nrm, x + (int64_t)s * D, l->post_ln, D, c->eps);
        matmul(scores, nrm, l->gate, 1, D, E);
        softmax_row(scores, E);
        int selected[128];
        if (K > 128) { fprintf(stderr, "pilot topk too large: %d\n", K); exit(1); }
        for (int kk = 0; kk < K; kk++) {
            int best = -1;
            float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int seen = 0;
                for (int j = 0; j < kk; j++) if (selected[j] == e) { seen = 1; break; }
                if (!seen && scores[e] > bv) { bv = scores[e]; best = e; }
            }
            selected[kk] = best;
            int dup = 0;
            for (size_t j = 0; j < n; j++) {
                if (items[j].layer == (uint16_t)layer && items[j].expert == (uint16_t)best) {
                    dup = 1;
                    break;
                }
            }
            if (!dup) {
                n = forecast_push(items, n, (size_t)S * K, (uint16_t)layer, (uint16_t)best,
                                  (uint32_t)(layer + 1), bv);
            }
        }
    }
    m->forecast_items += n;
    qwen_prefetch_forecast(m, items, n);
    free(nrm);
    free(scores);
    free(items);
}

static float *qwen_step(QwenModel *m, const int *ids, int S, int pos_base) {
    QwenCfg *c = &m->c;
    int D = c->hidden;
    float *x = falloc((int64_t)S * D);
    for (int s = 0; s < S; s++) memcpy(x + (int64_t)s * D, m->embed + (int64_t)ids[s] * D, D * sizeof(float));
    float *nrm = falloc((int64_t)S * D);
    float *tmp = falloc((int64_t)S * D);
    for (int i = 0; i < c->n_layers; i++) {
        QwenLayer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s * D, x + (int64_t)s * D, l->in_ln, D, c->eps);
        qwen_attention(m, l, i, nrm, S, pos_base, tmp);
        for (int64_t j = 0; j < (int64_t)S * D; j++) x[j] += tmp[j];
        if (g_speedax_pilot && i + 1 < c->n_layers) qwen_pilot_prefetch_next(m, i + 1, x, S);
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s * D, x + (int64_t)s * D, l->post_ln, D, c->eps);
        qwen_moe(m, l, i, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S * D; j++) x[j] += tmp[j];
    }
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S - 1) * D, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    matmul(logit, last, m->lm_head, 1, D, c->vocab);
    free(x);
    free(nrm);
    free(tmp);
    free(last);
    return logit;
}

static void qwen_generate(QwenModel *m, const int *prompt, int np, int n_new, int *out) {
    QwenCfg *c = &m->c;
    m->max_t = np + n_new;
    m->K = calloc(c->n_layers, sizeof(float *));
    m->V = calloc(c->n_layers, sizeof(float *));
    if (!m->K || !m->V) { fprintf(stderr, "OOM kv-cache table\n"); exit(1); }
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)c->n_kv_heads * m->max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_kv_heads * m->max_t * c->head_dim);
    }
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = qwen_step(m, prompt, np, 0);
    int len = np;
    for (int s = 0; s < n_new; s++) {
        int best = 0;
        float bv = logit[0];
        for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        free(logit);
        out[len++] = best;
        if (s == n_new - 1) break;
        int one = best;
        logit = qwen_step(m, &one, 1, len - 1);
    }
}

static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a) { fprintf(stderr, "missing ref key: %s\n", key); exit(1); }
    int *r = malloc((size_t)a->len * sizeof(int));
    if (!r) { fprintf(stderr, "OOM ref array\n"); exit(1); }
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len;
    return r;
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) {
        fprintf(stderr, "set SNAP=<Qwen MoE snapshot directory>\n");
        return 1;
    }
    g_speedax_prefetch = getenv("SPEEDAX_PREFETCH") ? atoi(getenv("SPEEDAX_PREFETCH")) : 0;
    g_speedax_pilot = getenv("SPEEDAX_PILOT") ? atoi(getenv("SPEEDAX_PILOT")) : 0;
    g_speedax_pilot_k = getenv("SPEEDAX_PILOT_K") ? atoi(getenv("SPEEDAX_PILOT_K")) : 0;
    if (g_speedax_pilot_k < 0) g_speedax_pilot_k = 0;
    QwenCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    qwen_load_cfg(&cfg, snap);
    qwen_print_cfg(&cfg);
    if (!qwen_validate_cfg(&cfg)) {
        fprintf(stderr, "invalid or unsupported Qwen MoE config in %s/config.json\n", snap);
        return 2;
    }
    QwenLayoutStats stats;
    memset(&stats, 0, sizeof(stats));
    if (!qwen_check_layout(&cfg, snap, &stats)) {
        fprintf(stderr, "invalid Qwen MoE tensor layout: dense_missing=%d expert_missing=%d scale_missing=%d\n",
                stats.dense_missing, stats.expert_missing, stats.expert_scale_missing);
        return 3;
    }
    if (argc < 2) {
        fprintf(stderr, "qwen_moe scaffold/layout check succeeded.\n");
        return 0;
    }

    const char *refpath = argv[1];
    int cap = argc > 2 ? atoi(argv[2]) : cfg.n_experts;
    if (cap < 1) cap = 1;
    FILE *f = fopen(refpath, "rb");
    if (!f) { perror(refpath); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    if (!buf) { fprintf(stderr, "OOM reading %s\n", refpath); return 1; }
    if (fread(buf, 1, n, f) != (size_t)n) {}
    buf[n] = 0;
    fclose(f);
    char *arena = NULL;
    jval *ref = json_parse(buf, &arena);
    int np, nfull;
    int *prompt = read_int_array(ref, "prompt_ids", &np);
    int *full = NULL;
    int compare_tokens = json_get(ref, "full_ids") != NULL;
    if (compare_tokens) {
        full = read_int_array(ref, "full_ids", &nfull);
    } else {
        jval *mn = json_get(ref, "max_new_tokens");
        if (!mn) { fprintf(stderr, "reference JSON needs full_ids or max_new_tokens\n"); return 1; }
        nfull = np + (int)mn->num;
    }
    int n_new = nfull - np;
    if (n_new < 0) { fprintf(stderr, "invalid reference lengths\n"); return 1; }

    printf("== Qwen MoE tiny forward, cache = %d experts/layer ==\n", cap);
    QwenModel m;
    qwen_model_init(&m, snap, cap);
    printf("resident weights loaded in %.3fs | RSS after load: %.3f GB\n", m.dense_load_s, rss_gb());
    int *out = malloc((size_t)nfull * sizeof(int));
    if (!out) { fprintf(stderr, "OOM output ids\n"); return 1; }
    double t0 = now_s();
    qwen_generate(&m, prompt, np, n_new, out);
    double dt = now_s() - t0;

    int match = 0;
    if (compare_tokens) {
        printf("\nReference: ");
        for (int i = np; i < nfull; i++) printf("%d ", full[i]);
    } else {
        printf("\nReference: <bench-only; no expected tokens>");
    }
    printf("\nC engine : ");
    for (int i = np; i < nfull; i++) {
        printf("%d ", out[i]);
        if (compare_tokens && out[i] == full[i]) match++;
    }
    if (compare_tokens) printf("\nMatching tokens: %d/%d\n", match, n_new);
    else printf("\nMatching tokens: skipped bench-only\n");
    double tot = (double)m.hits + (double)m.miss;
    printf("Expert cache hit rate: %.1f%% (hit=%llu miss=%llu)\n",
           tot ? 100.0 * (double)m.hits / tot : 0.0,
           (unsigned long long)m.hits,
           (unsigned long long)m.miss);
    printf("Speedax forecast items=%llu prefetch_calls=%llu enqueued=%llu dropped=%llu skipped_cached=%llu observed=%d pilot=%d pilot_k=%d\n",
           (unsigned long long)m.forecast_items,
           (unsigned long long)m.prefetch_calls,
           (unsigned long long)m.prefetch_enqueued,
           (unsigned long long)m.prefetch_dropped,
           (unsigned long long)m.forecast_skipped_cached,
           g_speedax_prefetch,
           g_speedax_pilot,
           g_speedax_pilot_k);
    printf("Speed: %.2f tok/s (%.3fs for %d tokens)\n", n_new / dt, dt, n_new);
    free(buf);
    free(arena);
    return (!compare_tokens || match == n_new) ? 0 : 4;
}
