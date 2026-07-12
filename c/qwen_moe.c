/* Qwen1.5-MoE backend scaffold for colibri.
 *
 * Milestone 0: model-specific config parsing and routed-expert identity
 * conventions. The full resident forward path will follow the OLMoE backend's
 * shape, but Qwen uses its own tensor names, tokenizer/config, and optional
 * shared experts.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    c->norm_topk = json_bool_default(root, "norm_topk_prob", 1);

    free(buf);
    free(arena);
}

static int qwen_validate_cfg(const QwenCfg *c) {
    return c->hidden > 0 && c->n_layers > 0 && c->n_heads > 0 &&
           c->n_kv_heads > 0 && c->n_experts > 0 && c->topk > 0 &&
           c->inter > 0 && c->vocab > 0 && c->head_dim > 0;
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

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    const char *snap = getenv("SNAP");
    if (!snap) {
        fprintf(stderr, "set SNAP=<Qwen MoE snapshot directory>\n");
        return 1;
    }
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
    fprintf(stderr, "qwen_moe forward path is not implemented yet; scaffold/layout check succeeded.\n");
    return 0;
}
