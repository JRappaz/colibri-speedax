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
    fprintf(stderr, "qwen_moe forward path is not implemented yet; scaffold/config load succeeded.\n");
    return 0;
}
