#ifndef SPEEDAX_FORECAST_H
#define SPEEDAX_FORECAST_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint16_t layer;
    uint16_t expert;
    uint32_t deadline_event;
    float probability;
} forecast_item;

typedef struct {
    const forecast_item *items;
    size_t n_items;
} forecast_view;

static int forecast_item_valid(const forecast_item *item) {
    return item && item->probability > 0.0f && item->probability <= 1.0f;
}

static int forecast_item_before(const forecast_item *a, const forecast_item *b) {
    if (a->deadline_event != b->deadline_event) return a->deadline_event < b->deadline_event;
    if (a->layer != b->layer) return a->layer < b->layer;
    return a->expert < b->expert;
}

static size_t forecast_push(
    forecast_item *out,
    size_t n,
    size_t cap,
    uint16_t layer,
    uint16_t expert,
    uint32_t deadline_event,
    float probability
) {
    if (!out || n >= cap) return n;
    forecast_item item = {layer, expert, deadline_event, probability};
    if (!forecast_item_valid(&item)) return n;
    out[n] = item;
    return n + 1;
}

#endif
