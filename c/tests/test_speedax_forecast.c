#include <stdio.h>
#include "../speedax_forecast.h"

static int fail(const char *message) {
    fprintf(stderr, "speedax forecast test failed: %s\n", message);
    return 1;
}

int main(void) {
    forecast_item items[2];
    size_t n = 0;
    n = forecast_push(items, n, 2, 3, 7, 12, 0.8f);
    n = forecast_push(items, n, 2, 2, 1, 10, 0.5f);
    n = forecast_push(items, n, 2, 9, 9, 99, 0.5f);
    if (n != 2) return fail("capacity not enforced");
    if (!forecast_item_valid(&items[0])) return fail("valid item rejected");
    if (!forecast_item_before(&items[1], &items[0])) return fail("deadline ordering");
    forecast_item bad = {0, 0, 1, 0.0f};
    if (forecast_item_valid(&bad)) return fail("zero probability accepted");
    puts("speedax forecast tests: ok");
    return 0;
}
