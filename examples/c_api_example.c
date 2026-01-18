/*
 * C API Example for Vship
 *
 * Demonstrates how to use Vship from C code
 *
 * Build:
 *   gcc -o c_api_example c_api_example.c -I../include -L../target/release -lvship_ffi
 *
 * Run:
 *   LD_LIBRARY_PATH=../target/release ./c_api_example
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vship.h"

int main() {
    printf("Vship C API Example\n");
    printf("===================\n\n");

    // Initialize Vship
    printf("Initializing Vship...\n");
    VshipHandle* ctx = vship_init();
    if (!ctx) {
        fprintf(stderr, "Failed to initialize Vship\n");
        return 1;
    }
    printf("✓ Vship initialized\n\n");

    // Create SSIMULACRA2 metric
    printf("Creating SSIMULACRA2 metric...\n");
    VshipMetricHandle* metric = vship_metric_create(ctx, VSHIP_METRIC_TYPE_SSIMULACRA2);
    if (!metric) {
        fprintf(stderr, "Failed to create metric\n");
        vship_destroy(ctx);
        return 1;
    }
    printf("✓ Metric created\n\n");

    // Create test images (640x480 for example)
    uint32_t width = 640;
    uint32_t height = 480;
    uint32_t pixel_count = width * height;
    uint32_t data_size = pixel_count * 3; // RGB

    printf("Creating test images (%ux%u)...\n", width, height);
    float* reference = (float*)malloc(data_size * sizeof(float));
    float* distorted = (float*)malloc(data_size * sizeof(float));

    if (!reference || !distorted) {
        fprintf(stderr, "Failed to allocate memory\n");
        vship_metric_destroy(metric);
        vship_destroy(ctx);
        return 1;
    }

    // Fill with test pattern
    for (uint32_t i = 0; i < data_size; i++) {
        reference[i] = (float)(i % 256) / 255.0f;
        distorted[i] = (float)((i + 10) % 256) / 255.0f;
    }
    printf("✓ Test images created\n\n");

    // Compute metric
    printf("Computing SSIMULACRA2 score...\n");
    double score;
    VshipErrorCode result = vship_metric_compute(
        metric,
        reference, width, height, VSHIP_IMAGE_FORMAT_RGB,
        distorted, width, height, VSHIP_IMAGE_FORMAT_RGB,
        &score
    );

    if (result != VSHIP_ERROR_CODE_SUCCESS) {
        const char* error_msg = vship_error_string(result);
        fprintf(stderr, "Failed to compute metric: %s\n", error_msg);
        free(reference);
        free(distorted);
        vship_metric_destroy(metric);
        vship_destroy(ctx);
        return 1;
    }

    printf("✓ SSIMULACRA2 score: %.4f\n\n", score);

    // Cleanup
    printf("Cleaning up...\n");
    free(reference);
    free(distorted);
    vship_metric_destroy(metric);
    vship_destroy(ctx);
    printf("✓ Done\n");

    return 0;
}
