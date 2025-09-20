#include <stdio.h>
#include <string.h>

#include "symnmf.h"

static int print_matrix(const matrix_t *matrix);
static int output_similarity_goal(const matrix_t *points);
static int output_degree_goal(const matrix_t *points);
static int output_normalized_goal(const matrix_t *points);

int symnmf_cli_main(int argc, char **argv) {
    matrix_t *points = NULL;
    int status = SYM_FAILURE;

    if (argc != 3) {
        goto error;
    }

    if (dataset_from_file(argv[2], &points) != SYM_SUCCESS) {
        goto error;
    }

    if (strcmp(argv[1], "sym") == 0) {
        status = output_similarity_goal(points);
    } else if (strcmp(argv[1], "ddg") == 0) {
        status = output_degree_goal(points);
    } else if (strcmp(argv[1], "norm") == 0) {
        status = output_normalized_goal(points);
    } else {
        goto error;
    }

    if (status != SYM_SUCCESS) {
        goto error;
    }

    matrix_free(points);
    return SYM_SUCCESS;

error:
    symnmf_log_error();
    matrix_free(points);
    return SYM_FAILURE;
}

int main(int argc, char **argv) {
    int status = symnmf_cli_main(argc, argv);
    return (status == SYM_SUCCESS) ? 0 : 1;
}

/* Write matrix entries using "%.4f" CSV formatting. */
static int print_matrix(const matrix_t *matrix) {
    size_t row;
    size_t col;

    if (matrix == NULL || matrix->data == NULL) {
        return SYM_FAILURE;
    }

    for (row = 0; row < matrix->rows; ++row) {
        for (col = 0; col < matrix->cols; ++col) {
            double value = MAT_AT(matrix, row, col);
            if (col == 0) {
                if (printf("%.4f", value) < 0) {
                    return SYM_FAILURE;
                }
            } else {
                if (printf(",%.4f", value) < 0) {
                    return SYM_FAILURE;
                }
            }
        }
        if (printf("\n") < 0) {
            return SYM_FAILURE;
        }
    }

    return SYM_SUCCESS;
}

static int output_similarity_goal(const matrix_t *points) {
    matrix_t *similarity = NULL;
    int status = SYM_FAILURE;

    similarity = matrix_create(points->rows, points->rows);
    if (similarity == NULL || similarity->data == NULL) {
        goto cleanup;
    }

    if (compute_similarity(points, similarity) != SYM_SUCCESS) {
        goto cleanup;
    }

    status = print_matrix(similarity);

cleanup:
    matrix_free(similarity);
    return status;
}

static int output_degree_goal(const matrix_t *points) {
    matrix_t *similarity = NULL;
    matrix_t *degree = NULL;
    int status = SYM_FAILURE;

    similarity = matrix_create(points->rows, points->rows);
    degree = matrix_create(points->rows, points->rows);
    if (similarity == NULL || degree == NULL || similarity->data == NULL || degree->data == NULL) {
        goto cleanup;
    }

    if (compute_similarity(points, similarity) != SYM_SUCCESS) {
        goto cleanup;
    }
    if (compute_degree(similarity, degree) != SYM_SUCCESS) {
        goto cleanup;
    }

    status = print_matrix(degree);

cleanup:
    matrix_free(similarity);
    matrix_free(degree);
    return status;
}

static int output_normalized_goal(const matrix_t *points) {
    matrix_t *similarity = NULL;
    matrix_t *degree = NULL;
    matrix_t *normalized = NULL;
    int status = SYM_FAILURE;

    similarity = matrix_create(points->rows, points->rows);
    degree = matrix_create(points->rows, points->rows);
    normalized = matrix_create(points->rows, points->rows);
    if (similarity == NULL || degree == NULL || normalized == NULL) {
        goto cleanup;
    }
    if (similarity->data == NULL || degree->data == NULL || normalized->data == NULL) {
        goto cleanup;
    }

    if (compute_similarity(points, similarity) != SYM_SUCCESS) {
        goto cleanup;
    }
    if (compute_degree(similarity, degree) != SYM_SUCCESS) {
        goto cleanup;
    }
    if (compute_normalized(similarity, degree, normalized) != SYM_SUCCESS) {
        goto cleanup;
    }

    status = print_matrix(normalized);

cleanup:
    matrix_free(similarity);
    matrix_free(degree);
    matrix_free(normalized);
    return status;
}
