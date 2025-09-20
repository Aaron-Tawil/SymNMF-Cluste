#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "symnmf.h"

#define DATASET_BUFFER_SIZE 8192
#define DATASET_ROW_LIMIT 1024

typedef struct {
    double *values;
    size_t capacity;
    size_t count;
    size_t rows;
    size_t cols;
} dataset_builder_t;

static int ensure_capacity(double **buffer, size_t *capacity, size_t required);
static void dataset_builder_init(dataset_builder_t *builder);
static void dataset_builder_free(dataset_builder_t *builder);
static int dataset_builder_append(dataset_builder_t *builder, const double *row_values, size_t row_count);
static int parse_row(char *line, double *row_values, size_t *row_count);
static int ingest_file(FILE *fp, dataset_builder_t *builder);
static int builder_to_matrix(const dataset_builder_t *builder, matrix_t **matrix_out);
static int is_ascii_space(char ch);

/* Print the assignment-mandated error message. */
void symnmf_log_error(void) {
    fprintf(stdout, "An Error Has Occurred\n");
}

/* Allocate a zero-initialized matrix. Returns NULL on failure. */
matrix_t *matrix_create(size_t rows, size_t cols) {
    matrix_t *matrix = NULL;
    size_t total;

    matrix = (matrix_t *)malloc(sizeof(*matrix));
    if (matrix == NULL) {
        return NULL;
    }

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = NULL;

    if (rows == 0 || cols == 0) {
        return matrix;
    }

    if (cols != 0 && rows > (size_t)-1 / cols) {
        free(matrix);
        return NULL;
    }

    total = rows * cols;
    if (total > (size_t)-1 / sizeof(double)) {
        free(matrix);
        return NULL;
    }

    matrix->data = (double *)calloc(total, sizeof(double));
    if (matrix->data == NULL) {
        free(matrix);
        return NULL;
    }

    return matrix;
}

void matrix_free(matrix_t *matrix) {
    if (matrix == NULL) {
        return;
    }

    free(matrix->data);
    free(matrix);
}

/* Read a dataset from a CSV-like text file into a dense matrix. */
int dataset_from_file(const char *path, matrix_t **out_matrix) {
    FILE *fp = NULL;
    dataset_builder_t builder;
    matrix_t *matrix = NULL;
    int status = SYM_FAILURE;

    if (out_matrix == NULL || path == NULL) {
        return SYM_FAILURE;
    }

    dataset_builder_init(&builder);
    *out_matrix = NULL;

    fp = fopen(path, "r");
    if (fp == NULL) {
        goto cleanup;
    }

    if (ingest_file(fp, &builder) != SYM_SUCCESS) {
        goto cleanup;
    }

    if (builder_to_matrix(&builder, &matrix) != SYM_SUCCESS) {
        goto cleanup;
    }

    *out_matrix = matrix;
    matrix = NULL;
    status = SYM_SUCCESS;

cleanup:
    if (fp != NULL) {
        fclose(fp);
    }
    dataset_builder_free(&builder);
    matrix_free(matrix);
    return status;
}

static int ensure_capacity(double **buffer, size_t *capacity, size_t required) {
    size_t new_capacity;
    double *tmp;

    if (buffer == NULL || capacity == NULL) {
        return SYM_FAILURE;
    }

    if (required <= *capacity) {
        return SYM_SUCCESS;
    }

    new_capacity = (*capacity == 0) ? 1024 : *capacity;
    while (new_capacity < required) {
        if (new_capacity > (size_t)-1 / 2) {
            return SYM_FAILURE;
        }
        new_capacity *= 2;
    }

    if (new_capacity > (size_t)-1 / sizeof(double)) {
        return SYM_FAILURE;
    }

    tmp = (double *)realloc(*buffer, new_capacity * sizeof(double));
    if (tmp == NULL) {
        return SYM_FAILURE;
    }

    *buffer = tmp;
    *capacity = new_capacity;
    return SYM_SUCCESS;
}

static void dataset_builder_init(dataset_builder_t *builder) {
    builder->values = NULL;
    builder->capacity = 0;
    builder->count = 0;
    builder->rows = 0;
    builder->cols = 0;
}

static void dataset_builder_free(dataset_builder_t *builder) {
    if (builder == NULL) {
        return;
    }

    free(builder->values);
    builder->values = NULL;
    builder->capacity = 0;
    builder->count = 0;
    builder->rows = 0;
    builder->cols = 0;
}

static int dataset_builder_append(dataset_builder_t *builder, const double *row_values, size_t row_count) {
    if (builder == NULL || row_values == NULL || row_count == 0) {
        return SYM_FAILURE;
    }

    if (builder->cols == 0) {
        builder->cols = row_count;
    } else if (row_count != builder->cols) {
        return SYM_FAILURE;
    }

    if (ensure_capacity(&builder->values, &builder->capacity, builder->count + row_count) != SYM_SUCCESS) {
        return SYM_FAILURE;
    }

    memcpy(builder->values + builder->count, row_values, row_count * sizeof(double));
    builder->count += row_count;
    builder->rows += 1;
    return SYM_SUCCESS;
}

static int parse_row(char *line, double *row_values, size_t *row_count) {
    char *token;
    size_t count = 0;

    if (line == NULL || row_values == NULL || row_count == NULL) {
        return SYM_FAILURE;
    }

    token = strtok(line, ", \t\r\n");
    while (token != NULL) {
        char *endptr = NULL;
        double value = strtod(token, &endptr);

        while (endptr != NULL && *endptr != '\0') {
            if (!is_ascii_space(*endptr)) {
                return SYM_FAILURE;
            }
            endptr++;
        }

        if (count >= DATASET_ROW_LIMIT) {
            return SYM_FAILURE;
        }

        row_values[count++] = value;
        token = strtok(NULL, ", \t\r\n");
    }

    *row_count = count;
    return SYM_SUCCESS;
}

/* Scan the input stream and append parsed rows to the builder. */
static int ingest_file(FILE *fp, dataset_builder_t *builder) {
    char buffer[DATASET_BUFFER_SIZE];
    double row_values[DATASET_ROW_LIMIT];

    if (fp == NULL || builder == NULL) {
        return SYM_FAILURE;
    }

    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        size_t row_count = 0;
        if (parse_row(buffer, row_values, &row_count) != SYM_SUCCESS) {
            return SYM_FAILURE;
        }
        if (row_count == 0) {
            continue;
        }
        if (dataset_builder_append(builder, row_values, row_count) != SYM_SUCCESS) {
            return SYM_FAILURE;
        }
    }

    if (ferror(fp) || builder->rows == 0 || builder->cols == 0) {
        return SYM_FAILURE;
    }

    return SYM_SUCCESS;
}

/* Materialize the builder contents into a newly allocated matrix. */
static int builder_to_matrix(const dataset_builder_t *builder, matrix_t **matrix_out) {
    matrix_t *matrix;

    if (builder == NULL || matrix_out == NULL) {
        return SYM_FAILURE;
    }

    matrix = matrix_create(builder->rows, builder->cols);
    if (matrix == NULL || matrix->data == NULL) {
        matrix_free(matrix);
        return SYM_FAILURE;
    }

    memcpy(matrix->data, builder->values, builder->count * sizeof(double));
    *matrix_out = matrix;
    return SYM_SUCCESS;
}

static int is_ascii_space(char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v';
}
