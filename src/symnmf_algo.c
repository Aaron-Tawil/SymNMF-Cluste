#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "symnmf.h"

static double squared_distance(const matrix_t *points, size_t lhs, size_t rhs);
static int compute_scale_vector(const matrix_t *degree, double **scale_out);
static int validate_normalized_args(const matrix_t *similarity, const matrix_t *degree, matrix_t *normalized, size_t *n_out);
static void populate_normalized(const matrix_t *similarity, const double *scale, matrix_t *normalized, size_t n);
static int validate_factorize_args(matrix_t *basis, const matrix_t *normalized, size_t k, size_t *n_out);
static int multiply_square_rect(const matrix_t *square, const matrix_t *rect, matrix_t *out);
static int compute_gram_matrix(const matrix_t *basis, matrix_t *gram);
static int multiply_basis_gram(const matrix_t *basis, const matrix_t *gram, matrix_t *out);
static double update_basis_step(matrix_t *basis, const matrix_t *numerator, const matrix_t *denominator, matrix_t *next, double beta);
static int run_iteration(matrix_t *basis, const matrix_t *normalized, matrix_t *numerator, matrix_t *denominator, matrix_t *gram, matrix_t *next, double beta, double *delta_out);

/* Compute the Gaussian similarity matrix A. */
int compute_similarity(const matrix_t *points, matrix_t *similarity) {
    size_t n;
    size_t i;
    size_t j;

    if (points == NULL || similarity == NULL || points->data == NULL || similarity->data == NULL) {
        return SYM_FAILURE;
    }

    n = points->rows;
    if (points->cols == 0 || similarity->rows != n || similarity->cols != n) {
        return SYM_FAILURE;
    }

    for (i = 0; i < n; ++i) {
        MAT_AT(similarity, i, i) = 0.0;
        for (j = i + 1; j < n; ++j) {
            double value = exp(-0.5 * squared_distance(points, i, j));
            MAT_AT(similarity, i, j) = value;
            MAT_AT(similarity, j, i) = value;
        }
    }

    return SYM_SUCCESS;
}

/* Populate the diagonal degree matrix D. */
int compute_degree(const matrix_t *similarity, matrix_t *degree) {
    size_t n;
    size_t i;
    size_t j;

    if (similarity == NULL || degree == NULL || similarity->data == NULL || degree->data == NULL) {
        return SYM_FAILURE;
    }

    n = similarity->rows;
    if (similarity->cols != n || degree->rows != n || degree->cols != n) {
        return SYM_FAILURE;
    }

    memset(degree->data, 0, n * n * sizeof(double));
    for (i = 0; i < n; ++i) {
        double sum = 0.0;
        for (j = 0; j < n; ++j) {
            sum += MAT_AT(similarity, i, j);
        }
        MAT_AT(degree, i, i) = sum;
    }

    return SYM_SUCCESS;
}

/* Compute W = D^{-1/2} A D^{-1/2}. */
int compute_normalized(const matrix_t *similarity, const matrix_t *degree, matrix_t *normalized) {
    double *scale = NULL;
    size_t n = 0;

    if (validate_normalized_args(similarity, degree, normalized, &n) != SYM_SUCCESS) {
        return SYM_FAILURE;
    }

    if (n == 0) {
        return SYM_SUCCESS;
    }

    if (compute_scale_vector(degree, &scale) != SYM_SUCCESS) {
        return SYM_FAILURE;
    }

    populate_normalized(similarity, scale, normalized, n);
    free(scale);
    return SYM_SUCCESS;
}

/* Run the relaxed SymNMF iterations until convergence or max_iter. */
int symnmf_factorize(matrix_t *basis, const matrix_t *normalized, size_t k, double epsilon, size_t max_iter) {
    matrix_t *numerator = NULL, *denominator = NULL, *gram = NULL, *next = NULL;
    size_t n = 0, iter;
    double delta = 0.0;
    const double beta = 0.5;
    int status = SYM_FAILURE;

    if (validate_factorize_args(basis, normalized, k, &n) != SYM_SUCCESS) {
        return SYM_FAILURE;
    }

    numerator = matrix_create(n, k);
    denominator = matrix_create(n, k);
    gram = matrix_create(k, k);
    next = matrix_create(n, k);
    if (numerator == NULL || denominator == NULL || gram == NULL || next == NULL) {
        goto cleanup;
    }

    for (iter = 0; iter < max_iter; ++iter) {
        if (run_iteration(basis, normalized, numerator, denominator, gram, next, beta, &delta) != SYM_SUCCESS) {
            goto cleanup;
        }
        if (delta < epsilon) {
            break;
        }
    }

    status = SYM_SUCCESS;

cleanup:
    matrix_free(numerator);
    matrix_free(denominator);
    matrix_free(gram);
    matrix_free(next);
    return status;
}

/* Return the squared Euclidean distance between two data points. */
static double squared_distance(const matrix_t *points, size_t lhs, size_t rhs) {
    size_t dim;
    double acc = 0.0;

    for (dim = 0; dim < points->cols; ++dim) {
        double diff = MAT_AT(points, lhs, dim) - MAT_AT(points, rhs, dim);
        acc += diff * diff;
    }

    return (acc < 0.0) ? 0.0 : acc;
}

/* Extract D^{-1/2} as a diagonal scale vector. */
static int compute_scale_vector(const matrix_t *degree, double **scale_out) {
    double *scale;
    size_t n;
    size_t i;

    if (degree == NULL || scale_out == NULL) {
        return SYM_FAILURE;
    }

    n = degree->rows;
    scale = (double *)malloc(n * sizeof(double));
    if (scale == NULL) {
        return SYM_FAILURE;
    }

    for (i = 0; i < n; ++i) {
        double diag = MAT_AT(degree, i, i);
        scale[i] = (diag <= 0.0) ? 0.0 : 1.0 / sqrt(diag);
    }

    *scale_out = scale;
    return SYM_SUCCESS;
}

/* Validate input matrices for compute_normalized and return their order. */
static int validate_normalized_args(const matrix_t *similarity, const matrix_t *degree, matrix_t *normalized, size_t *n_out) {
    size_t n;

    if (similarity == NULL || degree == NULL || normalized == NULL || n_out == NULL) {
        return SYM_FAILURE;
    }
    if (similarity->data == NULL || degree->data == NULL || normalized->data == NULL) {
        return SYM_FAILURE;
    }

    n = similarity->rows;
    if (similarity->cols != n || degree->rows != n || degree->cols != n) {
        return SYM_FAILURE;
    }
    if (normalized->rows != n || normalized->cols != n) {
        return SYM_FAILURE;
    }

    *n_out = n;
    return SYM_SUCCESS;
}

/* Populate the normalized matrix using the precomputed scale factors. */
static void populate_normalized(const matrix_t *similarity, const double *scale, matrix_t *normalized, size_t n) {
    size_t i;
    size_t j;

    memset(normalized->data, 0, n * n * sizeof(double));
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            double value = MAT_AT(similarity, i, j);
            if (scale[i] == 0.0 || scale[j] == 0.0) {
                value = 0.0;
            } else {
                value *= scale[i] * scale[j];
            }
            MAT_AT(normalized, i, j) = value;
            MAT_AT(normalized, j, i) = value;
        }
    }
}

/* Validate dimensions and output the shared order for factorization. */
static int validate_factorize_args(matrix_t *basis, const matrix_t *normalized, size_t k, size_t *n_out) {
    size_t n;

    if (basis == NULL || normalized == NULL || n_out == NULL) {
        return SYM_FAILURE;
    }
    if (basis->data == NULL || normalized->data == NULL) {
        return SYM_FAILURE;
    }
    if (normalized->rows != normalized->cols || normalized->cols == 0) {
        return SYM_FAILURE;
    }

    n = normalized->rows;
    if (basis->rows != n || basis->cols != k || k == 0) {
        return SYM_FAILURE;
    }

    *n_out = n;
    return SYM_SUCCESS;
}

/* Multiply an n x n matrix with an n x k matrix into out. */
static int multiply_square_rect(const matrix_t *square, const matrix_t *rect, matrix_t *out) {
    size_t n;
    size_t k;
    size_t i;
    size_t j;
    size_t p;

    if (square == NULL || rect == NULL || out == NULL) {
        return SYM_FAILURE;
    }

    n = square->rows;
    k = rect->cols;
    if (square->cols != n || rect->rows != n || out->rows != n || out->cols != k) {
        return SYM_FAILURE;
    }

    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            double acc = 0.0;
            for (p = 0; p < n; ++p) {
                acc += MAT_AT(square, i, p) * MAT_AT(rect, p, j);
            }
            MAT_AT(out, i, j) = acc;
        }
    }

    return SYM_SUCCESS;
}

/* Compute H^T H into the Gram matrix. */
static int compute_gram_matrix(const matrix_t *basis, matrix_t *gram) {
    size_t n;
    size_t k;
    size_t i;
    size_t j;
    size_t r;

    if (basis == NULL || gram == NULL) {
        return SYM_FAILURE;
    }

    n = basis->rows;
    k = basis->cols;
    if (gram->rows != k || gram->cols != k) {
        return SYM_FAILURE;
    }

    for (i = 0; i < k; ++i) {
        for (j = i; j < k; ++j) {
            double acc = 0.0;
            for (r = 0; r < n; ++r) {
                acc += MAT_AT(basis, r, i) * MAT_AT(basis, r, j);
            }
            MAT_AT(gram, i, j) = acc;
            MAT_AT(gram, j, i) = acc;
        }
    }

    return SYM_SUCCESS;
}

/* Multiply the basis matrix by its Gram matrix into out. */
static int multiply_basis_gram(const matrix_t *basis, const matrix_t *gram, matrix_t *out) {
    size_t n;
    size_t k;
    size_t i;
    size_t j;
    size_t p;

    if (basis == NULL || gram == NULL || out == NULL) {
        return SYM_FAILURE;
    }

    n = basis->rows;
    k = basis->cols;
    if (gram->rows != k || gram->cols != k || out->rows != n || out->cols != k) {
        return SYM_FAILURE;
    }

    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            double acc = 0.0;
            for (p = 0; p < k; ++p) {
                acc += MAT_AT(basis, i, p) * MAT_AT(gram, p, j);
            }
            MAT_AT(out, i, j) = acc;
        }
    }

    return SYM_SUCCESS;
}

/* Apply the relaxed multiplicative SymNMF update and track convergence delta. */
static double update_basis_step(matrix_t *basis, const matrix_t *numerator, const matrix_t *denominator, matrix_t *next, double beta) {
    size_t total;
    size_t idx;
    double diff = 0.0;
    double inertia = 1.0 - beta;

    total = basis->rows * basis->cols;
    for (idx = 0; idx < total; ++idx) {
        double h = basis->data[idx];
        double denom = denominator->data[idx];
        double ratio = 0.0;
        double delta;
        if (denom > 0.0) {
            ratio = numerator->data[idx] / denom;
        }
        next->data[idx] = h * (inertia + beta * ratio);
        if (next->data[idx] < 0.0) {
            next->data[idx] = 0.0;
        }
        delta = next->data[idx] - h;
        diff += delta * delta;
    }

    memcpy(basis->data, next->data, total * sizeof(double));
    return diff;
}

/* Perform one SymNMF iteration: compute WH, Gram matrix, and update H. */
static int run_iteration(matrix_t *basis, const matrix_t *normalized, matrix_t *numerator, matrix_t *denominator, matrix_t *gram, matrix_t *next, double beta, double *delta_out) {
    if (multiply_square_rect(normalized, basis, numerator) != SYM_SUCCESS) {
        return SYM_FAILURE;
    }
    if (compute_gram_matrix(basis, gram) != SYM_SUCCESS) {
        return SYM_FAILURE;
    }
    if (multiply_basis_gram(basis, gram, denominator) != SYM_SUCCESS) {
        return SYM_FAILURE;
    }
    *delta_out = update_basis_step(basis, numerator, denominator, next, beta);
    return SYM_SUCCESS;
}
