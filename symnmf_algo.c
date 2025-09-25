#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "symnmf.h"

/**
 * @brief Computes the squared Euclidean distance between two points.
 * @param points A matrix where each row is a data point.
 * @param lhs The index of the first point.
 * @param rhs The index of the second point.
 * @return The squared Euclidean distance between the two points.
 */
static double squared_distance(const matrix_t *points, size_t lhs, size_t rhs);

/**
 * @brief Computes the scaling vector D^{-1/2} from the degree matrix D.
 * @param degree The degree matrix D.
 * @param scale_out A pointer to a double pointer that will be updated to point to the new scaling vector.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
static int compute_scale_vector(const matrix_t *degree, double **scale_out);

/**
 * @brief Validates the arguments for the compute_normalized function.
 * @param similarity The similarity matrix.
 * @param degree The degree matrix.
 * @param normalized The output normalized similarity matrix.
 * @param n_out A pointer to store the number of rows/columns in the matrices.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
static int validate_normalized_args(const matrix_t *similarity, const matrix_t *degree, matrix_t *normalized, size_t *n_out);

/**
 * @brief Populates the normalized similarity matrix W = D^{-1/2} A D^{-1/2}.
 * @param similarity The similarity matrix A.
 * @param scale The scaling vector D^{-1/2}.
 * @param normalized The output normalized similarity matrix W.
 * @param n The number of rows/columns in the matrices.
 */
static void populate_normalized(const matrix_t *similarity, const double *scale, matrix_t *normalized, size_t n);

/**
 * @brief Validates the arguments for the symnmf_factorize function.
 * @param basis The basis matrix H.
 * @param normalized The normalized similarity matrix W.
 * @param k The number of clusters.
 * @param n_out A pointer to store the number of rows in the matrices.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
static int validate_factorize_args(matrix_t *basis, const matrix_t *normalized, size_t k, size_t *n_out);

/**
 * @brief Multiplies a square matrix with a rectangular matrix.
 * @param square The square matrix.
 * @param rect The rectangular matrix.
 * @param out The output matrix.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
static int multiply_square_rect(const matrix_t *square, const matrix_t *rect, matrix_t *out);

/**
 * @brief Computes the Gram matrix H^T H.
 * @param basis The basis matrix H.
 * @param gram The output Gram matrix.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
static int compute_gram_matrix(const matrix_t *basis, matrix_t *gram);

/**
 * @brief Multiplies the basis matrix H by its Gram matrix H^T H.
 * @param basis The basis matrix H.
 * @param gram The Gram matrix H^T H.
 * @param out The output matrix.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
static int multiply_basis_gram(const matrix_t *basis, const matrix_t *gram, matrix_t *out);

/**
 * @brief Performs a single update step for the basis matrix H.
 * @param basis The basis matrix H to be updated.
 * @param numerator The numerator of the update rule.
 * @param denominator The denominator of the update rule.
 * @param next The buffer to store the updated basis matrix.
 * @param beta The beta parameter for the update rule.
 * @return The squared Frobenius norm of the difference between the updated and previous basis matrices.
 */
static double update_basis_step(matrix_t *basis, const matrix_t *numerator, const matrix_t *denominator, matrix_t *next, double beta);

/**
 * @brief Runs a single iteration of the SymNMF algorithm.
 * @param basis The basis matrix H.
 * @param normalized The normalized similarity matrix W.
 * @param numerator A buffer for the numerator of the update rule.
 * @param denominator A buffer for the denominator of the update rule.
 * @param gram A buffer for the Gram matrix.
 * @param next A buffer for the updated basis matrix.
 * @param beta The beta parameter for the update rule.
 * @param delta_out A pointer to store the convergence delta.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
static int run_iteration(matrix_t *basis, const matrix_t *normalized, matrix_t *numerator, matrix_t *denominator, matrix_t *gram, matrix_t *next, double beta, double *delta_out);

/**
 * @brief Computes the similarity matrix from a set of data points.
 *
 * @param points A matrix where each row is a data point.
 * @param similarity The output similarity matrix.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
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

/**
 * @brief Computes the degree matrix from a similarity matrix.
 *
 * @param similarity The similarity matrix.
 * @param degree The output degree matrix.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
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

/**
 * @brief Computes the normalized similarity matrix.
 *
 * @param similarity The similarity matrix.
 * @param degree The degree matrix.
 * @param normalized The output normalized similarity matrix.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
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

/**
 * @brief Performs the SymNMF factorization.
 *
 * @param basis The initial basis matrix H, which will be updated in place.
 * @param normalized The normalized similarity matrix W.
 * @param k The number of clusters.
 * @param epsilon The convergence threshold.
 * @param max_iter The maximum number of iterations.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
int symnmf_factorize(matrix_t *basis, const matrix_t *normalized, size_t k, double epsilon, size_t max_iter) {
    matrix_t *numerator = NULL, *denominator = NULL, *gram = NULL, *next = NULL;
    size_t n = 0, iter; double delta = 0.0; const double beta = 0.5; int status = SYM_FAILURE;
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

static double squared_distance(const matrix_t *points, size_t lhs, size_t rhs) {
    size_t dim;
    double acc = 0.0;

    for (dim = 0; dim < points->cols; ++dim) {
        double diff = MAT_AT(points, lhs, dim) - MAT_AT(points, rhs, dim);
        acc += diff * diff;
    }

    return (acc < 0.0) ? 0.0 : acc;
}

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
    if (k <= 1 || k >= n) {
        return SYM_FAILURE;
    }
    if (basis->rows != n || basis->cols != k) {
        return SYM_FAILURE;
    }

    *n_out = n;
    return SYM_SUCCESS;
}

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
