#ifndef SYM_NMF_H
#define SYM_NMF_H

#include <stdlib.h>

#define SYM_SUCCESS 0
#define SYM_FAILURE 1

/**
 * @brief Represents a 2D matrix of doubles.
 */
typedef struct {
    size_t rows;    /**< Number of rows in the matrix. */
    size_t cols;    /**< Number of columns in the matrix. */
    double *data;   /**< Pointer to the matrix data, stored in row-major order. */
} matrix_t;

/**
 * @brief Macro to access an element at a specific row and column in a matrix.
 * @param m Pointer to the matrix.
 * @param r The row index.
 * @param c The column index.
 */
#define MAT_AT(m, r, c) ((m)->data[(r) * (m)->cols + (c)])

/**
 * @brief Creates a new matrix with the given dimensions.
 * @param rows The number of rows.
 * @param cols The number of columns.
 * @return A pointer to the newly created matrix, or NULL if an error occurred.
 */
matrix_t *matrix_create(size_t rows, size_t cols);

/**
 * @brief Frees the memory allocated for a matrix.
 * @param matrix A pointer to the matrix to be freed.
 */
void matrix_free(matrix_t *matrix);

/**
 * @brief Prints a generic error message to stdout.
 */
void symnmf_log_error(void);

/**
 * @brief Reads a dataset from a file into a matrix.
 * @param path The path to the input file.
 * @param out_matrix A pointer to a matrix pointer that will be updated to point to the new matrix.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
int dataset_from_file(const char *path, matrix_t **out_matrix);

/**
 * @brief Computes the similarity matrix from a set of points.
 * @param points A matrix where each row is a data point.
 * @param similarity The output similarity matrix.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
int compute_similarity(const matrix_t *points, matrix_t *similarity);

/**
 * @brief Computes the degree matrix from a similarity matrix.
 * @param similarity The similarity matrix.
 * @param degree The output degree matrix.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
int compute_degree(const matrix_t *similarity, matrix_t *degree);

/**
 * @brief Computes the normalized similarity matrix.
 * @param similarity The similarity matrix.
 * @param degree The degree matrix.
 * @param normalized The output normalized similarity matrix.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
int compute_normalized(const matrix_t *similarity, const matrix_t *degree, matrix_t *normalized);

/**
 * @brief Performs the SymNMF factorization.
 * @param basis The initial basis matrix H, which will be updated in place.
 * @param normalized The normalized similarity matrix W.
 * @param k The number of clusters.
 * @param epsilon The convergence threshold.
 * @param max_iter The maximum number of iterations.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
int symnmf_factorize(matrix_t *basis, const matrix_t *normalized, size_t k, double epsilon, size_t max_iter);

/**
 * @brief Main function for the command-line interface.
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @return SYM_SUCCESS on success, SYM_FAILURE on failure.
 */
int symnmf_cli_main(int argc, char **argv);

#endif /* SYM_NMF_H */
