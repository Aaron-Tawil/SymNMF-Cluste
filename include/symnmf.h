#ifndef SYM_NMF_H
#define SYM_NMF_H

#include <stdlib.h>

#define SYM_SUCCESS 0
#define SYM_FAILURE 1

typedef struct {
    size_t rows;
    size_t cols;
    double *data;
} matrix_t;

#define MAT_AT(m, r, c) ((m)->data[(r) * (m)->cols + (c)])

matrix_t *matrix_create(size_t rows, size_t cols);
void matrix_free(matrix_t *matrix);
void symnmf_log_error(void);

int dataset_from_file(const char *path, matrix_t **out_matrix);

int compute_similarity(const matrix_t *points, matrix_t *similarity);
int compute_degree(const matrix_t *similarity, matrix_t *degree);
int compute_normalized(const matrix_t *similarity, const matrix_t *degree, matrix_t *normalized);
int symnmf_factorize(matrix_t *basis, const matrix_t *normalized, size_t k, double epsilon, size_t max_iter);

int symnmf_cli_main(int argc, char **argv);

#endif /* SYM_NMF_H */
