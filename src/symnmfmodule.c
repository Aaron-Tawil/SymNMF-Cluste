#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

#include "symnmf.h"

#define SYM_DEFAULT_EPSILON 1e-4
#define SYM_DEFAULT_MAX_ITER 300

typedef enum {
    GOAL_SYM,
    GOAL_DDG,
    GOAL_NORM
} simple_goal_t;

static int pyobject_to_matrix(PyObject *obj, matrix_t **matrix_out);
static PyObject *matrix_to_pyobject(const matrix_t *matrix);
static PyObject *run_simple_goal(PyObject *points_obj, simple_goal_t goal);

/* Python wrapper for the similarity goal. */
static PyObject *sym(PyObject *self, PyObject *args) {
    PyObject *points_obj;
    (void)self;

    if (!PyArg_ParseTuple(args, "O:sym", &points_obj)) {
        return NULL;
    }

    return run_simple_goal(points_obj, GOAL_SYM);
}

/* Python wrapper for the degree goal. */
static PyObject *ddg(PyObject *self, PyObject *args) {
    PyObject *points_obj;
    (void)self;

    if (!PyArg_ParseTuple(args, "O:ddg", &points_obj)) {
        return NULL;
    }

    return run_simple_goal(points_obj, GOAL_DDG);
}

/* Python wrapper for the normalized goal. */
static PyObject *norm(PyObject *self, PyObject *args) {
    PyObject *points_obj;
    (void)self;

    if (!PyArg_ParseTuple(args, "O:norm", &points_obj)) {
        return NULL;
    }

    return run_simple_goal(points_obj, GOAL_NORM);
}

/* Run the SymNMF factorization entry point exposed to Python. */
static PyObject *symnmf(PyObject *self, PyObject *args) {
    PyObject *basis_obj;
    PyObject *normalized_obj;
    matrix_t *basis = NULL;
    matrix_t *normalized = NULL;
    PyObject *result = NULL;
    size_t k;

    (void)self;
    if (!PyArg_ParseTuple(args, "OO:symnmf", &basis_obj, &normalized_obj)) { return NULL; }
    if (pyobject_to_matrix(basis_obj, &basis) != 0) { goto cleanup; }
    if (pyobject_to_matrix(normalized_obj, &normalized) != 0) { goto cleanup; }
    if (normalized->rows != normalized->cols) { PyErr_SetString(PyExc_ValueError, "normalized matrix must be square"); goto cleanup; }
    if (basis->rows != normalized->rows) { PyErr_SetString(PyExc_ValueError, "basis and normalized dimensions differ"); goto cleanup; }

    k = basis->cols;
    if (k == 0) { PyErr_SetString(PyExc_ValueError, "basis must have at least one column"); goto cleanup; }
    if (symnmf_factorize(basis, normalized, k, SYM_DEFAULT_EPSILON, SYM_DEFAULT_MAX_ITER) != SYM_SUCCESS) { PyErr_SetString(PyExc_RuntimeError, "symnmf factorization failed"); goto cleanup; }

    result = matrix_to_pyobject(basis);

cleanup:
    matrix_free(normalized);
    matrix_free(basis);
    return result;
}

static PyMethodDef SymNMFMethods[] = {
    {"sym", sym, METH_VARARGS, "Compute similarity matrix"},
    {"ddg", ddg, METH_VARARGS, "Compute degree matrix"},
    {"norm", norm, METH_VARARGS, "Compute normalized similarity"},
    {"symnmf", symnmf, METH_VARARGS, "Run SymNMF factorization"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf_c",
    "Symmetric NMF routines implemented in C",
    -1,
    SymNMFMethods,
};

/* Module initialization hook for the CPython runtime. */
PyMODINIT_FUNC PyInit_symnmf_c(void) {
    return PyModule_Create(&symnmfmodule);
}

/* Convert a NumPy-like object into a freshly allocated matrix_t. */
static int pyobject_to_matrix(PyObject *obj, matrix_t **matrix_out) {
    Py_buffer view;
    matrix_t *matrix = NULL;
    size_t rows = 0, cols = 0, expected = 0;

    if (matrix_out == NULL) { PyErr_SetString(PyExc_RuntimeError, "internal error: matrix_out is NULL"); return -1; }
    if (PyObject_GetBuffer(obj, &view, PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) != 0) { PyErr_SetString(PyExc_TypeError, "expected a contiguous float64 array"); return -1; }

    rows = (size_t)view.shape[0];
    cols = (size_t)view.shape[1];
    if (view.ndim != 2 || rows == 0 || cols == 0) { PyErr_SetString(PyExc_ValueError, "expected a non-empty 2D array"); PyBuffer_Release(&view); return -1; }
    if (view.itemsize != (Py_ssize_t)sizeof(double)) { PyErr_SetString(PyExc_TypeError, "expected elements of type float64"); PyBuffer_Release(&view); return -1; }

    expected = rows * cols;
    if (expected > ((size_t)view.len / sizeof(double))) { PyErr_SetString(PyExc_ValueError, "buffer length mismatch"); PyBuffer_Release(&view); return -1; }

    matrix = matrix_create(rows, cols);
    if (matrix == NULL || matrix->data == NULL) { PyErr_NoMemory(); PyBuffer_Release(&view); matrix_free(matrix); return -1; }

    memcpy(matrix->data, view.buf, (size_t)view.len);
    PyBuffer_Release(&view);
    *matrix_out = matrix;
    return 0;
}

/* Convert a matrix_t into a list-of-lists Python object. */
static PyObject *matrix_to_pyobject(const matrix_t *matrix) {
    size_t row;
    size_t col;
    PyObject *rows = PyList_New(matrix->rows);

    if (rows == NULL) {
        return NULL;
    }

    for (row = 0; row < matrix->rows; ++row) {
        PyObject *row_obj = PyList_New(matrix->cols);
        if (row_obj == NULL) {
            Py_DECREF(rows);
            return NULL;
        }

        for (col = 0; col < matrix->cols; ++col) {
            PyObject *value = PyFloat_FromDouble(MAT_AT(matrix, row, col));
            if (value == NULL) {
                Py_DECREF(row_obj);
                Py_DECREF(rows);
                return NULL;
            }
            PyList_SET_ITEM(row_obj, col, value);
        }

        PyList_SET_ITEM(rows, row, row_obj);
    }

    return rows;
}

/* Shared implementation for the sym, ddg, and norm Python wrappers. */
static PyObject *run_simple_goal(PyObject *points_obj, simple_goal_t goal) {
    matrix_t *points = NULL;
    matrix_t *similarity = NULL;
    matrix_t *degree = NULL;
    matrix_t *normalized = NULL;
    PyObject *result = NULL;
    size_t n;

    if (pyobject_to_matrix(points_obj, &points) != 0) { goto cleanup; }

    n = points->rows;
    similarity = matrix_create(n, n);
    if (similarity == NULL || similarity->data == NULL) { PyErr_NoMemory(); goto cleanup; }
    if (compute_similarity(points, similarity) != SYM_SUCCESS) { PyErr_SetString(PyExc_RuntimeError, "failed to compute similarity matrix"); goto cleanup; }

    if (goal == GOAL_SYM) { result = matrix_to_pyobject(similarity); goto cleanup; }

    degree = matrix_create(n, n);
    if (degree == NULL || degree->data == NULL) { PyErr_NoMemory(); goto cleanup; }
    if (compute_degree(similarity, degree) != SYM_SUCCESS) { PyErr_SetString(PyExc_RuntimeError, "failed to compute degree matrix"); goto cleanup; }

    if (goal == GOAL_DDG) { result = matrix_to_pyobject(degree); goto cleanup; }

    normalized = matrix_create(n, n);
    if (normalized == NULL || normalized->data == NULL) { PyErr_NoMemory(); goto cleanup; }
    if (compute_normalized(similarity, degree, normalized) != SYM_SUCCESS) { PyErr_SetString(PyExc_RuntimeError, "failed to compute normalized matrix"); goto cleanup; }

    result = matrix_to_pyobject(normalized);

cleanup:
    matrix_free(normalized);
    matrix_free(degree);
    matrix_free(similarity);
    matrix_free(points);
    return result;
}
