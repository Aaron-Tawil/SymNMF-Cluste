# SymNMF Clustering

This project provides a high-performance implementation of the Symmetric Non-negative Matrix Factorization (SymNMF) algorithm for data clustering. It includes a command-line interface to execute the algorithm and its intermediate steps, as well as an analysis tool to compare its performance against K-Means.

The core logic is written in C for efficiency, wrapped in a Python C extension for ease of use.

## About The Project

The goal of SymNMF is to find a low-rank approximation of a normalized similarity matrix derived from a dataset. This approximation is then used to assign data points to clusters. This project implements the full algorithm, from calculating the initial similarity matrix to the final clustering assignment.

Key components:
- **`symnmf.py`**: A Python CLI for running the SymNMF algorithm and its sub-steps.
- **`analysis.py`**: A Python script to compare the clustering quality of SymNMF and a basic K-Means implementation using the silhouette score.
- **`symnmf.c` / `symnmf_c` module**: A C implementation of the core matrix operations for high performance, exposed to Python via a C extension.

## Getting Started

Follow these instructions to get the project built and ready to run.

### Prerequisites

- Python 3.8+
- `numpy`
- A C compiler (e.g., `gcc`)
- `make`

You can install the required Python package using pip:
```sh
pip install numpy
```

### Installation

The core of the project is a C extension that must be compiled first.

1.  **Build the C extension:**
    Run the following command from the project root directory. This will compile the C source files and create a `symnmf_c.so` file that the Python scripts can import.
    ```sh
    python3 setup.py build_ext --inplace
    ```

2.  **Build the C executable (Optional):**
    The project also includes a standalone C executable that can be built using the `Makefile`.
    ```sh
    make
    ```
    This will create an executable file named `symnmf` in the root directory.

## Usage

Once the C extension is built, you can use the Python scripts to perform clustering and analysis.

### Main Program (`symnmf.py`)

This script is the main interface for the SymNMF algorithm.

**Synopsis:**
```sh
python3 symnmf.py <k> <goal> <file_name>
```

**Arguments:**
- `k`: The desired number of clusters (integer).
- `goal`: The computation to perform:
    - `sym`: Compute and output the similarity matrix.
    - `ddg`: Compute and output the diagonal degree matrix.
    - `norm`: Compute and output the normalized similarity matrix.
    - `symnmf`: Perform the full SymNMF clustering and output the final `H` matrix.
- `file_name`: The path to the input data file.

**Example:**
```sh
python3 symnmf.py 3 symnmf data/input_1.txt
```

### Analysis Program (`analysis.py`)

This script runs both SymNMF and K-Means on a dataset and prints their respective silhouette scores to compare clustering quality.

**Synopsis:**
```sh
python3 analysis.py <k> <file_name>
```

**Arguments:**
- `k`: The desired number of clusters (integer).
- `file_name`: The path to the input data file.

**Example:**
```sh
python3 analysis.py 3 data/input_1.txt
```

## Running Tests

This project includes three test suites to ensure correctness, performance, and memory safety.

### 1. Main Tester (`tester.py`)

The `tester.py` script is the primary tool for verifying the implementation. It runs a series of randomized trials against the C implementation, the Python implementation, and the `symnmf_c` extension module. It also uses `valgrind` to check for memory leaks.

**To run the main tester:**

1.  **Build the C executable and Python extension:**
    ```sh
    make
    python3 setup.py build_ext --inplace
    ```

2.  **Run the tester script:**
    ```sh
    python3 tester.py
    ```
    The script will print a detailed summary of successes and failures for each goal (`sym`, `ddg`, `norm`, `symnmf`) across the different implementations.

### 2. Regression Tester (`compareTester.sh`)

The `tests/harness/compareTester.sh` script is a regression testing tool. It compares the output of the current project implementation against a legacy version (`Prev_final_100`) to ensure that changes have not introduced regressions. It tests the C executable, the Python scripts, and the analysis script, and also checks for memory leaks with `valgrind`.

**To run the regression tester:**

1.  **Navigate to the harness directory:**
    ```sh
    cd tests/harness/
    ```

2.  **Execute the test script:**
    ```sh
    ./compareTester.sh
    ```
    The script will compile both the current and legacy projects and run a series of comparisons on generated data, reporting any differences.

### 3. External Test Suite

The project also comes with an external test suite located in the `tests/external/project-tests-v8-27fca` directory. This suite uses a predefined set of inputs and expected outputs.

**To run the external tests:**

1.  **Ensure Correct Directory Structure:** The test script expects all project source files (`symnmf.py`, `analysis.py`, `setup.py`, `Makefile`, `src/`, `include/`, etc.) to be located inside the `a_b_project/` directory within the test suite. You may need to copy the files into this subdirectory.

2.  **Navigate to the Test Directory:**
    ```sh
    cd tests/external/project-tests-v8-27fca/
    ```

3.  **Execute the Test Script:**
    The `run_tests.sh` script will build the code, run tests against both Python and C implementations, and use `valgrind`.
    ```sh
    ./run_tests.sh
    ```
    You can pass arguments like `slow`, `quick`, or `edge` to run different subsets of tests.

## Project Structure

```
.
├── Makefile              # Compiles the standalone C executable
├── README.md             # This file
├── analysis.py           # Compares SymNMF and K-Means clustering
├── setup.py              # Builds the Python C extension
├── symnmf.py             # Main Python CLI for SymNMF
├── symnmf.c              # Standalone C executable source
├── symnmfmodule.c        # Python C extension wrapper
├── include/
│   └── symnmf.h          # Header for the C code
├── src/
│   ├── matrix_ops.c      # C implementation of matrix operations
│   └── symnmf_algo.c     # C implementation of the SymNMF algorithm
└── project-tests-v8-27fca/
    ├── run_tests.sh      # The main test script
    ├── a_b_project/      # The expected location for source files
    └── tests/            # Input/output files for the test cases
```
