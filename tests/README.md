This directory gathers all test harnesses, fixtures, and reference baselines used by the project.

- harness/ — local scripts and runtime assets such as `compareTester.sh` and Valgrind suppressions.
- external/ — third-party testers shipped by the course staff (e.g., TAU harnesses).
- legacy/Prev_final_100 — the reference implementation used for regression comparisons.

Available harnesses:
- `python3 tests/tester.py` — runs the project’s main randomized integration suite with optional Valgrind checks.
- `bash tests/external/project-tests-v8-27fca/run_tests.sh` — executes the official TA matrix of deterministic fixtures (set `SPEED` for subsets).
- `bash tests/harness/compareTester.sh` — compares current sources against `tests/legacy/Prev_final_100` while compiling and running Valgrind on both.
