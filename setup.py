"""Build script for the SymNMF Python extension."""

from __future__ import annotations

from setuptools import Extension, setup

try:
    import numpy as _np
except ImportError as exc:  # pragma: no cover - numpy is a tester dependency
    raise SystemExit("numpy is required to build symnmf_c") from exc

EXTENSION_NAME = "symnmf_c"

symnmf_extension = Extension(
    EXTENSION_NAME,
    sources=[
        "symnmfmodule.c",
        "symnmf_algo.c",
        "matrix_ops.c",
    ],
    include_dirs=[_np.get_include()],
)

if __name__ == "__main__":
    setup(name="symnmf", version="0.1.0", ext_modules=[symnmf_extension])
