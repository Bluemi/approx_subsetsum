from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "approx_subsetsum",
        ["src/approx_subsetsum.cpp"],
    ),
]

setup(
    name="approx_subsetsum",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
