from pathlib import Path

from setuptools import setup, find_packages

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="blape",
    version="1.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "blape": [
            "sample_data/raw/*.csv",
            "sample_data/baseline_removed/*.csv",
        ],
    },
    install_requires=[
        "numpy", "scipy", "pandas", "pybaselines", "tqdm", "scikit-learn"
    ],
    author="Juno Hwang",
    author_email="wnsdh10@snu.ac.kr",
    description="BLaPE(Blurred-Laplacian Peak Extraction)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/snu-heritage/blape-sers",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
) 
