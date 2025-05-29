from setuptools import setup, find_packages

setup(
    name="blape",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        numpy, scipy, pybaselines, tqdm, scikit-learn
    ],
    author="Juno Hwang",
    author_email="wnsdh10@snu.ac.kr",
    description="BLaPE(Blurred-Laplacian Peak Extraction)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/snu-heritage/blape-sers",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
) 