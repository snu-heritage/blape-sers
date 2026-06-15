# BLaPE (Blurred-Laplacian Peak Extraction)

**🚀 Quick Start: Check out our [blape-demo.ipynb](./blape-demo.ipynb) for a comprehensive demonstration!**

A Python package for SERS (surface-enhanced Raman spectroscopy) signal preprocessing and analysis using the BLaPE (blurred-Laplacian peak extraction) algorithm. This package bundles a ready-to-use sample dataset, fast BLaPE algorithm application, and evaluation pipelines for multiclass classification.

## Installation

### From GitHub (Recommended)

```bash
pip install git+https://github.com/snu-heritage/blape-sers.git
```

## Data

This repository ships a small, **label-balanced sample dataset** bundled inside the
package (`blape/sample_data/`), so the demo and examples run out of the box with no
external download. It contains a subset of the full study data — 24 sample codes
covering all 6 base materials, all 12 dyes, all 5 mordants, and all 3 aging
conditions (6 spectra per code) — stored as `*.csv` files (one wavenumber column
plus spectra columns) under `raw/` and `baseline_removed/`.

The sample data loads automatically via `blape.read_data()`. To use your own
dataset instead, pass a directory that contains `raw/` and/or `baseline_removed/`
subfolders: `blape.read_data(path='your_data')`.

## Quick Usage

### Basic BLaPE Processing

```python
import blape
import numpy as np

# Load the bundled sample data
data = blape.read_data()

# Get common wavenumber range
target_wn = blape.get_common_wavenumber_range(data)

# Apply BLaPE algorithm
for code, d in data.items():
    d['blape'] = blape.blape(d['signal'], original_wn=d['wavenumbers'], target_wn=target_wn)
```

### Multilabel Classification

```python
# Prepare data for machine learning
X, y_dict, label_encoders = blape.prepare_multilabel_data(data, feature_key='blape')

# Train multilabel models
models, X_train, X_test, y_train_dict, y_test_dict = blape.train_multilabel_models(
    X, y_dict, test_size=0.2, random_state=42
)

# Evaluate models
results = blape.evaluate_multilabel_models(models, X_test, y_test_dict, label_encoders)
```

### Custom Sigma Values

```python
# Apply BLaPE with custom sigma parameter
enhanced_signal = blape.blape(signal, original_wn=wavenumbers, target_wn=target_wn, sigma=30)
```

## Project Structure

```
blape/
├── blape/
│   ├── __init__.py          # Main package with core functions
│   ├── evaluation.py        # Model training and evaluation utilities
│   ├── downloader.py        # Bundled sample-data resolver
│   └── sample_data/         # Bundled SERS sample dataset (raw/ & baseline_removed/)
├── scripts/
│   └── make_sample_data.py  # How the bundled sample data was generated
├── blape-demo.ipynb         # Comprehensive demonstration notebook
├── test.py                  # Example usage and testing script
├── setup.py                 # Package setup configuration
└── requirements-dev.txt     # Development dependencies
```

## Contact

**Juno Hwang**  
Seoul National University  
Department of Science Education (Physics Major)  
Data Science Lab  
PhD Candidate  
Email: wnsdh10@snu.ac.kr