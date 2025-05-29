# BLaPE (Blurred-Laplacian Peak Extraction)

**🚀 Quick Start: Check out our [blape-demo.ipynb](./blape-demo.ipynb) for a comprehensive demonstration!**

**☁️ Try it online: [Run the demo directly in Google Colab](https://colab.research.google.com/drive/1txVTaIwoqdt0b7VSS80Vj45Vv4nStDQR) without any setup!**

A Python package for SERS (surface-enhanced Raman spectroscopy) signal preprocessing and analysis using the BLaPE (blurred-Laplacian peak extraction) algorithm. This package provides easy data download, fast BLaPE algorithm application, and evaluation pipelines for multiclass classification.

## Installation

### From GitHub (Recommended)

```bash
pip install git+https://github.com/snu-heritage/blape-sers.git
```

## Data

📥 [**[Link] SERS of Korean Traditional Organic Dye-Dyed Samples According to Aging Process**](https://zenodo.org/records/15487399)

By default, the data is downloaded to the `data/` folder in the current directory. After the download is complete, it is recommended to check that `*.csv` files have been created in that path.

Our data is provided through [**Zenodo**](https://zenodo.org/records/15487399), so for faster download, you can download the same data through [**our GoogleDrive folder**](https://drive.google.com/drive/folders/1o4CAkfUIpgeqJb1EIK4ruCsm3VMHinA5?usp=drive_link).

## Quick Usage

### Basic BLaPE Processing

```python
import blape
import numpy as np

# Download and load sample data
blape.download_data(path='data')
data = blape.read_data(path='data')

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
│   └── downloader.py        # Data download functionality
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