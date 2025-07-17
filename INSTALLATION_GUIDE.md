# Installation Guide - Marketing Analytics Toolkit

This guide helps you install the MarketingAI toolkit with the right dependencies for your system.

## 🐍 Python Version Compatibility

**Recommended Python versions:**
- **Python 3.9-3.11**: Full compatibility with all features
- **Python 3.8**: Compatible (some packages may need older versions)
- **Python 3.12+**: Most features work (may need newer package versions)

## 📦 Installation Options

### Option 1: Core Installation (Recommended First Step)

Install the core dependencies that work across all Python versions:

```bash
# Install core requirements
pip install -r requirements.txt
```

This installs:
- NumPy, Pandas, Matplotlib, Scipy, Scikit-learn
- Statsmodels, Prophet
- Seaborn, Plotly, NetworkX
- Jupyter notebooks
- Lifetimes (for CLV models)

### Option 2: Full Installation with Optional Dependencies

After the core installation, add optional features based on your needs:

#### For Deep Learning (LSTM Time Series Models)

**Python 3.8-3.11:**
```bash
pip install tensorflow>=2.8.0,<2.16.0
```

**Python 3.12+:**
```bash
pip install tensorflow>=2.16.0
```

**CPU-only version (lighter install):**
```bash
pip install tensorflow-cpu
```

#### For Bayesian Media Mix Models (PyMC)

```bash
pip install pymc>=5.0.0 arviz>=0.12.0 xarray>=2023.1.0
```

If PyMC installation fails, try the conda approach:
```bash
conda install -c conda-forge pymc arviz
```

#### Alternative Lightweight Bayesian Tools

If PyMC doesn't work on your system:
```bash
pip install emcee>=3.1.0 corner>=2.2.0
```

## 🚨 Troubleshooting Common Issues

### TensorFlow Installation Issues

**Error: "No matching distribution found for tensorflow"**

**Solution 1 - Check Python version:**
```bash
python --version
```

**Solution 2 - Try CPU-only version:**
```bash
pip install tensorflow-cpu
```

**Solution 3 - Use conda:**
```bash
conda install tensorflow
```

### PyMC Installation Issues

**Error: "Failed building wheel for pymc"**

**Solution 1 - Use conda:**
```bash
conda install -c conda-forge pymc
```

**Solution 2 - Install build dependencies:**
```bash
pip install cython numpy
pip install pymc
```

**Solution 3 - Use alternative:**
```bash
pip install emcee corner  # Lightweight MCMC alternative
```

### Apple Silicon (M1/M2) Mac Issues

**For TensorFlow:**
```bash
# Install Apple's optimized version
pip install tensorflow-macos
pip install tensorflow-metal  # For GPU acceleration
```

**For PyMC:**
```bash
conda install -c conda-forge pymc
```

### Windows Issues

**For PyMC/TensorFlow:**
```bash
# Use conda for better Windows compatibility
conda install -c conda-forge pymc tensorflow
```

## 🔧 Feature-Specific Installation

### If you only need specific models:

#### Bass Diffusion Models Only:
```bash
pip install numpy pandas matplotlib scipy scikit-learn networkx
```

#### Customer Lifetime Value Only:
```bash
pip install numpy pandas matplotlib scipy scikit-learn lifetimes
```

#### Time Series Models Only:
```bash
pip install numpy pandas matplotlib scipy scikit-learn prophet statsmodels
# Optional: pip install tensorflow  # for LSTM models
```

#### Price Elasticity Models Only:
```bash
pip install numpy pandas matplotlib scipy scikit-learn statsmodels cvxpy
```

#### Media Mix Models Only:
```bash
pip install numpy pandas matplotlib scipy scikit-learn statsmodels
# Optional: pip install pymc arviz  # for Bayesian MMM
```

## ✅ Verification

Test your installation by running:

```python
# Test core functionality
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import sklearn
import statsmodels.api as sm
import networkx as nx

# Test optional packages (if installed)
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed (optional)")

try:
    import pymc as pm
    print(f"PyMC version: {pm.__version__}")
except ImportError:
    print("PyMC not installed (optional)")

try:
    import prophet
    print("Prophet installed successfully")
except ImportError:
    print("Prophet installation issue")

print("Core installation verification complete!")
```

## 🚀 Quick Start

Once installed, try running a demo:

```python
# Run Bass diffusion model demo
python customer-analytics/bass-diffusion/bass_model.py

# Run Norton-Bass network model demo
python customer-analytics/bass-diffusion/norton_bass_network.py

# Open Jupyter notebook for interactive exploration
jupyter notebook notebooks/marketing_analytics_demo.ipynb
```

## 📝 Environment Setup Recommendations

### Using Virtual Environments (Recommended)

```bash
# Create virtual environment
python -m venv marketing_ai_env

# Activate it
# On Windows:
marketing_ai_env\Scripts\activate
# On Mac/Linux:
source marketing_ai_env/bin/activate

# Install packages
pip install -r requirements.txt
```

### Using Conda (Alternative)

```bash
# Create conda environment
conda create -n marketing_ai python=3.10

# Activate it
conda activate marketing_ai

# Install packages
pip install -r requirements.txt
conda install -c conda-forge pymc tensorflow
```

## 🔄 Updating Dependencies

Keep your installation up to date:

```bash
# Update core packages
pip install --upgrade -r requirements.txt

# Update optional packages
pip install --upgrade tensorflow pymc arviz
```

## 💡 Performance Tips

1. **Use conda for PyMC**: Generally more reliable than pip
2. **Install TensorFlow-CPU**: If you don't need GPU acceleration
3. **Use virtual environments**: Avoid package conflicts
4. **Check Python version**: Some packages work better with specific versions

## 🆘 Getting Help

If you're still having issues:

1. **Check Python version compatibility**
2. **Try conda instead of pip** for problematic packages
3. **Use CPU-only versions** of heavy packages
4. **Install packages one by one** to identify the problematic one
5. **Create a fresh virtual environment** if all else fails

## 📚 Package Documentation

- [TensorFlow Installation](https://www.tensorflow.org/install)
- [PyMC Installation](https://www.pymc.io/projects/docs/en/stable/installation.html)
- [Prophet Installation](https://facebook.github.io/prophet/docs/installation.html)
- [NetworkX Documentation](https://networkx.org/documentation/stable/install.html) 