# Bayesian Media Mix Model Setup Guide

This guide helps you set up and run the advanced Bayesian Media Mix Models using PyMC.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core requirements
pip install -r requirements.txt

# Additional Bayesian dependencies
pip install pymc arviz
```

### 2. Verify Installation

```python
import pymc as pm
import arviz as az
print("PyMC version:", pm.__version__)
print("ArviZ version:", az.__version__)
```

### 3. Run Bayesian MMM

```python
from customer_analytics.media_mix_models.media_mix_models import BayesianMMM, generate_mmm_data

# Generate sample data
data, _, _, _ = generate_mmm_data(n_periods=104, n_channels=4)
media_channels = [col for col in data.columns if col.startswith('Channel_')]

# Initialize and fit Bayesian MMM
bayesian_mmm = BayesianMMM(channels=media_channels)
bayesian_mmm.fit(
    media_data=data[media_channels],
    target=data['sales'],
    draws=1000,  # Increase for production
    tune=1000
)

# Analyze results
bayesian_mmm.summary()
contributions = bayesian_mmm.get_channel_contributions()
print(contributions)
```

## 🔧 Model Configuration

### MCMC Settings

```python
# For exploration (fast)
draws=500, tune=500

# For production (robust)
draws=2000, tune=2000, target_accept=0.95

# For complex models
draws=3000, tune=3000, target_accept=0.99, max_treedepth=15
```

### Prior Specifications

The model uses informed priors based on marketing knowledge:

- **Adstock**: `Beta(2, 1)` - favors moderate carryover effects
- **Saturation**: `Gamma(2, 1)` - allows flexible S-curves
- **Media Coefficients**: `Gamma(1, 1)` - ensures positive effects
- **Base Sales**: `Normal(0, 1)` - on scaled data

### Custom Priors

```python
# Example: Modify priors in the model code
adstock_rate = pm.Beta('adstock_rate', alpha=3, beta=2, shape=n_channels)  # Higher alpha = more carryover
saturation_shape = pm.Gamma('saturation_shape', alpha=3, beta=1, shape=n_channels)  # More S-shaped curves
```

## 📊 Model Diagnostics

### Convergence Checking

```python
# Plot diagnostics
bayesian_mmm.plot_diagnostics()

# Check R-hat values (should be < 1.01)
summary = az.summary(bayesian_mmm.trace)
print("R-hat range:", summary['r_hat'].min(), "-", summary['r_hat'].max())

# Check effective sample size (should be > 400)
print("ESS range:", summary['ess_bulk'].min(), "-", summary['ess_bulk'].max())
```

### Model Validation

```python
# Posterior predictive checks
predictions = bayesian_mmm.predict(media_data, n_samples=1000)
print("Prediction uncertainty:", predictions['std'].mean())

# Channel credibility
contributions = bayesian_mmm.get_channel_contributions()
print("Channels with >95% positive probability:")
print(contributions[contributions['prob_positive'] > 0.95])
```

## 💡 Best Practices

### Data Preparation

1. **Scale consistently**: Use same scaling for train/test
2. **Handle seasonality**: Include time controls or seasonal dummies
3. **Check data quality**: Remove outliers and missing values
4. **Sufficient data**: Minimum 52 weeks for weekly data

### Model Tuning

1. **Start simple**: Begin with default priors
2. **Increase draws**: More samples = better convergence
3. **Check chains**: Multiple chains should converge to same distribution
4. **Validate out-of-sample**: Always test on holdout data

### Interpretation

1. **Use credible intervals**: Report uncertainty, not just point estimates
2. **Check probability of effect**: `prob_positive` for significance
3. **Compare to business knowledge**: Validate against intuition
4. **Test incrementality**: Use model insights to design experiments

## 🚨 Troubleshooting

### Common Issues

**Divergent transitions**
```
Solution: Increase target_accept to 0.95-0.99
```

**Low effective sample size**
```
Solution: Increase draws or tune steps
```

**Poor convergence (R-hat > 1.01)**
```
Solution: 
- Increase tuning steps
- Use more chains
- Simplify model or priors
```

**Long sampling time**
```
Solution:
- Use fewer draws for exploration
- Scale data properly
- Use more informative priors
```

### Performance Optimization

```python
# Use multiple cores
import os
os.environ["OMP_NUM_THREADS"] = "4"

# Reduce draws for testing
draws=500, tune=500

# Use progress bar
pm.sample(..., progressbar=True)
```

## 📈 Advanced Features

### Hierarchical Modeling

```python
# Group channels by type (paid/organic)
channel_types = {'Channel_1': 0, 'Channel_2': 0, 'Channel_3': 1, 'Channel_4': 1}

# Add hierarchical structure to priors
type_effect = pm.Normal('type_effect', mu=0, sigma=0.5, shape=2)
channel_effect = pm.Normal('channel_effect', 
                          mu=type_effect[channel_type_idx], 
                          sigma=0.3, shape=n_channels)
```

### Model Comparison

```python
# Compare models using WAIC
waic = az.waic(bayesian_mmm.trace)
print(f"WAIC: {waic.waic:.2f}")

# Compare different models
models = {'simple': trace1, 'complex': trace2}
comparison = az.compare(models)
print(comparison)
```

### Custom Transformations

```python
# Custom adstock function
def custom_adstock(x, rate, power):
    # Power adstock instead of geometric
    pass

# Include in model
adstocked = custom_adstock(media_scaled, adstock_rate, adstock_power)
```

## 🎯 Production Deployment

### Model Artifacts

```python
# Save model and trace
import pickle

# Save trace
with open('mmm_trace.pkl', 'wb') as f:
    pickle.dump(bayesian_mmm.trace, f)

# Save scalers
with open('scalers.pkl', 'wb') as f:
    pickle.dump({
        'media_scaler': bayesian_mmm.media_scaler,
        'target_scaler': bayesian_mmm.target_scaler
    }, f)
```

### Automated Reporting

```python
def generate_mmm_report(bayesian_mmm, output_path):
    """Generate automated MMM report"""
    
    # Model diagnostics
    diagnostics = az.summary(bayesian_mmm.trace)
    
    # Channel contributions
    contributions = bayesian_mmm.get_channel_contributions()
    
    # Budget optimization
    optimal_budget = bayesian_mmm.optimize_budget_bayesian(total_budget)
    
    # Create report
    report = {
        'model_diagnostics': diagnostics,
        'channel_effects': contributions,
        'budget_recommendation': optimal_budget,
        'timestamp': pd.Timestamp.now()
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(report, f)
```

## 📚 Further Reading

- [PyMC Documentation](https://docs.pymc.io/)
- [ArviZ Documentation](https://arviz-devs.github.io/arviz/)
- [Bayesian Methods for Marketing Mix Modeling](https://research.google/pubs/pub46001/)
- [Media Mix Modeling at Scale](https://engineering.fb.com/2017/01/30/data-infrastructure/media-mix-modeling-at-scale/)

## 💬 Support

For issues specific to the Bayesian MMM implementation:
1. Check model diagnostics first
2. Verify data quality and scaling
3. Try simpler models before complex ones
4. Use the included example scripts as reference 