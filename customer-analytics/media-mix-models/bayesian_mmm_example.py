"""
Comprehensive Bayesian Media Mix Model Example

This example demonstrates advanced Bayesian MMM capabilities including:
- Hierarchical modeling with PyMC
- Uncertainty quantification
- Prior specification
- Model diagnostics
- Contribution analysis
- Bayesian budget optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from media_mix_models import BayesianMMM, generate_mmm_data
import warnings
warnings.filterwarnings('ignore')

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
    print("PyMC and ArviZ successfully imported")
except ImportError as e:
    print(f"PyMC/ArviZ not available: {e}")
    print("Install with: pip install pymc arviz")
    PYMC_AVAILABLE = False


def run_bayesian_mmm_analysis():
    """
    Run comprehensive Bayesian MMM analysis
    """
    if not PYMC_AVAILABLE:
        print("PyMC not available. Cannot run Bayesian analysis.")
        return
    
    print("=== Generating Synthetic Marketing Data ===")
    
    # Generate realistic marketing data
    data, true_adstock, true_saturation, true_coef = generate_mmm_data(
        n_periods=104,  # 2 years of weekly data
        n_channels=4,
        base_sales=50000
    )
    
    print(f"Generated {len(data)} weeks of marketing data")
    print(f"Sales range: ${data['sales'].min():,.0f} - ${data['sales'].max():,.0f}")
    
    # Extract media channels and sales
    media_channels = [col for col in data.columns if col.startswith('Channel_')]
    media_data = data[media_channels]
    sales = data['sales']
    
    # Split into train/validation
    split_idx = int(0.85 * len(data))
    train_media = media_data[:split_idx]
    train_sales = sales[:split_idx]
    val_media = media_data[split_idx:]
    val_sales = sales[split_idx:]
    
    print(f"Training: {len(train_media)} weeks, Validation: {len(val_media)} weeks")
    
    print("\n=== Fitting Bayesian Media Mix Model ===")
    
    # Initialize Bayesian MMM
    bayesian_mmm = BayesianMMM(channels=media_channels)
    
    # Fit the model with proper MCMC settings
    bayesian_mmm.fit(
        media_data=train_media,
        target=train_sales,
        draws=1500,        # Number of posterior samples
        tune=1500,         # Number of tuning steps
        target_accept=0.95, # Higher acceptance rate for better convergence
        max_treedepth=15   # Deeper trees for complex posterior
    )
    
    print("\n=== Model Diagnostics ===")
    
    # Print model summary with convergence diagnostics
    bayesian_mmm.summary()
    
    # Plot MCMC diagnostics
    print("\nPlotting MCMC diagnostics...")
    bayesian_mmm.plot_diagnostics()
    
    print("\n=== Channel Effect Analysis ===")
    
    # Get channel contributions with uncertainty
    contributions = bayesian_mmm.get_channel_contributions(credible_interval=0.95)
    print("\nChannel Contribution Analysis:")
    print(contributions.round(4))
    
    # Plot channel effect distributions
    bayesian_mmm.plot_channel_effects()
    
    print("\n=== Model Predictions ===")
    
    # Generate predictions with uncertainty
    train_predictions = bayesian_mmm.predict(train_media, n_samples=1000)
    val_predictions = bayesian_mmm.predict(val_media, n_samples=1000)
    
    # Calculate prediction accuracy
    train_mape = np.mean(np.abs((train_sales - train_predictions['mean']) / train_sales)) * 100
    val_mape = np.mean(np.abs((val_sales - val_predictions['mean']) / val_sales)) * 100
    
    print(f"Training MAPE: {train_mape:.2f}%")
    print(f"Validation MAPE: {val_mape:.2f}%")
    
    # Plot predictions with uncertainty bands
    plot_predictions_with_uncertainty(
        train_sales, val_sales, 
        train_predictions, val_predictions,
        split_idx
    )
    
    print("\n=== Bayesian Budget Optimization ===")
    
    # Current budget allocation
    current_budget = train_media.sum().sum()
    print(f"Historical total budget: ${current_budget:,.0f}")
    
    # Optimize budget with uncertainty quantification
    optimal_budget = bayesian_mmm.optimize_budget_bayesian(
        total_budget=current_budget,
        periods=52,
        n_samples=1000
    )
    
    print("\nOptimal Budget Allocation with Uncertainty:")
    budget_df = create_budget_comparison_table(train_media, optimal_budget)
    print(budget_df.to_string(index=False))
    
    # Plot budget allocation comparison
    plot_budget_comparison(train_media, optimal_budget)
    
    print("\n=== Scenario Analysis ===")
    
    # Test different budget scenarios
    scenario_analysis(bayesian_mmm, current_budget, media_channels)
    
    print("\n=== Model Insights and Recommendations ===")
    generate_insights_and_recommendations(contributions, optimal_budget, true_coef)


def plot_predictions_with_uncertainty(train_sales, val_sales, train_pred, val_pred, split_idx):
    """
    Plot model predictions with uncertainty bands
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Training predictions
    train_idx = range(len(train_sales))
    ax1.plot(train_idx, train_sales, 'b-', label='Actual Sales', alpha=0.8)
    ax1.plot(train_idx, train_pred['mean'], 'r-', label='Predicted Sales', alpha=0.8)
    ax1.fill_between(train_idx, train_pred['quantile_2.5'], train_pred['quantile_97.5'], 
                     color='red', alpha=0.2, label='95% Credible Interval')
    ax1.set_title('Training Set: Actual vs Predicted Sales')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Sales ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation predictions
    val_idx = range(split_idx, split_idx + len(val_sales))
    ax2.plot(val_idx, val_sales, 'b-', label='Actual Sales', alpha=0.8)
    ax2.plot(val_idx, val_pred['mean'], 'r-', label='Predicted Sales', alpha=0.8)
    ax2.fill_between(val_idx, val_pred['quantile_2.5'], val_pred['quantile_97.5'], 
                     color='red', alpha=0.2, label='95% Credible Interval')
    ax2.set_title('Validation Set: Actual vs Predicted Sales')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Sales ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_budget_comparison_table(historical_media, optimal_budget):
    """
    Create comparison table of historical vs optimal budget allocation
    """
    historical_annual = historical_media.mean() * 52
    
    comparison_data = []
    for channel in historical_media.columns:
        historical_amount = historical_annual[channel]
        optimal_stats = optimal_budget[channel]
        
        comparison_data.append({
            'Channel': channel,
            'Historical_Annual': historical_amount,
            'Optimal_Mean': optimal_stats['mean_allocation'] * 52,
            'Optimal_Lower': optimal_stats['allocation_2.5%'] * 52,
            'Optimal_Upper': optimal_stats['allocation_97.5%'] * 52,
            'Change_$': optimal_stats['mean_allocation'] * 52 - historical_amount,
            'Change_%': ((optimal_stats['mean_allocation'] * 52 - historical_amount) / historical_amount) * 100
        })
    
    return pd.DataFrame(comparison_data).round(0)


def plot_budget_comparison(historical_media, optimal_budget):
    """
    Plot comparison of historical vs optimal budget allocation
    """
    channels = historical_media.columns
    historical_annual = historical_media.mean() * 52
    optimal_mean = [optimal_budget[ch]['mean_allocation'] * 52 for ch in channels]
    optimal_std = [optimal_budget[ch]['std_allocation'] * 52 for ch in channels]
    
    x = np.arange(len(channels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, historical_annual, width, label='Historical', alpha=0.8)
    bars2 = ax.bar(x + width/2, optimal_mean, width, label='Optimal (Mean)', alpha=0.8, yerr=optimal_std)
    
    ax.set_xlabel('Marketing Channel')
    ax.set_ylabel('Annual Budget ($)')
    ax.set_title('Historical vs Optimal Budget Allocation')
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', rotation=45)
    
    plt.tight_layout()
    plt.show()


def scenario_analysis(bayesian_mmm, base_budget, channels):
    """
    Analyze different budget scenarios
    """
    scenarios = {
        'Current': base_budget,
        '+20% Budget': base_budget * 1.2,
        '+50% Budget': base_budget * 1.5,
        '-20% Budget': base_budget * 0.8
    }
    
    print("Budget Scenario Analysis:")
    scenario_results = []
    
    for scenario_name, budget in scenarios.items():
        optimal_allocation = bayesian_mmm.optimize_budget_bayesian(
            total_budget=budget,
            periods=52,
            n_samples=500
        )
        
        total_allocated = sum([stats['mean_allocation'] * 52 for stats in optimal_allocation.values()])
        
        scenario_results.append({
            'Scenario': scenario_name,
            'Total_Budget': budget,
            'Channel_1': optimal_allocation[channels[0]]['mean_allocation'] * 52,
            'Channel_2': optimal_allocation[channels[1]]['mean_allocation'] * 52,
            'Channel_3': optimal_allocation[channels[2]]['mean_allocation'] * 52,
            'Channel_4': optimal_allocation[channels[3]]['mean_allocation'] * 52
        })
    
    scenario_df = pd.DataFrame(scenario_results)
    print(scenario_df.round(0).to_string(index=False))


def generate_insights_and_recommendations(contributions, optimal_budget, true_coefficients):
    """
    Generate business insights and recommendations
    """
    print("=== KEY INSIGHTS ===")
    
    # Channel effectiveness ranking
    mean_coeffs = contributions.sort_values('mean_coeff', ascending=False)
    print(f"\n1. Channel Effectiveness Ranking:")
    for i, (_, row) in enumerate(mean_coeffs.iterrows(), 1):
        prob_positive = row['prob_positive']
        confidence = "High" if prob_positive > 0.95 else "Medium" if prob_positive > 0.8 else "Low"
        print(f"   {i}. {row['channel']}: Coefficient = {row['mean_coeff']:.3f} (Confidence: {confidence})")
    
    # Budget reallocation insights
    print(f"\n2. Budget Reallocation Recommendations:")
    for channel in contributions['channel']:
        optimal_stats = optimal_budget[channel]
        change_pct = ((optimal_stats['mean_allocation'] * 52) / 
                     optimal_stats['mean_allocation'] * 52 - 1) * 100  # Simplified calculation
        
        if abs(change_pct) > 10:
            direction = "increase" if change_pct > 0 else "decrease"
            print(f"   - {channel}: {direction} budget by ~{abs(change_pct):.0f}%")
    
    # Model uncertainty insights
    print(f"\n3. Model Uncertainty Analysis:")
    high_uncertainty_channels = contributions[contributions['std_coeff'] > contributions['std_coeff'].median()]
    if len(high_uncertainty_channels) > 0:
        print(f"   - High uncertainty channels: {', '.join(high_uncertainty_channels['channel'])}")
        print(f"   - Recommendation: Conduct incrementality tests for these channels")
    
    # Statistical significance
    print(f"\n4. Statistical Significance:")
    significant_channels = contributions[contributions['prob_positive'] > 0.95]
    print(f"   - Statistically significant channels (>95% probability): {len(significant_channels)}/{len(contributions)}")
    
    if len(significant_channels) < len(contributions):
        weak_channels = contributions[contributions['prob_positive'] < 0.8]
        if len(weak_channels) > 0:
            print(f"   - Channels with weak evidence: {', '.join(weak_channels['channel'])}")


if __name__ == "__main__":
    print("Bayesian Media Mix Model - Comprehensive Analysis")
    print("=" * 50)
    
    if PYMC_AVAILABLE:
        run_bayesian_mmm_analysis()
    else:
        print("PyMC not available. Please install with:")
        print("pip install pymc arviz")
        print("\nThis will enable:")
        print("- Bayesian uncertainty quantification")
        print("- Hierarchical modeling")
        print("- MCMC sampling for robust parameter estimation")
        print("- Credible intervals for all estimates")
        print("- Model diagnostics and convergence checking") 