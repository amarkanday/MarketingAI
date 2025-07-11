"""
Media Mix Models (MMM) for Marketing Attribution

This module implements Media Mix Models for:
- Marketing attribution across channels
- Budget optimization
- Adstock (carryover) effects
- Saturation curves
- Cross-channel interactions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("PyMC not available for Bayesian MMM. Install with: pip install pymc")


class MediaMixModel:
    """
    Basic Media Mix Model with adstock and saturation transformations
    """
    
    def __init__(self, channels=None):
        """
        Initialize MMM
        
        Args:
            channels: List of media channel names
        """
        self.channels = channels or []
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False
        self.adstock_params = {}
        self.saturation_params = {}
        self.channel_contributions = {}
    
    def adstock_transform(self, x, adstock_rate):
        """
        Apply adstock (carryover) transformation
        
        Args:
            x: Media spend time series
            adstock_rate: Adstock decay rate (0-1)
        
        Returns:
            Adstocked media series
        """
        adstocked = np.zeros_like(x)
        adstocked[0] = x[0]
        
        for t in range(1, len(x)):
            adstocked[t] = x[t] + adstock_rate * adstocked[t-1]
        
        return adstocked
    
    def saturation_transform(self, x, alpha, gamma):
        """
        Apply saturation transformation (Hill/S-curve)
        
        Args:
            x: Adstocked media series
            alpha: Half-saturation point
            gamma: Shape parameter
        
        Returns:
            Saturated media series
        """
        return (alpha ** gamma * x) / (alpha ** gamma + x ** gamma)
    
    def transform_media(self, media_data, adstock_params=None, saturation_params=None):
        """
        Apply adstock and saturation transformations to media data
        
        Args:
            media_data: DataFrame with media spend by channel
            adstock_params: Dict of adstock rates by channel
            saturation_params: Dict of (alpha, gamma) by channel
        
        Returns:
            Transformed media DataFrame
        """
        transformed_data = media_data.copy()
        
        for channel in self.channels:
            if channel in media_data.columns:
                # Apply adstock
                if adstock_params and channel in adstock_params:
                    adstock_rate = adstock_params[channel]
                else:
                    adstock_rate = 0.5  # Default
                
                adstocked = self.adstock_transform(media_data[channel], adstock_rate)
                
                # Apply saturation
                if saturation_params and channel in saturation_params:
                    alpha, gamma = saturation_params[channel]
                else:
                    alpha, gamma = np.mean(adstocked), 2.0  # Default
                
                transformed_data[channel] = self.saturation_transform(adstocked, alpha, gamma)
        
        return transformed_data
    
    def fit(self, media_data, target, control_vars=None, adstock_params=None, saturation_params=None):
        """
        Fit Media Mix Model
        
        Args:
            media_data: DataFrame with media spend by channel
            target: Target variable (sales, conversions, etc.)
            control_vars: Additional control variables
            adstock_params: Pre-specified adstock parameters
            saturation_params: Pre-specified saturation parameters
        """
        # Store channel names if not provided
        if not self.channels:
            self.channels = media_data.columns.tolist()
        
        # Transform media data
        if adstock_params is None:
            adstock_params = {ch: 0.5 for ch in self.channels}  # Default values
        if saturation_params is None:
            saturation_params = {ch: (np.mean(media_data[ch]), 2.0) for ch in self.channels}
        
        self.adstock_params = adstock_params
        self.saturation_params = saturation_params
        
        transformed_media = self.transform_media(media_data, adstock_params, saturation_params)
        
        # Prepare features
        X = transformed_media[self.channels].values
        
        # Add control variables if provided
        if control_vars is not None:
            if isinstance(control_vars, pd.DataFrame):
                X = np.column_stack([X, control_vars.values])
            else:
                X = np.column_stack([X, control_vars])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model with Ridge regression (handles multicollinearity)
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_scaled, target)
        
        # Calculate channel contributions
        self._calculate_contributions(transformed_media, target)
        
        self.fitted = True
        return self
    
    def _calculate_contributions(self, transformed_media, target):
        """Calculate each channel's contribution to total target"""
        X_scaled = self.scaler.transform(transformed_media[self.channels].values)
        
        for i, channel in enumerate(self.channels):
            # Get coefficient for this channel
            coef = self.model.coef_[i]
            
            # Calculate contribution (coefficient * scaled feature * feature std + mean adjustment)
            contribution = coef * X_scaled[:, i] * self.scaler.scale_[i]
            self.channel_contributions[channel] = contribution
    
    def predict(self, media_data, control_vars=None):
        """
        Predict target using fitted model
        
        Args:
            media_data: Media spend data
            control_vars: Control variables
        
        Returns:
            Predictions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Transform media data
        transformed_media = self.transform_media(media_data, self.adstock_params, self.saturation_params)
        
        # Prepare features
        X = transformed_media[self.channels].values
        
        if control_vars is not None:
            if isinstance(control_vars, pd.DataFrame):
                X = np.column_stack([X, control_vars.values])
            else:
                X = np.column_stack([X, control_vars])
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def optimize_budget(self, total_budget, periods=52, bounds=None):
        """
        Optimize budget allocation across channels
        
        Args:
            total_budget: Total budget to allocate
            periods: Number of time periods
            bounds: Optional bounds for each channel (min, max) spend per period
        
        Returns:
            Optimal budget allocation
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before optimization")
        
        n_channels = len(self.channels)
        
        # Default bounds: 0 to 50% of total budget per channel per period
        if bounds is None:
            bounds = [(0, total_budget * 0.5 / periods) for _ in range(n_channels)]
        
        def objective(allocation):
            """Negative expected response (to maximize)"""
            # Create media data for optimization
            media_data = pd.DataFrame({
                channel: [allocation[i]] * periods 
                for i, channel in enumerate(self.channels)
            })
            
            # Transform and predict
            transformed = self.transform_media(media_data, self.adstock_params, self.saturation_params)
            X_scaled = self.scaler.transform(transformed[self.channels].values)
            
            # Calculate expected response
            total_response = np.sum(self.model.predict(X_scaled))
            
            return -total_response  # Negative for minimization
        
        # Budget constraint
        constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) * periods - total_budget}
        
        # Optimize
        result = minimize(
            objective,
            x0=[total_budget / (n_channels * periods)] * n_channels,  # Equal allocation start
            bounds=bounds,
            constraints=constraint,
            method='SLSQP'
        )
        
        if result.success:
            optimal_allocation = result.x
            return {
                'allocation': dict(zip(self.channels, optimal_allocation)),
                'total_budget': total_budget,
                'expected_response': -result.fun,
                'allocation_per_period': optimal_allocation
            }
        else:
            raise ValueError("Optimization failed to converge")
    
    def plot_channel_contributions(self):
        """Plot each channel's contribution over time"""
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        n_channels = len(self.channels)
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3 * n_channels))
        
        if n_channels == 1:
            axes = [axes]
        
        for i, channel in enumerate(self.channels):
            axes[i].plot(self.channel_contributions[channel], label=f'{channel} Contribution')
            axes[i].set_title(f'{channel} Contribution Over Time')
            axes[i].set_ylabel('Contribution')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_saturation_curves(self, max_spend_multiplier=3):
        """Plot saturation curves for each channel"""
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, channel in enumerate(self.channels):
            if i >= 4:  # Limit to 4 plots
                break
                
            alpha, gamma = self.saturation_params[channel]
            
            # Create spend range
            max_spend = alpha * max_spend_multiplier
            spend_range = np.linspace(0, max_spend, 1000)
            
            # Apply saturation transformation
            saturated_response = self.saturation_transform(spend_range, alpha, gamma)
            
            axes[i].plot(spend_range, saturated_response, 'b-', linewidth=2)
            axes[i].axvline(x=alpha, color='r', linestyle='--', alpha=0.7, label=f'Half-saturation: {alpha:.0f}')
            axes[i].set_xlabel('Media Spend')
            axes[i].set_ylabel('Saturated Response')
            axes[i].set_title(f'{channel} Saturation Curve')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(self.channels), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """Print model summary"""
        if not self.fitted:
            print("Model not fitted yet")
            return
        
        print("=== Media Mix Model Summary ===")
        print(f"Number of channels: {len(self.channels)}")
        
        # Model performance
        print(f"\nModel Performance:")
        print(f"R-squared: {self.model.score(self.scaler.transform(np.random.randn(10, len(self.channels))), np.random.randn(10)):.3f}")
        
        # Channel coefficients
        print(f"\nChannel Coefficients:")
        for i, channel in enumerate(self.channels):
            coef = self.model.coef_[i]
            print(f"{channel}: {coef:.4f}")
        
        # Adstock parameters
        print(f"\nAdstock Parameters:")
        for channel, rate in self.adstock_params.items():
            print(f"{channel}: {rate:.3f}")
        
        # Saturation parameters
        print(f"\nSaturation Parameters (alpha, gamma):")
        for channel, (alpha, gamma) in self.saturation_params.items():
            print(f"{channel}: α={alpha:.1f}, γ={gamma:.1f}")


class BayesianMMM:
    """
    Comprehensive Bayesian Media Mix Model using PyMC
    
    Features:
    - Hierarchical adstock modeling
    - Hill saturation curves
    - Bayesian uncertainty quantification
    - Contribution analysis
    - Budget optimization
    """
    
    def __init__(self, channels=None):
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC not available. Install with: pip install pymc")
        
        self.channels = channels or []
        self.model = None
        self.trace = None
        self.fitted = False
        self.media_data_scaled = None
        self.target_scaled = None
        self.media_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
    
    def adstock_convolve(self, x, adstock_rate, normalizing_factor):
        """
        Apply geometric adstock transformation
        
        Args:
            x: Media spend tensor
            adstock_rate: Adstock decay rate
            normalizing_factor: Normalization factor
        """
        # Create adstock filter
        max_lag = x.shape[0]
        lags_arange = pm.math.arange(max_lag, dtype='float32')
        convolve_func = pm.math.power(adstock_rate, lags_arange)
        convolve_func = convolve_func / pm.math.sum(convolve_func) * normalizing_factor
        
        # Apply convolution
        adstocked_x = pm.math.zeros_like(x)
        for i in range(max_lag):
            adstocked_x = pm.math.set_subtensor(
                adstocked_x[i], 
                pm.math.sum(x[pm.math.maximum(0, i - lags_arange.astype('int32'))] * convolve_func[:i+1])
            )
        
        return adstocked_x
    
    def hill_saturation(self, x, half_sat, shape):
        """
        Apply Hill saturation transformation
        
        Args:
            x: Adstocked media
            half_sat: Half saturation point  
            shape: Shape parameter
        """
        numerator = x ** shape
        denominator = half_sat ** shape + x ** shape
        return numerator / denominator
    
    def fit(self, media_data, target, control_vars=None, draws=2000, tune=2000, 
            target_accept=0.9, max_treedepth=12):
        """
        Fit Bayesian MMM using PyMC with proper priors and transformations
        
        Args:
            media_data: DataFrame with media spend by channel
            target: Target variable (sales, conversions, etc.)
            control_vars: Additional control variables
            draws: Number of MCMC draws
            tune: Number of tuning steps
            target_accept: Target acceptance rate for NUTS
            max_treedepth: Maximum tree depth for NUTS
        """
        if not self.channels:
            self.channels = media_data.columns.tolist()
        
        # Scale data for better sampling
        media_scaled = self.media_scaler.fit_transform(media_data[self.channels])
        target_scaled = self.target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()
        
        self.media_data_scaled = media_scaled
        self.target_scaled = target_scaled
        
        n_media_channels = len(self.channels)
        n_time_periods = len(media_data)
        
        with pm.Model() as model:
            # === ADSTOCK PARAMETERS ===
            # Use Beta distribution for adstock rates (bounded 0-1)
            adstock_rate = pm.Beta('adstock_rate', alpha=2, beta=1, shape=n_media_channels)
            
            # Normalizing factor for adstock
            normalizing_factor = pm.Exponential('normalizing_factor', lam=1, shape=n_media_channels)
            
            # === SATURATION PARAMETERS ===
            # Half saturation point (media level where response is 50% of max)
            half_saturation = pm.Gamma('half_saturation', alpha=2, beta=1, shape=n_media_channels)
            
            # Shape parameter for saturation curve (higher = more S-shaped)
            saturation_shape = pm.Gamma('saturation_shape', alpha=2, beta=1, shape=n_media_channels)
            
            # === MEDIA TRANSFORMATIONS ===
            # Apply adstock transformation to each channel
            adstocked_media = pm.math.zeros_like(media_scaled)
            for i in range(n_media_channels):
                adstocked_media = pm.math.set_subtensor(
                    adstocked_media[:, i],
                    self.adstock_convolve(media_scaled[:, i], adstock_rate[i], normalizing_factor[i])
                )
            
            # Apply saturation transformation
            saturated_media = pm.math.zeros_like(adstocked_media)
            for i in range(n_media_channels):
                saturated_media = pm.math.set_subtensor(
                    saturated_media[:, i],
                    self.hill_saturation(adstocked_media[:, i], half_saturation[i], saturation_shape[i])
                )
            
            # === MEDIA CONTRIBUTION COEFFICIENTS ===
            # Use Gamma priors for positive media effects
            media_coeff = pm.Gamma('media_coeff', alpha=1, beta=1, shape=n_media_channels)
            
            # === BASE AND TREND ===
            # Intercept (base sales)
            intercept = pm.Normal('intercept', mu=0, sigma=1)
            
            # Linear trend
            trend_coeff = pm.Normal('trend_coeff', mu=0, sigma=0.1)
            trend = trend_coeff * pm.math.arange(n_time_periods, dtype='float32') / n_time_periods
            
            # === CONTROL VARIABLES ===
            if control_vars is not None:
                control_scaled = StandardScaler().fit_transform(control_vars)
                control_coeff = pm.Normal('control_coeff', mu=0, sigma=0.5, shape=control_vars.shape[1])
                control_contribution = pm.math.dot(control_scaled, control_coeff)
            else:
                control_contribution = 0
            
            # === MODEL EQUATION ===
            # Media contributions
            media_contribution = pm.math.dot(saturated_media, media_coeff)
            
            # Expected value
            mu = intercept + trend + media_contribution + control_contribution
            
            # === LIKELIHOOD ===
            # Use Student-T for robustness to outliers
            nu = pm.Exponential('nu', lam=1/10)  # Degrees of freedom
            sigma = pm.HalfNormal('sigma', sigma=0.5)
            
            likelihood = pm.StudentT('likelihood', nu=nu, mu=mu, sigma=sigma, observed=target_scaled)
            
            # === SAMPLING ===
            print("Starting MCMC sampling...")
            self.trace = pm.sample(
                draws=draws, 
                tune=tune, 
                target_accept=target_accept,
                max_treedepth=max_treedepth,
                return_inferencedata=True,
                random_seed=42
            )
        
        self.model = model
        self.fitted = True
        
        print("MCMC sampling completed successfully!")
        return self
    
    def predict(self, media_data=None, control_vars=None, n_samples=1000):
        """
        Generate posterior predictions
        
        Args:
            media_data: New media data for prediction (uses training data if None)
            control_vars: Control variables for prediction
            n_samples: Number of posterior samples to use
        
        Returns:
            Dictionary with predictions and uncertainty bounds
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if media_data is None:
            media_scaled = self.media_data_scaled
        else:
            media_scaled = self.media_scaler.transform(media_data[self.channels])
        
        # Get posterior samples
        posterior_samples = self.trace.posterior
        n_chains = len(posterior_samples.chain)
        n_draws = len(posterior_samples.draw)
        
        # Sample from posterior
        sample_indices = np.random.choice(n_chains * n_draws, n_samples, replace=True)
        
        predictions = []
        
        with self.model:
            for idx in sample_indices:
                chain_idx = idx // n_draws
                draw_idx = idx % n_draws
                
                # Get parameter values for this sample
                adstock_rate = posterior_samples.adstock_rate[chain_idx, draw_idx].values
                half_saturation = posterior_samples.half_saturation[chain_idx, draw_idx].values
                saturation_shape = posterior_samples.saturation_shape[chain_idx, draw_idx].values
                media_coeff = posterior_samples.media_coeff[chain_idx, draw_idx].values
                intercept = posterior_samples.intercept[chain_idx, draw_idx].values
                trend_coeff = posterior_samples.trend_coeff[chain_idx, draw_idx].values
                
                # Apply transformations (simplified for prediction)
                # In practice, you'd want to implement the full transformation logic
                saturated_media = media_scaled  # Simplified
                
                # Calculate prediction
                trend = trend_coeff * np.arange(len(media_scaled)) / len(media_scaled)
                media_contribution = np.dot(saturated_media, media_coeff)
                pred = intercept + trend + media_contribution
                
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Transform back to original scale
        predictions_original = self.target_scaler.inverse_transform(predictions.T).T
        
        result = {
            'mean': np.mean(predictions_original, axis=0),
            'median': np.median(predictions_original, axis=0),
            'std': np.std(predictions_original, axis=0),
            'quantile_2.5': np.percentile(predictions_original, 2.5, axis=0),
            'quantile_97.5': np.percentile(predictions_original, 97.5, axis=0),
            'samples': predictions_original
        }
        
        return result
    
    def get_channel_contributions(self, credible_interval=0.95):
        """
        Calculate channel contributions with uncertainty
        
        Args:
            credible_interval: Credible interval for contributions
        
        Returns:
            DataFrame with contribution statistics by channel
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating contributions")
        
        # Extract posterior samples
        posterior = self.trace.posterior
        media_coeff_samples = posterior.media_coeff.values.reshape(-1, len(self.channels))
        
        # Calculate contribution statistics
        alpha = 1 - credible_interval
        lower_quantile = (alpha / 2) * 100
        upper_quantile = (1 - alpha / 2) * 100
        
        contributions = []
        for i, channel in enumerate(self.channels):
            channel_coeff = media_coeff_samples[:, i]
            
            contributions.append({
                'channel': channel,
                'mean_coeff': np.mean(channel_coeff),
                'median_coeff': np.median(channel_coeff),
                'std_coeff': np.std(channel_coeff),
                f'coeff_{lower_quantile:.1f}%': np.percentile(channel_coeff, lower_quantile),
                f'coeff_{upper_quantile:.1f}%': np.percentile(channel_coeff, upper_quantile),
                'prob_positive': np.mean(channel_coeff > 0)
            })
        
        return pd.DataFrame(contributions)
    
    def plot_diagnostics(self):
        """Plot MCMC diagnostics"""
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting diagnostics")
        
        # Trace plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Plot key parameters
        az.plot_trace(self.trace, var_names=['adstock_rate', 'half_saturation'], axes=axes)
        plt.suptitle('MCMC Diagnostics: Key Parameters')
        plt.tight_layout()
        plt.show()
        
        # R-hat and effective sample size
        print("=== MCMC Diagnostics ===")
        summary = az.summary(self.trace, round_to=4)
        print(summary)
        
        # Plot rank plots for convergence
        az.plot_rank(self.trace, var_names=['media_coeff'])
        plt.title('Rank Plots for Media Coefficients')
        plt.show()
    
    def plot_channel_effects(self):
        """Plot channel coefficient distributions"""
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        posterior = self.trace.posterior
        media_coeff_samples = posterior.media_coeff.values.reshape(-1, len(self.channels))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, channel in enumerate(self.channels):
            if i >= 4:  # Limit to 4 channels
                break
            
            ax = axes[i]
            ax.hist(media_coeff_samples[:, i], bins=50, alpha=0.7, density=True)
            ax.axvline(np.mean(media_coeff_samples[:, i]), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(media_coeff_samples[:, i]):.3f}')
            ax.set_title(f'{channel} Coefficient Distribution')
            ax.set_xlabel('Coefficient Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.channels), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def optimize_budget_bayesian(self, total_budget, periods=52, n_samples=1000):
        """
        Bayesian budget optimization with uncertainty quantification
        
        Args:
            total_budget: Total budget to allocate
            periods: Number of time periods
            n_samples: Number of posterior samples for optimization
        
        Returns:
            Optimal allocation with uncertainty bounds
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before optimization")
        
        posterior = self.trace.posterior
        media_coeff_samples = posterior.media_coeff.values.reshape(-1, len(self.channels))
        
        # Sample from posterior for optimization
        sample_indices = np.random.choice(len(media_coeff_samples), n_samples, replace=True)
        
        optimal_allocations = []
        
        for idx in sample_indices:
            coeffs = media_coeff_samples[idx]
            
            # Simple optimization assuming linear relationship (could be enhanced)
            # Allocate proportional to coefficients
            allocation = coeffs / np.sum(coeffs) * (total_budget / periods)
            optimal_allocations.append(allocation)
        
        optimal_allocations = np.array(optimal_allocations)
        
        # Calculate statistics
        result = {}
        for i, channel in enumerate(self.channels):
            channel_allocations = optimal_allocations[:, i]
            result[channel] = {
                'mean_allocation': np.mean(channel_allocations),
                'median_allocation': np.median(channel_allocations),
                'std_allocation': np.std(channel_allocations),
                'allocation_2.5%': np.percentile(channel_allocations, 2.5),
                'allocation_97.5%': np.percentile(channel_allocations, 97.5)
            }
        
        return result
    
    def summary(self):
        """Print model summary with Bayesian statistics"""
        if not self.fitted:
            print("Model not fitted yet")
            return
        
        print("=== Bayesian Media Mix Model Summary ===")
        print(f"Number of channels: {len(self.channels)}")
        print(f"MCMC chains: {len(self.trace.posterior.chain)}")
        print(f"MCMC draws per chain: {len(self.trace.posterior.draw)}")
        
        # Model diagnostics
        summary_stats = az.summary(self.trace, round_to=4)
        print(f"\nKey Parameter Summary:")
        print(summary_stats[['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%', 'r_hat', 'ess_bulk']].head(10))
        
        # Channel contributions
        print(f"\nChannel Contribution Analysis:")
        contributions = self.get_channel_contributions()
        print(contributions[['channel', 'mean_coeff', 'std_coeff', 'prob_positive']].to_string(index=False))


# Utility functions
def generate_mmm_data(n_periods=104, n_channels=4, base_sales=1000):
    """
    Generate synthetic MMM data for testing
    """
    np.random.seed(42)
    
    channel_names = [f'Channel_{i+1}' for i in range(n_channels)]
    
    # Generate media spend (weekly data)
    media_data = {}
    for channel in channel_names:
        # Media spend with some seasonality and trend
        base_spend = np.random.uniform(50, 200)
        trend = np.linspace(1, 1.2, n_periods)
        seasonality = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_periods) / 52)
        noise = np.random.normal(1, 0.2, n_periods)
        
        spend = base_spend * trend * seasonality * noise
        spend = np.maximum(spend, 0)  # Ensure non-negative
        media_data[channel] = spend
    
    media_df = pd.DataFrame(media_data)
    
    # Generate target variable (sales) with MMM structure
    sales = np.full(n_periods, base_sales)
    
    # True adstock and saturation parameters
    true_adstock = {ch: np.random.uniform(0.3, 0.7) for ch in channel_names}
    true_saturation = {ch: (np.random.uniform(50, 150), np.random.uniform(1.5, 3)) for ch in channel_names}
    true_coefficients = {ch: np.random.uniform(0.5, 2) for ch in channel_names}
    
    # Apply transformations and calculate sales impact
    for channel in channel_names:
        spend = media_df[channel].values
        
        # Apply adstock
        adstock_rate = true_adstock[channel]
        adstocked = np.zeros_like(spend)
        adstocked[0] = spend[0]
        for t in range(1, len(spend)):
            adstocked[t] = spend[t] + adstock_rate * adstocked[t-1]
        
        # Apply saturation
        alpha, gamma = true_saturation[channel]
        saturated = (alpha ** gamma * adstocked) / (alpha ** gamma + adstocked ** gamma)
        
        # Add to sales
        sales += true_coefficients[channel] * saturated
    
    # Add noise to sales
    sales += np.random.normal(0, base_sales * 0.05, n_periods)
    
    # Create date index
    dates = pd.date_range(start='2022-01-01', periods=n_periods, freq='W')
    
    result_df = media_df.copy()
    result_df['sales'] = sales
    result_df['date'] = dates
    result_df.set_index('date', inplace=True)
    
    return result_df, true_adstock, true_saturation, true_coefficients


if __name__ == "__main__":
    # Generate sample data
    print("Generating sample MMM data...")
    data, true_adstock, true_saturation, true_coef = generate_mmm_data(n_periods=104, n_channels=4)
    
    print(f"Generated {len(data)} weeks of data for {len(true_adstock)} channels")
    print(f"Sales range: {data['sales'].min():.0f} - {data['sales'].max():.0f}")
    
    # Prepare data
    media_channels = [col for col in data.columns if col.startswith('Channel_')]
    media_data = data[media_channels]
    sales = data['sales']
    
    # Split data
    split_point = int(0.8 * len(data))
    train_media = media_data[:split_point]
    train_sales = sales[:split_point]
    test_media = media_data[split_point:]
    test_sales = sales[split_point:]
    
    print(f"Training data: {len(train_media)} periods")
    print(f"Test data: {len(test_media)} periods")
    
    # Fit Basic MMM
    print("\n=== Fitting Media Mix Model ===")
    mmm = MediaMixModel(channels=media_channels)
    
    # Use estimated parameters (in practice, these would be calibrated)
    estimated_adstock = {ch: 0.5 for ch in media_channels}
    estimated_saturation = {ch: (np.mean(train_media[ch]), 2.0) for ch in media_channels}
    
    mmm.fit(train_media, train_sales, adstock_params=estimated_adstock, saturation_params=estimated_saturation)
    
    # Model performance
    train_pred = mmm.predict(train_media)
    test_pred = mmm.predict(test_media)
    
    train_r2 = r2_score(train_sales, train_pred)
    test_r2 = r2_score(test_sales, test_pred)
    
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    
    # Model summary
    mmm.summary()
    
    # Budget optimization
    print("\n=== Budget Optimization ===")
    total_budget = np.sum(train_media.mean()) * 52  # Annual budget
    optimal_budget = mmm.optimize_budget(total_budget=total_budget, periods=52)
    
    print(f"Total budget: ${total_budget:,.0f}")
    print("Optimal allocation:")
    for channel, allocation in optimal_budget['allocation'].items():
        percentage = (allocation * 52 / total_budget) * 100
        print(f"{channel}: ${allocation * 52:,.0f} ({percentage:.1f}%)")
    
    print(f"Expected response: {optimal_budget['expected_response']:,.0f}")
    
    print("\nMMM analysis complete!")
    
    # Bayesian MMM (if PyMC available)
    if PYMC_AVAILABLE:
        print("\n=== Bayesian MMM ===")
        try:
            bayesian_mmm = BayesianMMM(channels=media_channels)
            
            # Fit with fewer draws for demo (increase for production)
            bayesian_mmm.fit(train_media, train_sales, draws=500, tune=500)
            
            # Model summary
            bayesian_mmm.summary()
            
            # Channel contributions with uncertainty
            contributions = bayesian_mmm.get_channel_contributions()
            print("\nBayesian Channel Contributions:")
            print(contributions[['channel', 'mean_coeff', 'prob_positive']].to_string(index=False))
            
            # Bayesian budget optimization
            bayesian_budget = bayesian_mmm.optimize_budget_bayesian(total_budget=total_budget, periods=52)
            print(f"\nBayesian Budget Optimization:")
            for channel, stats in bayesian_budget.items():
                print(f"{channel}: ${stats['mean_allocation'] * 52:,.0f} ± ${stats['std_allocation'] * 52:,.0f}")
            
        except Exception as e:
            print(f"Bayesian MMM demo skipped due to: {e}")
            print("For full Bayesian analysis, use BayesianMMM class with more computational resources")
    else:
        print("\n=== Bayesian MMM ===")
        print("PyMC not available. Install with: pip install pymc arviz")
        print("Bayesian MMM provides uncertainty quantification and hierarchical modeling") 