"""
Bass Diffusion Model for Product Adoption Forecasting

The Bass model predicts the adoption of new products over time using three parameters:
- m: Market potential (total number of adopters)
- p: Coefficient of innovation (external influence)
- q: Coefficient of imitation (internal influence)

Formula: f(t) = [(p + q)^2 / p] * [exp(-(p+q)*t) / (1 + (q/p)*exp(-(p+q)*t))^2]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class BassModel:
    """
    Bass Diffusion Model for predicting product adoption curves
    """
    
    def __init__(self):
        self.m = None  # Market potential
        self.p = None  # Innovation coefficient
        self.q = None  # Imitation coefficient
        self.fitted = False
    
    def cumulative_adopters(self, t, m, p, q):
        """
        Calculate cumulative adopters at time t
        
        Args:
            t: Time period
            m: Market potential
            p: Innovation coefficient
            q: Imitation coefficient
        
        Returns:
            Cumulative number of adopters
        """
        if p + q == 0:
            return 0
        
        exp_term = np.exp(-(p + q) * t)
        numerator = m * (1 - exp_term)
        denominator = 1 + (q / p) * exp_term if p != 0 else 1
        
        return numerator / denominator
    
    def instantaneous_adopters(self, t, m, p, q):
        """
        Calculate instantaneous adopters at time t (derivative of cumulative)
        
        Args:
            t: Time period
            m: Market potential
            p: Innovation coefficient  
            q: Imitation coefficient
        
        Returns:
            Number of new adopters at time t
        """
        if p + q == 0:
            return 0
            
        exp_term = np.exp(-(p + q) * t)
        numerator = m * (p + q) ** 2 * exp_term
        denominator = (p + (q * exp_term)) ** 2
        
        return numerator / denominator
    
    def fit(self, time_periods, observed_adopters, method='cumulative'):
        """
        Fit Bass model parameters to observed data
        
        Args:
            time_periods: Array of time periods
            observed_adopters: Array of observed adopters (cumulative or instantaneous)
            method: 'cumulative' or 'instantaneous'
        """
        
        def objective(params):
            m, p, q = params
            if m <= 0 or p <= 0 or q <= 0:
                return np.inf
                
            if method == 'cumulative':
                predicted = [self.cumulative_adopters(t, m, p, q) for t in time_periods]
            else:
                predicted = [self.instantaneous_adopters(t, m, p, q) for t in time_periods]
            
            return np.sum((np.array(observed_adopters) - np.array(predicted)) ** 2)
        
        # Initial guess
        initial_guess = [max(observed_adopters) * 2, 0.01, 0.1]
        
        # Constraints: all parameters must be positive
        bounds = [(1, None), (0.001, 1), (0.001, 1)]
        
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            self.m, self.p, self.q = result.x
            self.fitted = True
            return result
        else:
            raise ValueError("Optimization failed to converge")
    
    def predict(self, time_periods, method='cumulative'):
        """
        Predict adopters for given time periods
        
        Args:
            time_periods: Array of time periods to predict
            method: 'cumulative' or 'instantaneous'
        
        Returns:
            Array of predicted adopters
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if method == 'cumulative':
            return [self.cumulative_adopters(t, self.m, self.p, self.q) for t in time_periods]
        else:
            return [self.instantaneous_adopters(t, self.m, self.p, self.q) for t in time_periods]
    
    def get_peak_time(self):
        """
        Calculate the time when adoption rate peaks
        
        Returns:
            Time of peak adoption rate
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating peak time")
        
        return np.log(self.q / self.p) / (self.p + self.q)
    
    def get_inflection_point(self):
        """
        Calculate the inflection point of cumulative adoption curve
        
        Returns:
            (time, cumulative_adopters) at inflection point
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating inflection point")
        
        t_inflection = self.get_peak_time()
        y_inflection = self.cumulative_adopters(t_inflection, self.m, self.p, self.q)
        
        return t_inflection, y_inflection
    
    def plot_forecast(self, time_periods, observed_data=None, forecast_periods=None):
        """
        Plot the Bass model forecast
        
        Args:
            time_periods: Historical time periods
            observed_data: Historical observed data
            forecast_periods: Future periods to forecast
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Cumulative adoption
        cum_pred = self.predict(time_periods, method='cumulative')
        ax1.plot(time_periods, cum_pred, 'b-', label='Bass Model (Cumulative)', linewidth=2)
        
        if observed_data is not None:
            ax1.scatter(time_periods, observed_data, color='red', alpha=0.7, label='Observed Data')
        
        if forecast_periods is not None:
            forecast_cum = self.predict(forecast_periods, method='cumulative')
            ax1.plot(forecast_periods, forecast_cum, 'b--', alpha=0.7, label='Forecast')
        
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Cumulative Adopters')
        ax1.set_title('Bass Model: Cumulative Adoption')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Instantaneous adoption
        inst_pred = self.predict(time_periods, method='instantaneous')
        ax2.plot(time_periods, inst_pred, 'g-', label='Bass Model (Instantaneous)', linewidth=2)
        
        if forecast_periods is not None:
            forecast_inst = self.predict(forecast_periods, method='instantaneous')
            ax2.plot(forecast_periods, forecast_inst, 'g--', alpha=0.7, label='Forecast')
        
        # Mark peak
        peak_time = self.get_peak_time()
        peak_adopters = self.instantaneous_adopters(peak_time, self.m, self.p, self.q)
        ax2.axvline(x=peak_time, color='red', linestyle=':', alpha=0.7, label=f'Peak at t={peak_time:.1f}')
        ax2.scatter([peak_time], [peak_adopters], color='red', s=100, zorder=5)
        
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('New Adopters')
        ax2.set_title('Bass Model: Instantaneous Adoption Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def summary(self):
        """
        Print model summary statistics
        """
        if not self.fitted:
            print("Model not fitted yet")
            return
        
        peak_time, peak_adopters = self.get_peak_time(), self.instantaneous_adopters(self.get_peak_time(), self.m, self.p, self.q)
        inflection_time, inflection_adopters = self.get_inflection_point()
        
        print("=== Bass Diffusion Model Summary ===")
        print(f"Market Potential (m): {self.m:,.0f}")
        print(f"Innovation Coefficient (p): {self.p:.4f}")
        print(f"Imitation Coefficient (q): {self.q:.4f}")
        print(f"p + q: {self.p + self.q:.4f}")
        print(f"q/p ratio: {self.q/self.p:.2f}")
        print(f"\nKey Metrics:")
        print(f"Peak adoption time: {peak_time:.1f}")
        print(f"Peak adoption rate: {peak_adopters:,.0f}")
        print(f"Inflection point: t={inflection_time:.1f}, cumulative={inflection_adopters:,.0f}")
        print(f"50% adoption time: {inflection_time:.1f}")


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    
    # True parameters
    true_m, true_p, true_q = 1000, 0.03, 0.38
    
    # Generate synthetic observed data
    time_periods = np.arange(1, 25)
    true_cumulative = [BassModel().cumulative_adopters(t, true_m, true_p, true_q) for t in time_periods]
    
    # Add noise
    observed_cumulative = [max(0, val + np.random.normal(0, 20)) for val in true_cumulative]
    
    # Fit model
    model = BassModel()
    result = model.fit(time_periods, observed_cumulative, method='cumulative')
    
    # Print results
    model.summary()
    
    # Plot forecast
    forecast_periods = np.arange(25, 41)
    model.plot_forecast(time_periods, observed_cumulative, forecast_periods) 