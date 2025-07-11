"""
Price Elasticity Models for Marketing Analytics

This module implements various price elasticity models for:
- Demand response to pricing changes
- Optimal pricing strategies
- Competitive price analysis
- Revenue optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize_scalar, minimize
import warnings
warnings.filterwarnings('ignore')


class LogLogElasticityModel:
    """
    Log-Log elasticity model using linear regression on logarithmic data
    
    The model estimates: log(demand) = α + β * log(price) + ε
    Where β is the price elasticity of demand
    """
    
    def __init__(self, include_controls=True):
        """
        Initialize Log-Log elasticity model
        
        Args:
            include_controls: Whether to include control variables
        """
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False
        self.include_controls = include_controls
        self.elasticity = None
        self.feature_names = None
    
    def fit(self, data, price_col='price', demand_col='demand', control_cols=None):
        """
        Fit the elasticity model
        
        Args:
            data: DataFrame with price, demand, and control variables
            price_col: Name of price column
            demand_col: Name of demand column
            control_cols: List of control variable columns
        """
        # Prepare data
        df = data.copy()
        
        # Remove zero or negative values (can't take log)
        df = df[(df[price_col] > 0) & (df[demand_col] > 0)]
        
        if len(df) == 0:
            raise ValueError("No valid data points after removing zeros/negatives")
        
        # Take logarithms
        log_price = np.log(df[price_col])
        log_demand = np.log(df[demand_col])
        
        # Prepare features
        X = log_price.values.reshape(-1, 1)
        feature_names = ['log_price']
        
        # Add control variables if specified
        if self.include_controls and control_cols:
            for col in control_cols:
                if col in df.columns:
                    X = np.column_stack([X, df[col].values])
                    feature_names.append(col)
        
        self.feature_names = feature_names
        
        # Fit model
        self.model = LinearRegression()
        self.model.fit(X, log_demand)
        
        # Extract price elasticity (coefficient of log_price)
        self.elasticity = self.model.coef_[0]
        self.fitted = True
        
        return self
    
    def predict_demand(self, prices, controls=None):
        """
        Predict demand for given prices
        
        Args:
            prices: Array of prices
            controls: Control variables (if used in fitting)
        
        Returns:
            Predicted demand values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        prices = np.array(prices)
        
        # Take log of prices
        log_prices = np.log(prices).reshape(-1, 1)
        
        # Add controls if needed
        if self.include_controls and controls is not None:
            if isinstance(controls, dict):
                # Repeat control values for each price
                control_matrix = np.tile(list(controls.values()), (len(prices), 1))
                log_prices = np.column_stack([log_prices, control_matrix])
            else:
                log_prices = np.column_stack([log_prices, controls])
        
        # Predict log demand
        log_demand_pred = self.model.predict(log_prices)
        
        # Convert back to original scale
        demand_pred = np.exp(log_demand_pred)
        
        return demand_pred
    
    def calculate_elasticity_at_point(self, price, controls=None):
        """
        Calculate elasticity at a specific price point
        
        For log-log models, elasticity is constant and equals the coefficient
        """
        return self.elasticity
    
    def optimal_price(self, cost_per_unit, price_range=(1, 1000), controls=None):
        """
        Find optimal price that maximizes profit
        
        Args:
            cost_per_unit: Variable cost per unit
            price_range: (min_price, max_price) to search
            controls: Control variables
        
        Returns:
            Optimal price and expected profit
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before optimization")
        
        def profit_function(price):
            demand = self.predict_demand([price], controls)[0]
            revenue = price * demand
            total_cost = cost_per_unit * demand
            profit = revenue - total_cost
            return -profit  # Minimize negative profit = maximize profit
        
        result = minimize_scalar(profit_function, bounds=price_range, method='bounded')
        
        optimal_price = result.x
        optimal_demand = self.predict_demand([optimal_price], controls)[0]
        optimal_profit = -result.fun
        
        return {
            'optimal_price': optimal_price,
            'optimal_demand': optimal_demand,
            'optimal_profit': optimal_profit,
            'elasticity': self.elasticity
        }
    
    def plot_demand_curve(self, price_range=None, controls=None, n_points=100):
        """
        Plot the demand curve
        
        Args:
            price_range: (min_price, max_price) range to plot
            controls: Control variables
            n_points: Number of points to plot
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        if price_range is None:
            price_range = (1, 100)  # Default range
        
        prices = np.linspace(price_range[0], price_range[1], n_points)
        demands = self.predict_demand(prices, controls)
        
        plt.figure(figsize=(10, 6))
        plt.plot(prices, demands, 'b-', linewidth=2, label=f'Demand Curve (ε = {self.elasticity:.2f})')
        plt.xlabel('Price')
        plt.ylabel('Demand')
        plt.title('Price-Demand Relationship')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """Print model summary"""
        if not self.fitted:
            print("Model not fitted yet")
            return
        
        print("=== Price Elasticity Model Summary ===")
        print(f"Price Elasticity of Demand: {self.elasticity:.3f}")
        print(f"R-squared: {self.model.score(self.model.feature_names_in_.reshape(-1, 1) if hasattr(self.model, 'feature_names_in_') else np.array([[0]]), np.array([0])):.3f}")
        
        interpretation = "inelastic" if abs(self.elasticity) < 1 else "elastic"
        print(f"Demand is {interpretation}")
        
        if self.elasticity < 0:
            print(f"1% increase in price → {abs(self.elasticity):.1f}% decrease in demand")
        else:
            print("Warning: Positive elasticity (unusual for normal goods)")


class CompetitivePriceModel:
    """
    Model that includes competitive pricing effects
    """
    
    def __init__(self):
        self.model = None
        self.fitted = False
        self.own_price_elasticity = None
        self.cross_price_elasticity = None
    
    def fit(self, data, own_price_col='own_price', competitor_price_col='competitor_price', 
            demand_col='demand'):
        """
        Fit competitive price model
        
        Args:
            data: DataFrame with own price, competitor price, and demand
            own_price_col: Name of own price column
            competitor_price_col: Name of competitor price column
            demand_col: Name of demand column
        """
        df = data.copy()
        
        # Remove invalid values
        valid_mask = (df[own_price_col] > 0) & (df[competitor_price_col] > 0) & (df[demand_col] > 0)
        df = df[valid_mask]
        
        if len(df) == 0:
            raise ValueError("No valid data points")
        
        # Take logarithms
        log_own_price = np.log(df[own_price_col])
        log_competitor_price = np.log(df[competitor_price_col])
        log_demand = np.log(df[demand_col])
        
        # Prepare features: [log_own_price, log_competitor_price]
        X = np.column_stack([log_own_price, log_competitor_price])
        
        # Fit model
        self.model = LinearRegression()
        self.model.fit(X, log_demand)
        
        # Extract elasticities
        self.own_price_elasticity = self.model.coef_[0]
        self.cross_price_elasticity = self.model.coef_[1]
        self.fitted = True
        
        return self
    
    def predict_demand(self, own_prices, competitor_prices):
        """
        Predict demand given own and competitor prices
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        own_prices = np.array(own_prices)
        competitor_prices = np.array(competitor_prices)
        
        log_own = np.log(own_prices)
        log_comp = np.log(competitor_prices)
        
        X = np.column_stack([log_own, log_comp])
        log_demand_pred = self.model.predict(X)
        
        return np.exp(log_demand_pred)
    
    def optimal_competitive_price(self, competitor_price, cost_per_unit, price_range=(1, 1000)):
        """
        Find optimal price given competitor's price
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before optimization")
        
        def profit_function(own_price):
            demand = self.predict_demand([own_price], [competitor_price])[0]
            profit = (own_price - cost_per_unit) * demand
            return -profit
        
        result = minimize_scalar(profit_function, bounds=price_range, method='bounded')
        
        optimal_price = result.x
        optimal_demand = self.predict_demand([optimal_price], [competitor_price])[0]
        optimal_profit = -result.fun
        
        return {
            'optimal_price': optimal_price,
            'competitor_price': competitor_price,
            'optimal_demand': optimal_demand,
            'optimal_profit': optimal_profit,
            'own_elasticity': self.own_price_elasticity,
            'cross_elasticity': self.cross_price_elasticity
        }
    
    def summary(self):
        """Print competitive model summary"""
        if not self.fitted:
            print("Model not fitted yet")
            return
        
        print("=== Competitive Price Model Summary ===")
        print(f"Own Price Elasticity: {self.own_price_elasticity:.3f}")
        print(f"Cross Price Elasticity: {self.cross_price_elasticity:.3f}")
        
        if self.cross_price_elasticity > 0:
            print("Products are substitutes (positive cross-elasticity)")
        else:
            print("Products are complements (negative cross-elasticity)")


class DynamicPricingModel:
    """
    Dynamic pricing model that considers time-varying factors
    """
    
    def __init__(self):
        self.models = {}  # Store models for different time periods
        self.fitted = False
    
    def fit(self, data, price_col='price', demand_col='demand', time_col='time_period'):
        """
        Fit separate models for different time periods
        
        Args:
            data: DataFrame with price, demand, and time period
            price_col: Name of price column
            demand_col: Name of demand column  
            time_col: Name of time period column
        """
        df = data.copy()
        
        # Fit model for each time period
        for period in df[time_col].unique():
            period_data = df[df[time_col] == period]
            
            if len(period_data) > 10:  # Minimum data points
                model = LogLogElasticityModel(include_controls=False)
                try:
                    model.fit(period_data, price_col, demand_col)
                    self.models[period] = model
                except:
                    continue
        
        self.fitted = len(self.models) > 0
        return self
    
    def get_elasticity_by_period(self):
        """Get elasticity estimates for each time period"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        elasticities = {}
        for period, model in self.models.items():
            elasticities[period] = model.elasticity
        
        return elasticities
    
    def plot_elasticity_trends(self):
        """Plot how elasticity changes over time"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        elasticities = self.get_elasticity_by_period()
        periods = sorted(elasticities.keys())
        values = [elasticities[p] for p in periods]
        
        plt.figure(figsize=(12, 6))
        plt.plot(periods, values, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=-1, color='r', linestyle='--', alpha=0.7, label='Unit Elastic')
        plt.xlabel('Time Period')
        plt.ylabel('Price Elasticity')
        plt.title('Price Elasticity Trends Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Utility functions
def generate_sample_pricing_data(n_periods=100, base_price=50, base_demand=1000):
    """
    Generate sample pricing data for testing
    """
    np.random.seed(42)
    
    data = []
    
    for period in range(n_periods):
        # Price varies randomly around base price
        price = base_price * (1 + np.random.normal(0, 0.2))
        price = max(price, 1)  # Ensure positive price
        
        # Demand responds to price with elasticity around -1.5
        true_elasticity = -1.5
        demand = base_demand * (price / base_price) ** true_elasticity
        
        # Add noise
        demand *= (1 + np.random.normal(0, 0.1))
        demand = max(demand, 1)  # Ensure positive demand
        
        # Add some control variables
        seasonality = 1 + 0.2 * np.sin(2 * np.pi * period / 12)  # Monthly seasonality
        trend = 1 + 0.001 * period  # Slight upward trend
        
        data.append({
            'period': period,
            'price': round(price, 2),
            'demand': round(demand * seasonality * trend, 0),
            'seasonality': round(seasonality, 3),
            'trend': round(trend, 3),
            'time_period': period // 10  # Group into 10-period chunks
        })
    
    return pd.DataFrame(data)


def generate_competitive_pricing_data(n_periods=100):
    """
    Generate sample data with competitive pricing effects
    """
    np.random.seed(42)
    
    data = []
    base_own_price = 50
    base_comp_price = 55
    base_demand = 1000
    
    for period in range(n_periods):
        # Own price varies
        own_price = base_own_price * (1 + np.random.normal(0, 0.15))
        own_price = max(own_price, 1)
        
        # Competitor price varies independently  
        comp_price = base_comp_price * (1 + np.random.normal(0, 0.15))
        comp_price = max(comp_price, 1)
        
        # Demand responds to both prices
        own_elasticity = -1.2
        cross_elasticity = 0.8  # Substitute products
        
        demand = base_demand * \
                (own_price / base_own_price) ** own_elasticity * \
                (comp_price / base_comp_price) ** cross_elasticity
        
        # Add noise
        demand *= (1 + np.random.normal(0, 0.1))
        demand = max(demand, 1)
        
        data.append({
            'period': period,
            'own_price': round(own_price, 2),
            'competitor_price': round(comp_price, 2),
            'demand': round(demand, 0)
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Generate sample data
    print("Generating sample pricing data...")
    sample_data = generate_sample_pricing_data(n_periods=200)
    
    # Basic Price Elasticity Model
    print("\n=== Basic Price Elasticity Model ===")
    basic_model = LogLogElasticityModel(include_controls=True)
    basic_model.fit(sample_data, 'price', 'demand', ['seasonality', 'trend'])
    basic_model.summary()
    
    # Find optimal price
    optimal_result = basic_model.optimal_price(cost_per_unit=20, price_range=(10, 100))
    print(f"\nOptimal Pricing:")
    print(f"Price: ${optimal_result['optimal_price']:.2f}")
    print(f"Demand: {optimal_result['optimal_demand']:.0f}")
    print(f"Profit: ${optimal_result['optimal_profit']:.2f}")
    
    # Competitive Price Model
    print("\n=== Competitive Price Model ===")
    comp_data = generate_competitive_pricing_data(n_periods=150)
    comp_model = CompetitivePriceModel()
    comp_model.fit(comp_data, 'own_price', 'competitor_price', 'demand')
    comp_model.summary()
    
    # Optimal competitive price
    comp_optimal = comp_model.optimal_competitive_price(
        competitor_price=60, 
        cost_per_unit=20, 
        price_range=(10, 100)
    )
    print(f"\nOptimal Competitive Pricing (competitor at $60):")
    print(f"Optimal Price: ${comp_optimal['optimal_price']:.2f}")
    print(f"Expected Demand: {comp_optimal['optimal_demand']:.0f}")
    print(f"Expected Profit: ${comp_optimal['optimal_profit']:.2f}")
    
    # Dynamic Pricing Model
    print("\n=== Dynamic Pricing Model ===")
    dynamic_model = DynamicPricingModel()
    dynamic_model.fit(sample_data, 'price', 'demand', 'time_period')
    
    elasticities_by_period = dynamic_model.get_elasticity_by_period()
    print("Elasticity by time period:")
    for period, elasticity in sorted(elasticities_by_period.items()):
        print(f"Period {period}: {elasticity:.3f}")
    
    print("\nPrice elasticity models ready for use!") 