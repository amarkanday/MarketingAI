"""
Customer Lifetime Value (CLV) Models

This module implements various CLV models including:
1. Buy-Till-You-Die (BTYD) models (BG/NBD, Pareto/NBD)
2. Probabilistic CLV modeling
3. Cohort-based retention analysis
4. Simple CLV calculations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln, gamma
import warnings
warnings.filterwarnings('ignore')


class BGNBDModel:
    """
    BG/NBD (Beta Geometric / Negative Binomial Distribution) Model
    
    This model predicts:
    - Customer purchase behavior (frequency)
    - Customer churn probability
    - Customer lifetime value
    """
    
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.r = None
        self.s = None
        self.fitted = False
    
    def _log_likelihood(self, params, frequency, recency, T):
        """Calculate log-likelihood for parameter estimation"""
        alpha, beta, r, s = params
        
        if any(p <= 0 for p in params):
            return -np.inf
        
        # BG/NBD likelihood calculation
        ll = 0
        for f, rec, t in zip(frequency, recency, T):
            if f == 0:
                # Customer made no purchases
                ll += gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
                ll += gammaln(beta + 1) - gammaln(alpha + beta + 1)
                ll += gammaln(r + s) - gammaln(r) - gammaln(s)
                ll += gammaln(s + 1) - gammaln(r + s + 1)
            else:
                # Customer made purchases
                ll += gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
                ll += gammaln(alpha + f) + gammaln(beta + 1) - gammaln(alpha + beta + f + 1)
                ll += gammaln(r + s) - gammaln(r) - gammaln(s)
                ll += gammaln(r + f) + gammaln(s + 1) - gammaln(r + s + f + 1)
                ll += f * np.log(rec / t) if rec > 0 else 0
        
        return ll
    
    def fit(self, frequency, recency, T):
        """
        Fit BG/NBD model to customer transaction data
        
        Args:
            frequency: Number of purchases per customer
            recency: Time of last purchase
            T: Total observation period per customer
        """
        
        def objective(params):
            return -self._log_likelihood(params, frequency, recency, T)
        
        # Initial parameter guess
        initial_guess = [1.0, 1.0, 1.0, 1.0]
        bounds = [(0.01, 50), (0.01, 50), (0.01, 50), (0.01, 50)]
        
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            self.alpha, self.beta, self.r, self.s = result.x
            self.fitted = True
            return result
        else:
            raise ValueError("Optimization failed to converge")
    
    def predict_purchases(self, t, frequency, recency, T):
        """
        Predict number of purchases in next t periods
        
        Args:
            t: Future time periods
            frequency: Historical frequency
            recency: Historical recency
            T: Historical observation period
        
        Returns:
            Expected number of purchases
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Calculate expected purchases using BG/NBD formula
        alpha_term = (self.alpha + frequency) / (self.alpha + self.beta + frequency)
        beta_term = self.beta / (self.alpha + self.beta + frequency)
        r_term = (self.r + frequency) / (self.r + self.s + frequency)
        
        expected_purchases = alpha_term * r_term * t
        
        return expected_purchases
    
    def probability_alive(self, frequency, recency, T):
        """
        Calculate probability that customer is still alive (not churned)
        
        Args:
            frequency: Number of purchases
            recency: Time of last purchase
            T: Total observation period
        
        Returns:
            Probability of being alive
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating P(alive)")
        
        if frequency == 0:
            return 1.0
        
        # BG/NBD probability alive formula
        numerator = (self.beta + frequency - 1) * (self.s + 1)
        denominator = (self.alpha + self.beta + frequency - 1) * (self.r + self.s + frequency)
        
        prob_alive = numerator / denominator
        prob_alive *= (T / recency) ** self.r if recency > 0 else 1
        
        return min(prob_alive, 1.0)


class SimpleCLVModel:
    """
    Simple CLV model using basic metrics
    """
    
    def __init__(self):
        self.average_order_value = None
        self.purchase_frequency = None
        self.gross_margin = None
        self.churn_rate = None
    
    def fit(self, transaction_data):
        """
        Fit simple CLV model to transaction data
        
        Args:
            transaction_data: DataFrame with columns ['customer_id', 'amount', 'date']
        """
        # Calculate average order value
        self.average_order_value = transaction_data['amount'].mean()
        
        # Calculate purchase frequency (purchases per customer per time period)
        customer_stats = transaction_data.groupby('customer_id').agg({
            'amount': ['count', 'mean'],
            'date': ['min', 'max']
        }).reset_index()
        
        customer_stats.columns = ['customer_id', 'frequency', 'avg_amount', 'first_purchase', 'last_purchase']
        
        # Calculate time span for each customer
        customer_stats['time_span'] = (customer_stats['last_purchase'] - customer_stats['first_purchase']).dt.days
        customer_stats['time_span'] = customer_stats['time_span'].fillna(1)  # Handle single-purchase customers
        
        # Calculate frequency per unit time
        self.purchase_frequency = (customer_stats['frequency'] / (customer_stats['time_span'] / 365)).mean()
        
        # Estimate churn rate (simplified)
        total_customers = customer_stats.shape[0]
        recent_customers = customer_stats[customer_stats['last_purchase'] > customer_stats['last_purchase'].max() - pd.Timedelta(days=90)].shape[0]
        self.churn_rate = 1 - (recent_customers / total_customers)
        
        # Set default gross margin if not provided
        if self.gross_margin is None:
            self.gross_margin = 0.2  # 20% default
    
    def calculate_clv(self, time_horizon_years=5):
        """
        Calculate Customer Lifetime Value
        
        Args:
            time_horizon_years: Time horizon for CLV calculation
        
        Returns:
            CLV value
        """
        if any(x is None for x in [self.average_order_value, self.purchase_frequency, self.churn_rate]):
            raise ValueError("Model must be fitted before CLV calculation")
        
        # Simple CLV formula
        annual_value = self.average_order_value * self.purchase_frequency * self.gross_margin
        
        # Calculate retention rate
        retention_rate = 1 - self.churn_rate
        
        # CLV with discount rate (5% annual)
        discount_rate = 0.05
        clv = 0
        
        for year in range(1, time_horizon_years + 1):
            year_value = annual_value * (retention_rate ** year)
            discounted_value = year_value / ((1 + discount_rate) ** year)
            clv += discounted_value
        
        return clv


class CohortAnalysis:
    """
    Cohort analysis for customer retention and CLV
    """
    
    def __init__(self):
        self.cohort_data = None
        self.retention_table = None
    
    def create_cohorts(self, transaction_data, cohort_period='M'):
        """
        Create customer cohorts based on first purchase date
        
        Args:
            transaction_data: DataFrame with columns ['customer_id', 'amount', 'date']
            cohort_period: 'M' for monthly, 'Q' for quarterly, 'Y' for yearly
        """
        # Ensure date column is datetime
        transaction_data['date'] = pd.to_datetime(transaction_data['date'])
        
        # Get first purchase date for each customer
        customer_cohorts = transaction_data.groupby('customer_id')['date'].min().reset_index()
        customer_cohorts.columns = ['customer_id', 'cohort_group']
        
        # Create period groups
        if cohort_period == 'M':
            customer_cohorts['cohort_group'] = customer_cohorts['cohort_group'].dt.to_period('M')
            transaction_data['period'] = transaction_data['date'].dt.to_period('M')
        elif cohort_period == 'Q':
            customer_cohorts['cohort_group'] = customer_cohorts['cohort_group'].dt.to_period('Q')
            transaction_data['period'] = transaction_data['date'].dt.to_period('Q')
        elif cohort_period == 'Y':
            customer_cohorts['cohort_group'] = customer_cohorts['cohort_group'].dt.to_period('Y')
            transaction_data['period'] = transaction_data['date'].dt.to_period('Y')
        
        # Merge cohort information with transaction data
        df = transaction_data.merge(customer_cohorts, on='customer_id')
        
        # Calculate period number
        df['period_number'] = (df['period'] - df['cohort_group']).apply(attrgetter('n'))
        
        self.cohort_data = df
        return df
    
    def calculate_retention_rates(self):
        """
        Calculate retention rates for each cohort
        """
        if self.cohort_data is None:
            raise ValueError("Must create cohorts first")
        
        # Count unique customers in each cohort/period combination
        cohort_sizes = self.cohort_data.groupby('cohort_group')['customer_id'].nunique()
        cohort_table = self.cohort_data.groupby(['cohort_group', 'period_number'])['customer_id'].nunique().reset_index()
        
        # Calculate retention rates
        cohort_table['cohort_size'] = cohort_table['cohort_group'].map(cohort_sizes)
        cohort_table['retention_rate'] = cohort_table['customer_id'] / cohort_table['cohort_size']
        
        # Pivot to create retention table
        self.retention_table = cohort_table.pivot(index='cohort_group', 
                                                 columns='period_number', 
                                                 values='retention_rate')
        
        return self.retention_table
    
    def plot_retention_heatmap(self):
        """
        Plot retention rate heatmap
        """
        if self.retention_table is None:
            raise ValueError("Must calculate retention rates first")
        
        plt.figure(figsize=(15, 8))
        plt.imshow(self.retention_table.values, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Retention Rate')
        plt.title('Cohort Retention Rates')
        plt.xlabel('Period Number')
        plt.ylabel('Cohort Group')
        
        # Add text annotations
        for i in range(len(self.retention_table.index)):
            for j in range(len(self.retention_table.columns)):
                if not pd.isna(self.retention_table.iloc[i, j]):
                    plt.text(j, i, f'{self.retention_table.iloc[i, j]:.2f}', 
                            ha='center', va='center', color='black', fontsize=8)
        
        plt.tight_layout()
        plt.show()


# Utility functions
def generate_sample_data(n_customers=1000, time_periods=365):
    """
    Generate sample transaction data for testing
    """
    np.random.seed(42)
    
    data = []
    start_date = pd.Timestamp('2023-01-01')
    
    for customer_id in range(n_customers):
        # Random customer behavior
        avg_days_between_purchases = np.random.exponential(30)
        avg_order_value = np.random.lognormal(4, 0.5)
        churn_probability = np.random.beta(2, 8)
        
        current_date = start_date + pd.Timedelta(days=np.random.randint(0, 90))
        
        while current_date < start_date + pd.Timedelta(days=time_periods):
            # Check if customer churns
            if np.random.random() < churn_probability:
                break
            
            # Generate transaction
            amount = max(1, np.random.normal(avg_order_value, avg_order_value * 0.3))
            data.append({
                'customer_id': customer_id,
                'date': current_date,
                'amount': round(amount, 2)
            })
            
            # Next purchase date
            days_to_next = np.random.exponential(avg_days_between_purchases)
            current_date += pd.Timedelta(days=days_to_next)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Generate sample data
    print("Generating sample transaction data...")
    sample_data = generate_sample_data(n_customers=500, time_periods=365)
    print(f"Generated {len(sample_data)} transactions for {sample_data['customer_id'].nunique()} customers")
    
    # Simple CLV Model
    print("\n=== Simple CLV Model ===")
    simple_model = SimpleCLVModel()
    simple_model.gross_margin = 0.25  # 25% margin
    simple_model.fit(sample_data)
    
    clv_5_year = simple_model.calculate_clv(time_horizon_years=5)
    print(f"Average Order Value: ${simple_model.average_order_value:.2f}")
    print(f"Purchase Frequency: {simple_model.purchase_frequency:.2f} purchases/year")
    print(f"Churn Rate: {simple_model.churn_rate:.2%}")
    print(f"5-Year CLV: ${clv_5_year:.2f}")
    
    # Cohort Analysis
    print("\n=== Cohort Analysis ===")
    cohort_analysis = CohortAnalysis()
    
    from operator import attrgetter  # Import needed for cohort analysis
    
    cohort_data = cohort_analysis.create_cohorts(sample_data, cohort_period='M')
    retention_rates = cohort_analysis.calculate_retention_rates()
    
    print("Retention rates calculated. Use plot_retention_heatmap() to visualize.")
    print(f"Average retention rate after 1 period: {retention_rates.iloc[:, 1].mean():.2%}")
    print(f"Average retention rate after 3 periods: {retention_rates.iloc[:, 3].mean():.2%}") 