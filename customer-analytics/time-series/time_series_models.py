"""
Time Series Forecasting Models for Marketing Analytics

This module implements various time series models for predicting:
- Sales forecasting
- Demand planning
- Seasonal analysis
- Trend decomposition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")


class ARIMAForecast:
    """
    ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting
    """
    
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize ARIMA model
        
        Args:
            order: (p, d, q) parameters for ARIMA model
                   p: autoregressive order
                   d: differencing order
                   q: moving average order
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.data = None
    
    def fit(self, data, date_column=None, value_column=None):
        """
        Fit ARIMA model to time series data
        
        Args:
            data: DataFrame with time series data or Series
            date_column: Name of date column (if DataFrame)
            value_column: Name of value column (if DataFrame)
        """
        if isinstance(data, pd.DataFrame):
            if date_column and value_column:
                data = data.set_index(date_column)[value_column]
            else:
                data = data.iloc[:, -1]  # Use last column as value
        
        self.data = data
        
        # Check for stationarity
        if not self._is_stationary(data):
            print("Warning: Time series may not be stationary. Consider differencing.")
        
        # Fit ARIMA model
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        
        return self.fitted_model
    
    def _is_stationary(self, data):
        """Check if time series is stationary using ADF test"""
        result = adfuller(data.dropna())
        return result[1] <= 0.05  # p-value <= 0.05 indicates stationarity
    
    def predict(self, steps=30):
        """
        Generate forecasts
        
        Args:
            steps: Number of periods to forecast
            
        Returns:
            Forecasted values with confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.fitted_model.forecast(steps=steps)
        conf_int = self.fitted_model.get_forecast(steps=steps).conf_int()
        
        # Create forecast DataFrame
        forecast_index = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'forecast': forecast,
            'lower_ci': conf_int.iloc[:, 0],
            'upper_ci': conf_int.iloc[:, 1]
        }, index=forecast_index)
        
        return forecast_df
    
    def plot_forecast(self, steps=30, train_periods=100):
        """
        Plot historical data and forecast
        
        Args:
            steps: Number of periods to forecast
            train_periods: Number of historical periods to show
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before plotting")
        
        forecast_df = self.predict(steps)
        
        # Plot
        plt.figure(figsize=(15, 6))
        
        # Historical data
        recent_data = self.data.tail(train_periods)
        plt.plot(recent_data.index, recent_data.values, 'b-', label='Historical', linewidth=2)
        
        # Forecast
        plt.plot(forecast_df.index, forecast_df['forecast'], 'r--', label='Forecast', linewidth=2)
        plt.fill_between(forecast_df.index, 
                        forecast_df['lower_ci'], 
                        forecast_df['upper_ci'], 
                        color='red', alpha=0.2, label='Confidence Interval')
        
        plt.title(f'ARIMA{self.order} Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def auto_arima(self, data, max_p=3, max_d=2, max_q=3):
        """
        Automatically select best ARIMA parameters using AIC
        
        Args:
            data: Time series data
            max_p, max_d, max_q: Maximum values to test for each parameter
        """
        best_aic = np.inf
        best_order = None
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        print(f"Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")
        self.order = best_order
        return best_order


class ProphetForecast:
    """
    Facebook Prophet model for time series forecasting with seasonality
    """
    
    def __init__(self):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available. Install with: pip install prophet")
        
        self.model = None
        self.fitted = False
    
    def fit(self, data, date_column='ds', value_column='y'):
        """
        Fit Prophet model
        
        Args:
            data: DataFrame with columns ['ds', 'y'] for date and value
            date_column: Name of date column
            value_column: Name of value column
        """
        # Prepare data for Prophet
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a DataFrame with date and value columns")
        
        prophet_data = data[[date_column, value_column]].copy()
        prophet_data.columns = ['ds', 'y']
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        # Initialize and fit Prophet model
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        self.model.fit(prophet_data)
        self.fitted = True
        
        return self.model
    
    def predict(self, periods=30, freq='D'):
        """
        Generate forecasts
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of forecast ('D', 'W', 'M', etc.)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        
        return forecast
    
    def plot_forecast(self, periods=30, freq='D'):
        """
        Plot Prophet forecast with components
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of forecast
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        forecast = self.predict(periods, freq)
        
        # Main forecast plot
        fig1 = self.model.plot(forecast, figsize=(15, 6))
        plt.title('Prophet Forecast')
        plt.tight_layout()
        plt.show()
        
        # Components plot
        fig2 = self.model.plot_components(forecast, figsize=(15, 10))
        plt.tight_layout()
        plt.show()
        
        return forecast


class LSTMForecast:
    """
    LSTM neural network for time series forecasting
    """
    
    def __init__(self, lookback_window=60, lstm_units=50):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        self.lookback_window = lookback_window
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = MinMaxScaler()
        self.fitted = False
    
    def _create_sequences(self, data):
        """Create input sequences for LSTM"""
        X, y = [], []
        for i in range(self.lookback_window, len(data)):
            X.append(data[i-self.lookback_window:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def fit(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """
        Fit LSTM model
        
        Args:
            data: Time series data as array or Series
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
        """
        # Prepare data
        if isinstance(data, pd.Series):
            data = data.values
        
        data = data.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        if len(X) == 0:
            raise ValueError(f"Not enough data points. Need at least {self.lookback_window + 1} points.")
        
        # Reshape for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Build LSTM model
        self.model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.lookback_window, 1)),
            Dropout(0.2),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        self.fitted = True
        self.training_data = scaled_data
        
        return history
    
    def predict(self, steps=30):
        """
        Generate forecasts
        
        Args:
            steps: Number of steps to forecast
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use last lookback_window points for prediction
        last_sequence = self.training_data[-self.lookback_window:]
        predictions = []
        
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            X = current_sequence.reshape(1, self.lookback_window, 1)
            
            # Predict next value
            pred = self.model.predict(X, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence (rolling window)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def plot_forecast(self, data, steps=30, train_periods=200):
        """
        Plot LSTM forecast
        
        Args:
            data: Original training data
            steps: Number of steps to forecast
            train_periods: Number of historical periods to show
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        predictions = self.predict(steps)
        
        # Plot
        plt.figure(figsize=(15, 6))
        
        # Historical data
        if isinstance(data, pd.Series):
            recent_data = data.tail(train_periods)
            plt.plot(recent_data.index, recent_data.values, 'b-', label='Historical', linewidth=2)
            
            # Create future index
            future_index = pd.date_range(
                start=recent_data.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            )
        else:
            recent_data = data[-train_periods:]
            plt.plot(range(len(recent_data)), recent_data, 'b-', label='Historical', linewidth=2)
            future_index = range(len(recent_data), len(recent_data) + steps)
        
        # Forecast
        plt.plot(future_index, predictions, 'r--', label='LSTM Forecast', linewidth=2)
        
        plt.title('LSTM Forecast')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class SeasonalDecomposition:
    """
    Seasonal decomposition and analysis
    """
    
    def __init__(self):
        self.decomposition = None
    
    def decompose(self, data, model='additive', period=None):
        """
        Decompose time series into trend, seasonal, and residual components
        
        Args:
            data: Time series data
            model: 'additive' or 'multiplicative'
            period: Seasonal period (auto-detected if None)
        """
        self.decomposition = seasonal_decompose(data, model=model, period=period)
        return self.decomposition
    
    def plot_decomposition(self):
        """Plot decomposition components"""
        if self.decomposition is None:
            raise ValueError("Must decompose data first")
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 10))
        
        self.decomposition.observed.plot(ax=axes[0], title='Original')
        self.decomposition.trend.plot(ax=axes[1], title='Trend')
        self.decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        self.decomposition.resid.plot(ax=axes[3], title='Residual')
        
        plt.tight_layout()
        plt.show()


# Utility functions
def generate_sample_time_series(periods=365, trend=0.1, seasonality=True, noise_level=0.1):
    """
    Generate sample time series data for testing
    """
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    # Base trend
    trend_component = np.linspace(100, 100 + trend * periods, periods)
    
    # Seasonal component
    if seasonality:
        seasonal_component = 20 * np.sin(2 * np.pi * np.arange(periods) / 365) + \
                           10 * np.sin(2 * np.pi * np.arange(periods) / 7)
    else:
        seasonal_component = np.zeros(periods)
    
    # Noise
    noise = np.random.normal(0, noise_level * 100, periods)
    
    # Combine components
    values = trend_component + seasonal_component + noise
    values = np.maximum(values, 0)  # Ensure non-negative values
    
    return pd.Series(values, index=dates)


if __name__ == "__main__":
    # Generate sample data
    print("Generating sample time series data...")
    ts_data = generate_sample_time_series(periods=400, trend=0.05, seasonality=True)
    
    # Split into train/test
    train_size = int(0.8 * len(ts_data))
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]
    
    print(f"Training data: {len(train_data)} points")
    print(f"Test data: {len(test_data)} points")
    
    # ARIMA Model
    print("\n=== ARIMA Model ===")
    arima_model = ARIMAForecast(order=(2, 1, 2))
    arima_model.fit(train_data)
    arima_forecast = arima_model.predict(steps=len(test_data))
    print(f"ARIMA forecast generated for {len(arima_forecast)} periods")
    
    # Prophet Model (if available)
    if PROPHET_AVAILABLE:
        print("\n=== Prophet Model ===")
        prophet_data = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data.values
        })
        prophet_model = ProphetForecast()
        prophet_model.fit(prophet_data)
        prophet_forecast = prophet_model.predict(periods=len(test_data))
        print(f"Prophet forecast generated")
    
    # LSTM Model (if TensorFlow available)
    if TENSORFLOW_AVAILABLE:
        print("\n=== LSTM Model ===")
        lstm_model = LSTMForecast(lookback_window=30, lstm_units=50)
        history = lstm_model.fit(train_data, epochs=20, verbose=0)
        lstm_forecast = lstm_model.predict(steps=len(test_data))
        print(f"LSTM forecast generated for {len(lstm_forecast)} periods")
    
    # Seasonal Decomposition
    print("\n=== Seasonal Decomposition ===")
    decomp = SeasonalDecomposition()
    components = decomp.decompose(train_data, model='additive', period=365)
    print("Seasonal decomposition completed")
    
    # Calculate accuracy metrics
    if len(test_data) > 0:
        arima_mae = mean_absolute_error(test_data.values, arima_forecast['forecast'].values)
        arima_rmse = np.sqrt(mean_squared_error(test_data.values, arima_forecast['forecast'].values))
        
        print(f"\nARIMA Accuracy:")
        print(f"MAE: {arima_mae:.2f}")
        print(f"RMSE: {arima_rmse:.2f}")
    
    print("\nTime series models ready for use!") 