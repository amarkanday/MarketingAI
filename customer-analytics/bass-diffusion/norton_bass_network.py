"""
Generalized Norton-Bass Model with Network Effects

This module implements the Norton-Bass model for multi-generational product diffusion
with network effects. The Norton-Bass model extends the classical Bass model to handle
successive product generations and technology substitution effects.

Author: Marketing Analytics Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import odeint
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NortonBassNetworkModel:
    """
    Generalized Norton-Bass Model with Network Effects
    
    This class implements the Norton-Bass model for multi-generational products
    with network effects, allowing for:
    - Multiple product generations with substitution effects
    - Social network influence on adoption decisions
    - Heterogeneous adoption propensities across network positions
    - Dynamic network evolution over time
    """
    
    def __init__(self, n_generations: int = 3, network_size: int = 1000):
        """
        Initialize the Norton-Bass Network Model
        
        Parameters:
        -----------
        n_generations : int
            Number of product generations to model
        network_size : int
            Size of the social network
        """
        self.n_generations = n_generations
        self.network_size = network_size
        self.generations = {}
        self.network = None
        self.network_influence = None
        self.fitted_params = {}
        
    def create_network(self, 
                      network_type: str = 'small_world',
                      **network_params) -> nx.Graph:
        """
        Create a social network for diffusion modeling
        
        Parameters:
        -----------
        network_type : str
            Type of network ('small_world', 'scale_free', 'random', 'complete')
        **network_params : dict
            Additional parameters for network generation
            
        Returns:
        --------
        nx.Graph : Generated network
        """
        np.random.seed(42)  # For reproducibility
        
        if network_type == 'small_world':
            k = network_params.get('k', 6)
            p = network_params.get('p', 0.3)
            self.network = nx.watts_strogatz_graph(self.network_size, k, p)
            
        elif network_type == 'scale_free':
            m = network_params.get('m', 3)
            self.network = nx.barabasi_albert_graph(self.network_size, m)
            
        elif network_type == 'random':
            p = network_params.get('p', 0.01)
            self.network = nx.erdos_renyi_graph(self.network_size, p)
            
        elif network_type == 'complete':
            self.network = nx.complete_graph(self.network_size)
            
        else:
            raise ValueError(f"Unknown network type: {network_type}")
            
        # Calculate network influence metrics
        self._calculate_network_influence()
        
        return self.network
    
    def _calculate_network_influence(self):
        """Calculate network influence metrics for each node"""
        if self.network is None:
            raise ValueError("Network not created yet. Call create_network() first.")
            
        # Calculate various centrality measures
        degree_centrality = nx.degree_centrality(self.network)
        betweenness_centrality = nx.betweenness_centrality(self.network)
        closeness_centrality = nx.closeness_centrality(self.network)
        eigenvector_centrality = nx.eigenvector_centrality(self.network)
        
        # Combine centrality measures into influence score
        self.network_influence = {}
        for node in self.network.nodes():
            influence = (
                0.3 * degree_centrality[node] +
                0.2 * betweenness_centrality[node] +
                0.2 * closeness_centrality[node] +
                0.3 * eigenvector_centrality[node]
            )
            self.network_influence[node] = influence
    
    def norton_bass_equations(self, 
                             adoptions: np.ndarray, 
                             t: float, 
                             params: Dict) -> np.ndarray:
        """
        System of differential equations for Norton-Bass model
        
        Parameters:
        -----------
        adoptions : np.ndarray
            Current adoption levels for each generation
        t : float
            Current time
        params : dict
            Model parameters
            
        Returns:
        --------
        np.ndarray : Rate of change for each generation
        """
        n_gen = len(adoptions)
        dadoptions_dt = np.zeros(n_gen)
        
        for i in range(n_gen):
            # Extract parameters for generation i
            m_i = params[f'm_{i}']  # Market potential
            p_i = params[f'p_{i}']  # Innovation coefficient
            q_i = params[f'q_{i}']  # Imitation coefficient
            
            # Network effect modifier
            network_effect = self._calculate_network_effect(adoptions[i], t, i)
            
            # Current non-adopters
            remaining_market = m_i - adoptions[i]
            
            if remaining_market <= 0:
                dadoptions_dt[i] = 0
                continue
            
            # Basic diffusion force
            diffusion_force = (p_i + q_i * (adoptions[i] / m_i)) * remaining_market
            
            # Apply network effects
            diffusion_force *= network_effect
            
            # Substitution effects from newer generations
            substitution_effect = 0
            for j in range(i + 1, n_gen):
                if f'sub_{j}_{i}' in params:
                    sub_rate = params[f'sub_{j}_{i}']
                    substitution_effect += sub_rate * adoptions[j] * (adoptions[i] / m_i)
            
            # Generation launch delay
            launch_time = params.get(f'launch_{i}', 0)
            if t < launch_time:
                diffusion_force = 0
            
            dadoptions_dt[i] = diffusion_force - substitution_effect
            
        return dadoptions_dt
    
    def _calculate_network_effect(self, 
                                 current_adoption: float, 
                                 t: float, 
                                 generation: int) -> float:
        """
        Calculate network effect multiplier based on current adoption and network structure
        
        Parameters:
        -----------
        current_adoption : float
            Current adoption level
        t : float
            Current time
        generation : int
            Product generation index
            
        Returns:
        --------
        float : Network effect multiplier
        """
        if self.network_influence is None:
            return 1.0
        
        # Calculate adoption probability based on network influence
        avg_influence = np.mean(list(self.network_influence.values()))
        adoption_rate = current_adoption / self.network_size if self.network_size > 0 else 0
        
        # Network effect increases with adoption and network connectivity
        network_multiplier = 1 + 0.5 * avg_influence * adoption_rate
        
        # Add network clustering effect
        clustering_coeff = nx.average_clustering(self.network)
        cluster_effect = 1 + 0.3 * clustering_coeff * adoption_rate
        
        return network_multiplier * cluster_effect
    
    def simulate_diffusion(self, 
                          params: Dict, 
                          time_periods: int = 100,
                          dt: float = 0.1) -> pd.DataFrame:
        """
        Simulate multi-generational diffusion with network effects
        
        Parameters:
        -----------
        params : dict
            Model parameters
        time_periods : int
            Number of time periods to simulate
        dt : float
            Time step size
            
        Returns:
        --------
        pd.DataFrame : Simulation results
        """
        # Time vector
        t = np.arange(0, time_periods, dt)
        
        # Initial conditions (all generations start with 0 adopters)
        initial_adoptions = np.zeros(self.n_generations)
        
        # Solve the system of ODEs
        solution = odeint(self.norton_bass_equations, initial_adoptions, t, args=(params,))
        
        # Create results DataFrame
        columns = [f'Generation_{i+1}' for i in range(self.n_generations)]
        results_df = pd.DataFrame(solution, columns=columns)
        results_df['Time'] = t
        
        # Calculate cumulative and incremental adoptions
        for i in range(self.n_generations):
            col = f'Generation_{i+1}'
            results_df[f'{col}_Incremental'] = results_df[col].diff().fillna(0)
            results_df[f'{col}_MarketShare'] = results_df[col] / params[f'm_{i}']
        
        # Total adoption across all generations
        gen_cols = [f'Generation_{i+1}' for i in range(self.n_generations)]
        results_df['Total_Adoption'] = results_df[gen_cols].sum(axis=1)
        
        return results_df
    
    def estimate_parameters(self, 
                           data: pd.DataFrame,
                           method: str = 'mle') -> Dict:
        """
        Estimate Norton-Bass model parameters from data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical adoption data with columns for each generation
        method : str
            Estimation method ('mle', 'nls')
            
        Returns:
        --------
        dict : Estimated parameters
        """
        def objective_function(params_vector):
            # Convert parameter vector to dictionary
            params_dict = self._vector_to_params(params_vector)
            
            # Simulate with current parameters
            simulated = self.simulate_diffusion(params_dict, len(data))
            
            # Calculate error
            error = 0
            for i in range(self.n_generations):
                col = f'Generation_{i+1}'
                if col in data.columns:
                    observed = data[col].values
                    predicted = simulated[col].values[:len(observed)]
                    error += np.sum((observed - predicted) ** 2)
            
            return error
        
        # Initial parameter guess
        initial_params = self._get_initial_params(data)
        param_vector = self._params_to_vector(initial_params)
        
        # Parameter bounds
        bounds = self._get_parameter_bounds()
        
        # Optimize
        if method == 'mle':
            result = differential_evolution(objective_function, bounds, seed=42)
        else:
            result = minimize(objective_function, param_vector, bounds=bounds)
        
        # Convert back to parameter dictionary
        self.fitted_params = self._vector_to_params(result.x)
        
        return self.fitted_params
    
    def _get_initial_params(self, data: pd.DataFrame) -> Dict:
        """Get initial parameter estimates"""
        params = {}
        
        for i in range(self.n_generations):
            col = f'Generation_{i+1}'
            if col in data.columns:
                max_adoption = data[col].max()
                params[f'm_{i}'] = max_adoption * 1.2  # Market potential
                params[f'p_{i}'] = 0.01  # Innovation coefficient
                params[f'q_{i}'] = 0.1   # Imitation coefficient
                params[f'launch_{i}'] = i * 10  # Launch timing
                
                # Substitution parameters
                for j in range(i):
                    params[f'sub_{i}_{j}'] = 0.05
        
        return params
    
    def _params_to_vector(self, params: Dict) -> np.ndarray:
        """Convert parameter dictionary to vector for optimization"""
        vector = []
        for i in range(self.n_generations):
            vector.extend([params[f'm_{i}'], params[f'p_{i}'], params[f'q_{i}']])
            if i > 0:
                vector.append(params[f'launch_{i}'])
                for j in range(i):
                    vector.append(params[f'sub_{i}_{j}'])
        return np.array(vector)
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict:
        """Convert parameter vector back to dictionary"""
        params = {}
        idx = 0
        
        for i in range(self.n_generations):
            params[f'm_{i}'] = vector[idx]
            params[f'p_{i}'] = vector[idx + 1]
            params[f'q_{i}'] = vector[idx + 2]
            idx += 3
            
            if i > 0:
                params[f'launch_{i}'] = vector[idx]
                idx += 1
                for j in range(i):
                    params[f'sub_{i}_{j}'] = vector[idx]
                    idx += 1
            else:
                params[f'launch_{i}'] = 0
        
        return params
    
    def _get_parameter_bounds(self) -> List[Tuple]:
        """Get parameter bounds for optimization"""
        bounds = []
        
        for i in range(self.n_generations):
            bounds.extend([
                (1, 1e6),      # Market potential
                (0.001, 0.1),  # Innovation coefficient
                (0.01, 1.0)    # Imitation coefficient
            ])
            
            if i > 0:
                bounds.append((0, 50))  # Launch timing
                for j in range(i):
                    bounds.append((0, 0.2))  # Substitution rate
        
        return bounds
    
    def forecast(self, 
                periods: int = 50,
                confidence_intervals: bool = True) -> pd.DataFrame:
        """
        Generate forecasts using fitted parameters
        
        Parameters:
        -----------
        periods : int
            Number of periods to forecast
        confidence_intervals : bool
            Whether to include confidence intervals
            
        Returns:
        --------
        pd.DataFrame : Forecast results
        """
        if not self.fitted_params:
            raise ValueError("Model not fitted. Call estimate_parameters() first.")
        
        forecast_df = self.simulate_diffusion(self.fitted_params, periods)
        
        if confidence_intervals:
            # Add uncertainty through parameter perturbation
            n_simulations = 100
            forecasts = []
            
            for _ in range(n_simulations):
                # Perturb parameters
                perturbed_params = self._perturb_parameters(self.fitted_params)
                sim_forecast = self.simulate_diffusion(perturbed_params, periods)
                forecasts.append(sim_forecast)
            
            # Calculate confidence intervals
            for i in range(self.n_generations):
                col = f'Generation_{i+1}'
                values = np.array([f[col].values for f in forecasts])
                
                forecast_df[f'{col}_CI_Lower'] = np.percentile(values, 2.5, axis=0)
                forecast_df[f'{col}_CI_Upper'] = np.percentile(values, 97.5, axis=0)
        
        return forecast_df
    
    def _perturb_parameters(self, params: Dict, noise_level: float = 0.1) -> Dict:
        """Add noise to parameters for uncertainty quantification"""
        perturbed = params.copy()
        
        for key, value in params.items():
            if isinstance(value, (int, float)):
                noise = np.random.normal(0, noise_level * abs(value))
                perturbed[key] = max(0.001, value + noise)  # Ensure positive values
        
        return perturbed
    
    def plot_diffusion_curves(self, 
                             results: pd.DataFrame,
                             show_network_metrics: bool = True,
                             figsize: Tuple = (15, 10)) -> plt.Figure:
        """
        Plot multi-generational diffusion curves with network analysis
        
        Parameters:
        -----------
        results : pd.DataFrame
            Simulation or forecast results
        show_network_metrics : bool
            Whether to show network analysis plots
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure : Generated figure
        """
        if show_network_metrics and self.network is not None:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()
        
        # Plot 1: Cumulative adoption by generation
        ax1 = axes[0]
        for i in range(self.n_generations):
            col = f'Generation_{i+1}'
            if col in results.columns:
                ax1.plot(results['Time'], results[col], 
                        label=f'Generation {i+1}', linewidth=2)
                
                # Add confidence intervals if available
                ci_lower = f'{col}_CI_Lower'
                ci_upper = f'{col}_CI_Upper'
                if ci_lower in results.columns and ci_upper in results.columns:
                    ax1.fill_between(results['Time'], 
                                   results[ci_lower], 
                                   results[ci_upper], 
                                   alpha=0.2)
        
        ax1.set_title('Cumulative Adoption by Generation')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Adopters')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Market share evolution
        ax2 = axes[1]
        for i in range(self.n_generations):
            col = f'Generation_{i+1}_MarketShare'
            if col in results.columns:
                ax2.plot(results['Time'], results[col], 
                        label=f'Generation {i+1}', linewidth=2)
        
        ax2.set_title('Market Share Evolution')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Market Share')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Incremental adoption (adoption rate)
        ax3 = axes[2]
        for i in range(self.n_generations):
            col = f'Generation_{i+1}_Incremental'
            if col in results.columns:
                ax3.plot(results['Time'], results[col], 
                        label=f'Generation {i+1}', linewidth=2)
        
        ax3.set_title('Adoption Rate by Generation')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('New Adopters per Period')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Total adoption across all generations
        ax4 = axes[3]
        if 'Total_Adoption' in results.columns:
            ax4.plot(results['Time'], results['Total_Adoption'], 
                    'k-', linewidth=3, label='Total Adoption')
            ax4.set_title('Total Adoption Across All Generations')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Total Adopters')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Network analysis plots (if network exists)
        if show_network_metrics and self.network is not None and len(axes) > 4:
            # Plot 5: Network structure
            ax5 = axes[4]
            pos = nx.spring_layout(self.network, seed=42)
            
            # Color nodes by influence
            node_colors = [self.network_influence[node] for node in self.network.nodes()]
            
            nx.draw(self.network, pos, ax=ax5, 
                   node_color=node_colors, 
                   node_size=30,
                   cmap='viridis',
                   with_labels=False,
                   edge_color='gray',
                   alpha=0.7)
            ax5.set_title('Social Network Structure\n(Color = Influence Score)')
            
            # Plot 6: Network metrics
            ax6 = axes[5]
            
            # Calculate network metrics
            degree_dist = [self.network.degree(n) for n in self.network.nodes()]
            influence_dist = list(self.network_influence.values())
            
            ax6.scatter(degree_dist, influence_dist, alpha=0.6)
            ax6.set_xlabel('Node Degree')
            ax6.set_ylabel('Influence Score')
            ax6.set_title('Degree vs Influence Score')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_network_effects(self) -> Dict:
        """
        Analyze network structure and its impact on diffusion
        
        Returns:
        --------
        dict : Network analysis results
        """
        if self.network is None:
            raise ValueError("Network not created. Call create_network() first.")
        
        analysis = {}
        
        # Basic network metrics
        analysis['network_size'] = self.network.number_of_nodes()
        analysis['network_edges'] = self.network.number_of_edges()
        analysis['density'] = nx.density(self.network)
        analysis['average_clustering'] = nx.average_clustering(self.network)
        
        # Centrality statistics
        influence_values = list(self.network_influence.values())
        analysis['avg_influence'] = np.mean(influence_values)
        analysis['influence_std'] = np.std(influence_values)
        analysis['influence_range'] = max(influence_values) - min(influence_values)
        
        # Network structure analysis
        analysis['diameter'] = nx.diameter(self.network) if nx.is_connected(self.network) else 'Disconnected'
        analysis['average_path_length'] = nx.average_shortest_path_length(self.network) if nx.is_connected(self.network) else 'N/A'
        
        # Degree distribution
        degrees = [self.network.degree(n) for n in self.network.nodes()]
        analysis['avg_degree'] = np.mean(degrees)
        analysis['degree_std'] = np.std(degrees)
        
        return analysis
    
    def generate_sample_data(self, 
                            periods: int = 50,
                            noise_level: float = 0.1) -> pd.DataFrame:
        """
        Generate sample multi-generational adoption data with network effects
        
        Parameters:
        -----------
        periods : int
            Number of time periods
        noise_level : float
            Amount of noise to add to the data
            
        Returns:
        --------
        pd.DataFrame : Sample data
        """
        # Create a sample network
        self.create_network('small_world', k=6, p=0.3)
        
        # Sample parameters for demonstration
        sample_params = {}
        
        for i in range(self.n_generations):
            sample_params[f'm_{i}'] = (i + 1) * 1000  # Increasing market potential
            sample_params[f'p_{i}'] = 0.01 + i * 0.005  # Slightly increasing innovation
            sample_params[f'q_{i}'] = 0.15 - i * 0.02   # Decreasing imitation (word-of-mouth)
            sample_params[f'launch_{i}'] = i * 15       # Staggered launches
            
            # Substitution effects (newer generations cannibalize older ones)
            for j in range(i):
                sample_params[f'sub_{i}_{j}'] = 0.1 * (1 - 0.3 * (i - j - 1))
        
        # Simulate clean data
        clean_data = self.simulate_diffusion(sample_params, periods)
        
        # Add noise
        sample_data = clean_data.copy()
        for i in range(self.n_generations):
            col = f'Generation_{i+1}'
            if col in sample_data.columns:
                noise = np.random.normal(0, noise_level * sample_data[col].std(), len(sample_data))
                sample_data[col] += noise
                sample_data[col] = np.maximum(0, sample_data[col])  # Ensure non-negative
        
        return sample_data


def demo_norton_bass_network():
    """Demonstration of Norton-Bass model with network effects"""
    print("Norton-Bass Model with Network Effects - Demonstration")
    print("=" * 60)
    
    # Initialize model
    model = NortonBassNetworkModel(n_generations=3, network_size=500)
    
    # Generate sample data
    print("1. Generating sample multi-generational adoption data...")
    sample_data = model.generate_sample_data(periods=60, noise_level=0.05)
    
    # Analyze network
    print("2. Analyzing social network structure...")
    network_analysis = model.analyze_network_effects()
    
    print(f"   Network size: {network_analysis['network_size']}")
    print(f"   Network density: {network_analysis['density']:.4f}")
    print(f"   Average clustering: {network_analysis['average_clustering']:.4f}")
    print(f"   Average influence: {network_analysis['avg_influence']:.4f}")
    
    # Estimate parameters
    print("3. Estimating model parameters...")
    estimated_params = model.estimate_parameters(sample_data)
    
    print("   Estimated parameters:")
    for gen in range(model.n_generations):
        print(f"   Generation {gen + 1}:")
        print(f"     Market Potential: {estimated_params[f'm_{gen}']:.0f}")
        print(f"     Innovation Coeff: {estimated_params[f'p_{gen}']:.4f}")
        print(f"     Imitation Coeff:  {estimated_params[f'q_{gen}']:.4f}")
    
    # Generate forecast
    print("4. Generating forecasts...")
    forecast = model.forecast(periods=80, confidence_intervals=True)
    
    # Create visualizations
    print("5. Creating visualizations...")
    fig = model.plot_diffusion_curves(forecast, show_network_metrics=True)
    plt.show()
    
    # Print some insights
    print("6. Model Insights:")
    total_adoption_final = forecast['Total_Adoption'].iloc[-1]
    print(f"   Total projected adoption: {total_adoption_final:.0f}")
    
    # Find peak adoption rates for each generation
    for i in range(model.n_generations):
        col = f'Generation_{i+1}_Incremental'
        if col in forecast.columns:
            peak_rate = forecast[col].max()
            peak_time = forecast.loc[forecast[col].idxmax(), 'Time']
            print(f"   Generation {i+1} peak adoption rate: {peak_rate:.1f} at time {peak_time:.1f}")


if __name__ == "__main__":
    demo_norton_bass_network() 