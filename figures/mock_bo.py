import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.linalg import cholesky, cho_solve
from scipy.stats import norm
import os

# Set nice figure aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['font.size'] = 12

# Set random seed for reproducibility
np.random.seed(42)

# Create a mock 1D function (ground truth)
def true_function(x):
    return -0.8*x**3 + 3*x**2 - 4*x - 0.5

# Generate observed data points
n_observations = 6
x_observed = np.array([0.1, 0.3, 0.4, 0.6, 0.75, 0.9])
y_observed = true_function(x_observed) + np.random.normal(0, 0.05, n_observations)

# Dense grid for predictions
x_test = np.linspace(0, 1, 200)

# Actual GP implementation
def rbf_kernel(x1, x2, length_scale=0.2):
    """RBF/Gaussian kernel"""
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return np.exp(-.5 * sqdist / length_scale**2)

def predict_gp(x_train, y_train, x_test, length_scale=0.1, noise=0.001):
    """Make GP predictions with RBF kernel"""
    # Compute kernel matrices
    K = rbf_kernel(x_train.reshape(-1, 1), x_train.reshape(-1, 1), length_scale)
    K_s = rbf_kernel(x_train.reshape(-1, 1), x_test.reshape(-1, 1), length_scale)
    K_ss = rbf_kernel(x_test.reshape(-1, 1), x_test.reshape(-1, 1), length_scale)
    
    # Add noise to diagonal
    K += noise * np.eye(len(x_train))
    
    # Compute posterior mean and covariance
    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), y_train)
    mu = K_s.T @ alpha
    
    v = cho_solve((L, True), K_s)
    cov = K_ss - K_s.T @ v
    std = np.sqrt(np.diag(cov))
    
    return mu, std, cov

# Acquisition functions
def probability_improvement(mean, std, best_f, xi=0.01):
    """Probability of Improvement acquisition function"""
    z = (mean - best_f - xi) / (std + 1e-9)
    return norm.cdf(z)

def expected_improvement(mean, std, best_f, xi=0.01):
    """Expected Improvement acquisition function"""
    z = (mean - best_f - xi) / (std + 1e-9)
    return (mean - best_f - xi) * norm.cdf(z) + std * norm.pdf(z)

def upper_confidence_bound(mean, std, kappa=2.0):
    """Upper Confidence Bound acquisition function"""
    return mean + kappa * std

if __name__ == "__main__":
    # Create a figure with 4 subplots (GP + 3 acquisition functions)
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.3)

    # Perform GP regression
    mu, std, cov = predict_gp(x_observed, y_observed, x_test, length_scale=0.1, noise=0.001)

    # Generate sample functions from posterior
    n_samples = 15
    L = cholesky(cov + 1e-10 * np.eye(len(x_test)), lower=True)
    samples = mu.reshape(-1, 1) + L @ np.random.normal(0, 1, (len(x_test), n_samples))

    # Find current best observed value
    best_observed = np.max(y_observed)

    # 1. Top plot: GP surrogate model
    ax1 = fig.add_subplot(gs[0])

    # Plot sample functions from posterior
    for i in range(n_samples):
        ax1.plot(x_test, samples[:, i], 'blueviolet', alpha=0.15, linewidth=1.25)

    # Plot confidence intervals
    ax1.fill_between(x_test, mu - 2*std, mu + 2*std, color='blueviolet', alpha=0.1)

    # Plot mean prediction
    ax1.plot(x_test, mu, 'blueviolet', linewidth=4, label='GP Mean')

    # Plot observations
    ax1.scatter(x_observed, y_observed, c='yellow', s=200, zorder=10, edgecolors='k', linewidths=2)

    # Plot true function
    ax1.plot(x_test, true_function(x_test), 'k--', alpha=1)

    # Remove spines, ticks and labels for clean look
    ax1.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # 2. Probability of Improvement (PI)
    ax2 = fig.add_subplot(gs[1])
    pi_values = probability_improvement(mu, std, best_observed, xi=0.05)
    pi_color = '#FF5733'  # Orange-red

    ax2.plot(x_test, pi_values, color=pi_color, linewidth=3)
    ax2.fill_between(x_test, 0, pi_values, color=pi_color, alpha=0.3)

    # Mark the maximum point
    pi_max_x = x_test[np.argmax(pi_values)]
    pi_max_y = np.max(pi_values)

    # Remove spines, ticks
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylim(0, 1)

    # 3. Expected Improvement (EI)
    ax3 = fig.add_subplot(gs[2])
    ei_values = expected_improvement(mu, std, best_observed, xi=0.01)
    ei_color = '#33A1FF'  # Blue

    ax3.plot(x_test, ei_values, color=ei_color, linewidth=3)
    ax3.fill_between(x_test, 0, ei_values, color=ei_color, alpha=0.3)

    # Mark the maximum point
    ei_max_x = x_test[np.argmax(ei_values)]
    ei_max_y = np.max(ei_values)

    # Remove spines, ticks
    ax3.spines[['right', 'top']].set_visible(False)
    ax3.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_ylim(0, 1)

    # 4. Upper Confidence Bound (UCB)
    ax4 = fig.add_subplot(gs[3])
    kappa = 2.0
    ucb_values = upper_confidence_bound(mu, std, kappa=kappa)
    ucb_color = '#8A33FF'  # Purple

    ax4.plot(x_test, ucb_values, color=ucb_color, linewidth=3)
    ax4.fill_between(x_test, mu, ucb_values, color=ucb_color, alpha=0.3)

    # Mark the maximum point
    ucb_max_x = x_test[np.argmax(ucb_values)]
    ucb_max_y = np.max(ucb_values)

    # Remove spines, ticks
    ax4.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_ylim(min(ucb_values), max(ucb_values) + 0.2)

    plt.tight_layout()
    
    # Save figure to ./pngs/mock_bo.png
    os.makedirs('./pngs', exist_ok=True)
    plt.savefig('./pngs/mock_bo.png', dpi=300, bbox_inches='tight')
    plt.close()