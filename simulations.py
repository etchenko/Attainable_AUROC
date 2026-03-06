import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc
from statsmodels.stats.proportion import proportion_confint

# Generate survival data based on the cox model
def generate_cox_samples(n_samples, baseline_hazard, beta, X):
    risk = np.exp(np.dot(X, beta))
    U = np.random.uniform(0, 1, n_samples)
    times = -np.log(U) / (baseline_hazard * risk)
    return times

# Spit data into training and test with proper formatting
def split_data(X, time, event, train_size=0.8):
    n_samples = len(X)
    split_idx = int(n_samples * train_size)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    time_train, time_test = time[:split_idx], time[split_idx:]
    event_train, event_test = event[:split_idx], event[split_idx:]

    y_train = np.array([(e, t) for e, t in zip(event_train, time_train)], dtype=[('event', 'bool'), ('time', 'float')])
    y_test = np.array([(e, t) for e, t in zip(event_test, time_test)], dtype=[('event', 'bool'), ('time', 'float')])
    
    return X_train, y_train, X_test, y_test

# Compute the Time-dependent AUC using the Cox-Proportional Hazards model
def compute_auc(X, time, event, va_times):
    X_train, y_train, X_test, y_test = split_data(X, time, event)
    model = CoxPHSurvivalAnalysis().fit(X_train, y_train)
    risk_scores = model.predict(X_test)
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, va_times)
    return auc, mean_auc

def create_competing_cox_data(n_samples):
    np.random.seed(1)
    X = np.random.standard_normal((n_samples, 4))
    t1 = generate_cox_samples(n_samples, 0.01, [-0.9, -0.3, -0.1, 0.5], X)
    t2 = generate_cox_samples(n_samples, 0.02, [0.2, 0.7, -0.1, -0.9], X)
    C = np.random.exponential(scale=10, size=n_samples)
    time3, time2 = np.minimum(np.minimum(t1, t2), C), np.minimum(t1, C)
    event1, event2, event3 = np.array([True] * n_samples), np.zeros(n_samples, dtype=bool), np.zeros(n_samples, dtype=bool)
    event3[(t1 <= t2) & (t1 <= C)] = True
    event2[(t1 <= C)] = True
    return X, t1, event1, time2, event2, time3, event3

def create_competing_log_model(n_samples):
    np.random.seed(1)
    X = np.random.standard_normal(n_samples)
    # Latent times for Event 1 (e.g., Cancer)
    risk1 = np.exp(0.5 * X)
    T1 = -np.log(np.random.uniform(0, 1, n_samples)) / (0.01 * risk1)
    # Latent times for Event 2 (e.g., Heart Disease)
    risk2 = np.exp(0.3 * X)
    T2 = -np.log(np.random.uniform(0, 1,n_samples)) / (0.02 * risk2)
    # Right - Censoring
    C = np.full(n_samples, 100.0)
    
    # Final observed data
    time1 = np.minimum(T1, C)
    time2 = np.minimum(np.minimum(time1, T2))
    event1, event2= np.array([True]*n_samples), np.array([True]*n_samples)
    event2[(T1 <= C) & (T1<= T2)] = True
    
    return X.reshape(-1,1), time1, event1, time2, event2


# Compare time-dependent auc for survival data with and without competing risk
def compare_competing_risks_effect(n_samples, data_generating_function):
    X, t1, event1, time2, event2, time3, event3 = data_generating_function(n_samples)

    va_times = np.linspace(1, 30, 100)

    auc1, _ = compute_auc(X, t1, event1, va_times)
    auc2, _ = compute_auc(X, time2, event2, va_times)
    auc3, _ = compute_auc(X, time3, event3, va_times)

    plt.hist(t1, bins=30, alpha=0.5, label="Only Event")
    plt.hist(time2, bins=30, alpha=0.5, label="With Censoring")
    plt.hist(time3, bins=30, alpha=0.5, label="With Competing Risk")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Distribution of Event Times")
    plt.show()

    plot_multiple_lines(va_times, {"No Censoring": auc1, "Only Censoring": auc2, "Competing Risks": auc3},
                        title="Time-dependent AUC Comparison", xlabel="Time", ylabel="Time-dependent AUC")


def plot_multiple_lines(x_values, y_data_dict, title="Line Plot", xlabel="X-axis", ylabel="Y-axis", filename=None):
    """
    Plots multiple lines on the same graph.
    Parameters:
    - x_values: List or array of x-axis values.
    - y_data_dict: Dictionary where keys are labels and values are lists/arrays of y-values.
    """
    plt.figure(figsize=(10, 6))
    
    for label, y_values in y_data_dict.items():
        plt.plot(x_values, y_values, marker='o', label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    if filename:
        plt.savefig(filename)
        print(f"Plot saved successfully as {filename}")





def simulate_survival_data(n_samples, baseline_hazard, beta, X, censoring: bool = True):

    times = generate_cox_samples(n_samples, baseline_hazard, beta, X)
    # Simulate censoring
    if censoring:
        censoring_times = np.random.exponential(scale=10, size=n_samples)
        observed_times = np.minimum(times, censoring_times)
        events = times <= censoring_times
    else:        
        observed_times = times
        events = np.ones(n_samples, dtype=bool)
    
    return X, observed_times, events

def generate_competing_risks(n_samples=1000, hazard1=0.01, hazard2=0.02, beta1=0.5, beta2=-0.5, censoring: bool = True):
    X = np.random.standard_normal(n_samples)
    X = X.reshape(-1, 1)
    t1 = generate_cox_samples(n_samples, hazard1, [beta1], X.reshape(-1, 1))
    t2 = generate_cox_samples(n_samples, hazard2, [beta2], X.reshape(-1, 1))
    
    # Censoring
    if censoring:
        C = np.random.exponential(scale=10, size=n_samples)
    else:
        C = np.full(n_samples, np.inf)
    
    # Final observed data
    time = np.minimum(np.minimum(t1, t2), C)
    event = np.zeros(n_samples)
    event[(t1 <= t2) & (t1 <= C)] = 1
    event[(t2 < t1) & (t2 <= C)] = 2
    
    return X, time, event

def simulate_data(n_samples, baseline_hazard, beta, X, censoring: bool = True):
    # Simulate Data
    X, time, event = simulate_survival_data(n_samples, baseline_hazard, beta, X, censoring)
    X_train, X_test = X[:8000], X[8000:]
    time_train, time_test = time[:8000], time[8000:]
    event_train, event_test = event[:8000], event[8000:]

    y_train = np.array([(e, t) for e, t in zip(event_train, time_train)], 
                    dtype=[('event', 'bool'), ('time', 'float')])
    y_test = np.array([(e, t) for e, t in zip(event_test, time_test)], 
                    dtype=[('event', 'bool'), ('time', 'float')])
    return X_train, y_train, X_test, y_test

def simulate_competing_risks_data(n_samples, hazard1, hazard2, beta1, beta2, censoring: bool = True):
    X, time, event = generate_competing_risks(n_samples, hazard1, hazard2, beta1, beta2, censoring)
    X_train, X_test = X[:8000], X[8000:]
    time_train, time_test = time[:8000], time[8000:]
    event_train, event_test = event[:8000], event[8000:]

    y_train = np.array([(e, t) for e, t in zip(event_train, time_train)], 
                    dtype=[('event', 'int'), ('time', 'float')])
    y_test = np.array([(e, t) for e, t in zip(event_test, time_test)], 
                    dtype=[('event', 'int'), ('time', 'float')])
    return X_train, y_train, X_test, y_test

def calculate_auroc(estimator, X_train, y_train, X_test, y_test, va_times=None):
    estimator.fit(X_train, y_train)
    risk_scores = estimator.predict(X_test)
    if va_times is None:
        va_times = np.percentile(y_test["time"], np.linspace(10, 90, 15))
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, va_times)
    return va_times, auc, mean_auc

def calculate_auroc_competing_risks(estimator, X_train, y_train, X_test, y_test, va_times=None):
    # 2. Fit standard Cox model for the primary event
    model = estimator.fit(X_train, y_train)

    # 3. Predict Risk Scores
    risk_scores = model.predict(X_test)

    # 4. Define Evaluation Times
    if va_times is None:
        va_times = np.percentile(y_test["time"], np.linspace(10, 90, 15))

    # 5. Calculate Time-Dependent AUC
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, va_times)
    return va_times, auc, mean_auc


def simulate_censoring_effect():
    X = np.random.rand(10000, 2) * [50, 1]
    
    # Generate datasets first
    cens_data = simulate_data(10000, 0.01, [0.05, 0.5], X=X)
    no_cens_data = simulate_data(10000, 0.01, [0.05, 0.5], X=X, censoring=False)
    
    # Find common time range
    min_time = max(cens_data[3]["time"].min(), no_cens_data[3]["time"].min())
    max_time = min(cens_data[3]["time"].max(), no_cens_data[3]["time"].max())
    va_times = np.linspace(min_time, max_time * 0.9, 15)
    
    va_times1, auc, mean_auc = calculate_auroc(CoxPHSurvivalAnalysis(), *cens_data, va_times=va_times)
    va_times2, auc2, mean_auc2 = calculate_auroc(CoxPHSurvivalAnalysis(), *no_cens_data, va_times=va_times)

    plt.plot(va_times1, auc, marker="o", label="Censoring")
    plt.plot(va_times2, auc2, marker="o", label="No Censoring")
    plt.axhline(mean_auc, linestyle="--", color="red")
    plt.xlabel("Time")
    plt.ylabel("Time-dependent AUC")
    plt.legend()
    plt.show()

def simulate_competing_risks_effect():
    X = np.random.standard_normal(10000).reshape(10000, 1)
    
    # Generate all datasets first
    cr_data = simulate_competing_risks_data(10000, 0.01, 0.02, 0.5, -0.5)
    cens_data = simulate_data(10000, 0.01, [0.5], X=X)
    no_cens_data = simulate_data(10000, 0.01, [0.5], X=X, censoring=False)
    
    # Find the common time range across all datasets
    min_time = max(cr_data[3]["time"].min(), cens_data[3]["time"].min(), no_cens_data[3]["time"].min())
    max_time = min(cr_data[3]["time"].max(), cens_data[3]["time"].max(), no_cens_data[3]["time"].max())
    va_times = np.linspace(min_time, max_time * 0.9, 15)  # Use 90% of max to be safe
    
    va_times1, auc, mean_auc = calculate_auroc_competing_risks(CoxPHSurvivalAnalysis(), *cr_data, va_times=va_times)
    va_times2, auc2, mean_auc2 = calculate_auroc(CoxPHSurvivalAnalysis(), *cens_data, va_times=va_times)
    va_times3, auc3, mean_auc3 = calculate_auroc(CoxPHSurvivalAnalysis(), *no_cens_data, va_times=va_times)
    
    plt.plot(va_times1, auc, marker="o", label="Competing Risks")
    plt.plot(va_times2, auc2, marker="o", label="Only Censoring")
    plt.plot(va_times3, auc3, marker="o", label="No Censoring")
    plt.axhline(mean_auc, linestyle="--", color="red")
    plt.xlabel("Time")
    plt.ylabel("Time-dependent AUC")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #simulate_censoring_effect()
    #simulate_competing_risks_effect()
    #compare_competing_risks_effect(1000, create_competing_cox_data)
    compare_competing_risks_effect(10000, create_competing_log_model)
