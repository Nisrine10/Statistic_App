import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, norm, poisson, uniform, expon

def plot_distribution(data, title):
    st.subheader(title)
    sns.set()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data, kde=True, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def generate_statistical_samples():
    st.sidebar.subheader("Statistical Distributions")
    distribution = st.sidebar.selectbox("Choose a distribution", ["Bernoulli", "Binomial", "Normal", "Poisson", "Uniform (Discrete)", "Uniform (Continuous)", "Exponential"])

    if distribution == "Bernoulli":
        p = st.sidebar.number_input('Probability of Success (p)', 0.0, 1.0, 0.5)
        data = bernoulli.rvs(p, size=1000)
        plot_distribution(data, "Bernoulli Distribution")

    elif distribution == "Binomial":
        n = st.sidebar.number_input('Number of Trials (n)', 1, 100, 10)
        p = st.sidebar.number_input('Probability of Success (p)', 0.0, 1.0, 0.5)
        k = np.arange(0, n+1)
        probability = binom.pmf(k, n, p)
        data = binom.rvs(n, p, size=1000)
        plot_distribution(data, "Binomial Distribution")

    elif distribution == "Normal":
        mu = st.sidebar.number_input('Mean (μ)', value=0.0)
        sigma = st.sidebar.number_input('Standard Deviation (σ)', value=1.0)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
        probability = norm.pdf(x, mu, sigma)
        data = norm.rvs(mu, sigma, size=1000)
        plot_distribution(data, "Normal Distribution")

    elif distribution == "Poisson":
        lam = st.sidebar.number_input('Rate (λ)', 0, 10, 2)
        k = np.arange(0, 20)
        probability = poisson.pmf(k, lam)
        data = poisson.rvs(lam, size=1000)
        plot_distribution(data, "Poisson Distribution")

    elif distribution == "Uniform (Discrete)":
        a = st.sidebar.number_input('Lower Bound (a)', value=0.0)
        b = st.sidebar.number_input('Upper Bound (b)', value=1.0)
        k = np.arange(a, b+1)
        probability = np.full_like(k, 1/(b-a))
        data = uniform.rvs(a, b - a + 1, size=1000)
        plot_distribution(data, "Uniform Distribution (Discrete)")

    elif distribution == "Uniform (Continuous)":
        a = st.sidebar.number_input('Lower Bound (a)', value=0.0)
        b = st.sidebar.number_input('Upper Bound (b)', value=1.0)
        x = np.linspace(a, b, 1000)
        probability = np.full_like(x, 1/(b-a))
        data = uniform.rvs(a, b - a, size=1000)
        plot_distribution(data, "Uniform Distribution (Continuous)")

    elif distribution == "Exponential":
        lam = st.sidebar.number_input('Rate (λ)', 0.1, 2.0, 1.0)
        x = np.linspace(0, 5, 1000)
        probability = expon.pdf(x, scale=1/lam)
        data = expon.rvs(scale=1/lam, size=1000)
        plot_distribution(data, "Exponential Distribution")

def perform_statistics_laws():
    st.title("Statistical Distribution Generator")
    generate_statistical_samples()

# Exécuter l'application
if __name__ == '__main__':
    perform_statistics_laws()
