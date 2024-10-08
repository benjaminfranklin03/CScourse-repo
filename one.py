import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Setting up the Streamlit interface
def main():
    st.title('Portfolio Stress Testing & Scenario Analysis')
    st.header('Build Your Portfolio')

    # Asset tickers (for simplicity, using well-known stocks or ETFs)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BRK-B", "JPM", "V", "JNJ", "NVDA"]

    # Allow the user to allocate weights across the 10 assets
    st.subheader("Adjust Portfolio Weights")
    weights = []
    for ticker in tickers:
        weight = st.slider(f"Weight for {ticker}", min_value=0, max_value=100, value=10, step=1)
        weights.append(weight)

    # Normalize weights to sum to 100%
    weight_sum = sum(weights)
    if weight_sum > 0:
        weights = [w / weight_sum for w in weights]
    else:
        st.error("Total weight must be greater than 0.")
        return

    st.write("Your portfolio weights (normalized):", {tickers[i]: weights[i] for i in range(len(tickers))})

    # Fetch historical data for the assets
    st.subheader("Fetch Historical Data")
    start_date = st.date_input("Select Start Date", pd.to_datetime('2021-01-01'))
    end_date = st.date_input("Select End Date", pd.to_datetime('2023-01-01'))
    
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    if data.empty:
        st.error("No data available for the selected dates.")
        return

    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Portfolio return and volatility
    portfolio_returns = returns.dot(weights)
    portfolio_mean = portfolio_returns.mean() * 252  # Annualized return
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility

    st.subheader("Portfolio Performance Metrics")
    st.write(f"Expected Annual Return: {portfolio_mean:.2%}")
    st.write(f"Expected Annual Volatility: {portfolio_volatility:.2%}")

    # Calculate Sharpe Ratio
    risk_free_rate = st.number_input("Risk-Free Rate (in %):", value=2.0) / 100
    sharpe_ratio = (portfolio_mean - risk_free_rate) / portfolio_volatility
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Value at Risk (VaR) Calculation
    confidence_level = st.slider("Confidence Level for VaR (in %):", min_value=90, max_value=99, value=95, step=1)
    var = np.percentile(portfolio_returns, 100 - confidence_level)
    st.subheader("Value at Risk (VaR)")
    st.write(f"Value at Risk at {confidence_level}% confidence level: {var:.2%}")

    # Conditional Value at Risk (CVaR) Calculation
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    st.subheader("Conditional Value at Risk (CVaR)")
    st.write(f"Conditional Value at Risk at {confidence_level}% confidence level: {cvar:.2%}")

    # Scenario Analysis: Stress Testing
    st.subheader("Stress Testing & Scenario Analysis")
    scenarios = {
        "Market Crash": -0.3,  # Assume a 30% market decline
        "Economic Boom": 0.2,  # Assume a 20% market increase
        "Interest Rate Hike": -0.1,  # Assume a 10% decline due to rate hike
        "Pandemic Shock": -0.25,  # Assume a 25% decline due to pandemic shock
    }

    scenario_results = {}
    for scenario, shock in scenarios.items():
        scenario_return = portfolio_mean + shock
        scenario_results[scenario] = scenario_return

    # Display scenario results
    st.write("Scenario Analysis Results:")
    for scenario, result in scenario_results.items():
        st.write(f"{scenario}: Expected Return: {result:.2%}")

    # Correlation Heatmap
    st.subheader("Asset Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Asset Correlation Heatmap")
    st.pyplot(fig_corr)

    # Efficient Frontier Calculation
    st.subheader("Efficient Frontier")
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        random_weights = np.random.random(len(tickers))
        random_weights /= np.sum(random_weights)
        portfolio_return = np.sum(random_weights * returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(random_weights.T, np.dot(returns.cov() * 252, random_weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = sharpe_ratio

    max_sharpe_idx = np.argmax(results[2])
    st.write(f"Optimal Portfolio with Maximum Sharpe Ratio:")
    st.write(f"Return: {results[0, max_sharpe_idx]:.2%}, Volatility: {results[1, max_sharpe_idx]:.2%}, Sharpe Ratio: {results[2, max_sharpe_idx]:.2f}")

    # Plot Efficient Frontier
    fig_ef, ax_ef = plt.subplots()
    ax_ef.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o')
    ax_ef.set_title('Efficient Frontier')
    ax_ef.set_xlabel('Volatility (Risk)')
    ax_ef.set_ylabel('Return')
    st.pyplot(fig_ef)

    # Plot portfolio returns distribution
    st.subheader("Portfolio Returns Distribution")
    fig, ax = plt.subplots()
    portfolio_returns.hist(bins=50, ax=ax)
    ax.set_title("Portfolio Daily Returns Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
