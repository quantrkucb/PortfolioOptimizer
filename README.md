# PortfolioOptimizer
Overview
This project is a Streamlit-based web application for optimizing portfolios using Modern Portfolio Optimization (MVO), including the Merton Matrix Model for Minimum Variance Portfolio (MVP) construction and the Black-Litterman Model for incorporating investor views. Users can input their desired stock tickers and parameters, calculate optimal portfolios, and visualize the Efficient Frontier.

Features
Portfolio Optimization using the Merton Matrix Model:

Calculate Minimum Variance Portfolio (MVP)
Calculate target return portfolio weights
Visualize the efficient frontier and optimal portfolios
Black-Litterman Model:

Incorporate investor views through matrix P (investment views) and vector Q (expected returns)
Adjust portfolio weights based on these views
Calculate the adjusted portfolio returns and risks
Dynamic Asset Class Selection:

Choose between equities and futures markets
Automatically adjust inputs based on the selected asset class
Interactive Dashboard:

Users can adjust inputs (stocks, market index, start/end dates, target return)
Real-time updates and plotting of results using Plotly graphs for visualization
