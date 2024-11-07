import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta


@st.cache
def calculate_optimal_portfolio(stocks_, mkt_, sd, ed, target_return):
    # Merton Matrix Model - Min Variance Portfolio

    # Timeframe
    sd = pd.to_datetime(sd)
    ed = pd.to_datetime(ed)

    if relativedelta(ed, sd).days > 0 and relativedelta(ed, sd).years < 1:
        time_frame = 1
    else:
        time_frame = relativedelta(ed, sd).years

    # Choose stocks
    data = yf.download(stocks_, start=sd, end=ed)
    mkt_data = yf.download(mkt_, start=sd, end=ed)

    # Data treatment
    ticker = '^TNX'
    risk_free_data = yf.Ticker(ticker).history(period=f'{time_frame}y')
    rf = risk_free_data['Close'].mean() / 100

    data = data['Adj Close']

    if data.isna().sum().sum() > 0.1 * len(data):
        raise AssertionError("Too much missing data")
    else:
        data = data.dropna()

    stock_returns = data.pct_change().dropna()
    df_statistics = pd.DataFrame()
    df_statistics['Return'] = (stock_returns + 1).prod()**(252 / len(stock_returns)) - 1
    df_statistics['Std'] = stock_returns.std() * np.sqrt(252)
    mkt_returns = mkt_data['Adj Close'].pct_change().dropna()
    mkt_returns = mkt_returns.reindex(mkt_returns.index.intersection(stock_returns.index))
    sp = (mkt_returns + 1).prod()**(252 / len(mkt_returns)) - 1

    beta_ = {}
    for i in stock_returns:
        y = np.array(stock_returns[i]).reshape(-1, 1)  # Dependent variable (stock returns)
        X = np.array(mkt_returns).reshape(-1, 1)  # Independent variable (market returns)

        # Adds a column for the intercept (constant term)
        X = np.hstack((np.ones((len(X), 1)), X))
        
        # Using lstsq for numerical stability instead of inv
        beta_[i], _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    beta_only = {stock: coeff[1] for stock, coeff in beta_.items()}

    beta_df = pd.DataFrame(list(beta_only.values()), index=beta_only.keys(), columns=['Beta'])
    df_statistics['beta'] = beta_df
    df_statistics = df_statistics.T
    er = {}
    for i in df_statistics:
        er[i] = rf + df_statistics.loc['beta', i] * (sp - rf)
    df_statistics.loc['Expected Return (CAPM)'] = er.values()
    stock_cov = stock_returns.cov() * 252
    stock_cov_inv = np.linalg.inv(stock_cov)
    stock_cov_inv = pd.DataFrame(stock_cov_inv)
    stock_cov_inv.columns = stock_cov.columns
    e_ = [1] * len(stock_cov.columns)
    r_ = list(df_statistics.loc['Expected Return (CAPM)'])
    h_ = np.matmul(e_, stock_cov_inv)
    g_ = np.matmul(r_, stock_cov_inv)
    alpha = np.matmul(e_, h_.T)
    beta = np.matmul(e_, g_.T)
    gamma = np.matmul(r_, g_.T)
    delta = alpha * gamma - beta**2
    MVP = h_ / alpha
    pf_return = beta / alpha
    pf_std = np.sqrt(1 / alpha)

    risk_ = {}
    x = []
    for j in range(0, int(np.round(pf_return, 3) * 2000), 1):
        risk_[j] = np.sqrt((alpha * ((j / 1000) ** 2) - 2 * beta * (j / 1000) + gamma) / delta)
        x.append(j / 1000)

    lambda_ = (gamma - beta * target_return) / delta
    mew_ = (alpha * target_return - beta) / delta
    optimal_pf = lambda_ * h_ + mew_ * g_

    return MVP, pf_return, pf_std, optimal_pf, x, risk_, stock_cov



def black_litterman(stocks_, mkt_, sd, ed, target_return, P, Q, uncertainty=0.025):
    # Timeframe
    sd = pd.to_datetime(sd)
    ed = pd.to_datetime(ed)

    if relativedelta(ed, sd).days > 0 and relativedelta(ed, sd).years < 1:
        time_frame = 1
    else:
        time_frame = relativedelta(ed, sd).years

    # Market Cap Calculation
    caps = {}
    for i in stocks_:
        ticker = yf.Ticker(i)
        all_dates = ticker.quarterly_income_stmt.columns
        total_shares = ticker.quarterly_income_stmt[all_dates[0]]['Basic Average Shares']
        stock_price = ticker.history(start=all_dates[0]-timedelta(2, 0, 0), end=all_dates[0])['Close']
        market_cap = stock_price * total_shares
        caps[i] = market_cap.values[0]
    market_caps = caps
    total_cap = sum(market_caps.values())
    
    # Calculate Market Capitalization Weights
    df = pd.DataFrame(market_caps.items(), columns=['Stock', 'Market Cap'])
    df['Cap Ratio'] = df['Market Cap'] / total_cap
    market_cap_weights = df['Cap Ratio'].values
    tau = uncertainty

    # Stock and Market Data
    data = yf.download(stocks_, start=sd, end=ed)
    mkt_data = yf.download(mkt_, start=sd, end=ed)

    # Risk-Free Rate
    ticker = '^TNX'
    risk_free_data = yf.Ticker(ticker).history(period=f'{time_frame}y')
    rf = risk_free_data['Close'].mean() / 100

    data = data['Adj Close']
    data = data.dropna()
    stock_returns = data.pct_change().dropna()

    # Expected Returns and Covariance
    df_statistics = pd.DataFrame()
    df_statistics['Return'] = (stock_returns + 1).prod() ** (252 / len(stock_returns)) - 1
    stock_cov = stock_returns.cov() * 252

    # Implied Equilibrium Returns
    implied_eq_returns = tau * np.matmul(stock_cov, market_cap_weights)

    # Black-Litterman Adjustment
    omega = np.diag(np.diag(np.matmul(np.matmul(P, stock_cov), P.T)))
    M_inverse = np.linalg.inv(np.matmul(np.matmul(P.T, np.linalg.inv(omega)), P) + np.linalg.inv(tau * stock_cov))
    adjusted_expected_returns = M_inverse @ (np.matmul(np.matmul(P.T, np.linalg.inv(omega)), Q) + np.matmul(np.linalg.inv(tau * stock_cov), implied_eq_returns))

    # Portfolio Weights Calculation
    stock_cov_inv = np.linalg.inv(stock_cov)
    weights = np.matmul(stock_cov_inv, adjusted_expected_returns - rf) / np.sum(np.matmul(stock_cov_inv, adjusted_expected_returns - rf))

    # Portfolio Return and Variance
    portfolio_return = np.dot(weights, adjusted_expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(stock_cov, weights))

    return market_cap_weights, portfolio_return, portfolio_variance, weights



def home():
    # Streamlit app layout
    st.title("Portfolio Optimizer")

    if 'asset_class' not in st.session_state:
        st.session_state.asset_class = 'Cash'  # Default value

    # Function to handle button clicks
    def select_asset_class(class_name):
        st.session_state.asset_class = class_name

    col1, col2 = st.columns(2)

    with col1:
    # Check which button was clicked
        if st.button("Cash (Equities)"):
            select_asset_class('Cash')
    with col2:
        if st.button("Futures"):
            select_asset_class('Futures')

    # Set default values based on the selected asset class
    if st.session_state.asset_class == 'Cash':
        stocks_input = "TSLA,AAPL,MSFT,WMT,CAT"
        mkt_input = "^GSPC"
    else:  # For Futures
        st.title("Black-Litterman for Futures")

        st.header("Input Matrix P (Investor Views)")
        # Input for Matrix P
        rows = st.number_input("Number of views (rows in P)", min_value=1, value=2)
        cols = len(MVP)

        P_data = []
        for i in range(rows):
            row = st.text_input(f"Enter view {i+1} values (comma separated)", "")
            if row:
                P_data.append([float(x.strip()) for x in row.split(",")])

        P_matrix = pd.DataFrame(P_data)

        st.header("Input Vector Q (View Returns)")
        # Input for Vector Q
        Q_data = []
        for i in range(rows):
            q_value = st.number_input(f"Enter view {i+1} return", value=0.0)
            Q_data.append(q_value)

        Q_vector = np.array(Q_data)

        st.header("Input for Uncertainty (tau)")
        tau = st.number_input("Enter uncertainty for Black-Litterman model", value=0.025, min_value=0.0, max_value=1.0, step=0.001)

        # Calculate Black-Litterman adjusted portfolio weights using the provided views
        adjusted_market_cap_weights, adjusted_portfolio_return, adjusted_portfolio_variance, adjusted_weights = black_litterman(
            stocks_, mkt_, start_date, end_date, target_return, P_matrix.values, Q_vector, uncertainty=tau)

        # Display adjusted Black-Litterman portfolio results
        st.write("Adjusted Black-Litterman Portfolio Weights:")
        st.dataframe(pd.DataFrame(adjusted_weights, index=stocks_, columns=["Weights"]), width=160)  # Adjust the width as needed
        st.write(f"Adjusted Portfolio Return: {round(adjusted_portfolio_return * 100, 2)} %")
        st.write(f"Adjusted Portfolio Std Dev: {round(np.sqrt(adjusted_portfolio_variance) * 100, 2)} %")

        # Display the efficient frontier with Black-Litterman
        st.header("Black-Litterman Efficient Frontier")

        # Efficient frontier plot (same as above)
        fig = go.Figure()

        # Add the efficient frontier line
        fig.add_trace(go.Scatter(x= list(risk_.values()), y=x, mode='lines', name='Efficient Frontier'))

        # Add the adjusted Black-Litterman portfolio point
        portfolio_std_dev_bl = np.sqrt(np.dot(adjusted_weights.T, np.dot(stock_cov, adjusted_weights)))

        fig.add_trace(go.Scatter(x=[portfolio_std_dev_bl], y=[adjusted_portfolio_return], mode='markers', 
                                marker=dict(color='blue', size=12, symbol='star'),
                                name='Black-Litterman Portfolio'))

        # Update layout for Black-Litterman
        fig.update_layout(
            title="Black-Litterman Efficient Frontier",
            xaxis_title="Risk (Standard Deviation) (%)",
            yaxis_title="Return (%)",
            yaxis_tickformat=".2%", 
            xaxis_tickformat=".2%", 
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    # Display a section for additional information and conclusions
    st.write("### Additional Information")
    st.write("This tool applies modern portfolio theory to calculate an optimal portfolio based on historical data.")
    st.write("Use the Black-Litterman model to incorporate your views on the returns of different assets.")
    st.write("The efficient frontier is displayed to help visualize the trade-off between risk and return for different portfolios.")
