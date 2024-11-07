import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta


@st.cache_data
def calculate_optimal_portfolio(stocks_, mkt_, sd, ed, target_return):
    # Merton Matrix Model - Min Variance Portfolio

    # Timeframe
    sd = pd.Timestamp(sd)
    ed = pd.Timestamp(ed)

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
        raise AssertionError
    else:
        data = data.dropna()

    stock_returns = data.pct_change().dropna()
    df_statistics = pd.DataFrame()
    df_statistics['Return'] = (stock_returns + 1).prod()**(252 / len(stock_returns)) - 1
    df_statistics['Std'] = stock_returns.std() * np.sqrt(252)
    mkt_returns = mkt_data['Adj Close'].pct_change().dropna()
    mkt_returns = mkt_returns.reindex(mkt_returns.index.intersection(stock_returns.index))
    sp = (mkt_returns + 1).product()**(252 / len(mkt_returns)) - 1

    beta_ = {}
    for i in stock_returns:
        y = np.array(stock_returns[i]).reshape(-1, 1)  # Dependent variable (stock returns)
        X = np.array(mkt_returns).reshape(-1, 1)  # Independent variable (market returns)

        # Adds a column for the intercept (constant term)
        X = np.hstack((np.ones((len(X), 1)), X))
        beta_[i] = np.linalg.inv(X.T @ X) @ (X.T @ y)

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
    # r_ = np.array(r_).reshape(1, -1) 
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
    sd = pd.Timestamp(sd)
    ed = pd.Timestamp(ed)

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
        caps[i] = (market_cap.values[0])
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

    # Create two columns for buttons

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
        stocks_input = "CL=F,GC=F,SI=F"
        mkt_input = "GSG"

    # User inputs
    stocks_input = st.text_input("Enter stock tickers (comma separated)", stocks_input)
    mkt_input = st.text_input("Enter market index", mkt_input)
    start_date = st.date_input("Start Date", pd.Timestamp('2010-01-01').date())
    end_date = st.date_input("End Date", pd.Timestamp('2020-01-01').date())
    target_return = st.number_input("Target Annualised Return (%)", value=20.0, format="%.1f", step=0.1) / 100  # Convert to decimal
    # Convert input string into a list
    stocks_ = [stock.strip() for stock in stocks_input.split(",")]
    mkt_ = [stock.strip() for stock in mkt_input.split(",")]

    # Calculate optimal portfolio and efficient frontier
    MVP, pf_return, pf_std, optimal_pf, x, risk_, stock_cov = calculate_optimal_portfolio(stocks_, mkt_, start_date, end_date, target_return)

    # Calculate portfolio risk
    portfolio_std_dev = np.sqrt(np.dot(optimal_pf.T, np.dot(stock_cov, optimal_pf)))

    # Plot the efficient frontier
    fig = go.Figure()

    # Add the efficient frontier line
    fig.add_trace(go.Scatter(x= list(risk_.values()), y=x, mode='lines', name='Efficient Frontier'))

    # Add the optimal portfolio point
    fig.add_trace(go.Scatter(x=[portfolio_std_dev], y=[target_return], mode='markers',  # Convert back to percentage for display
                            marker=dict(color='red', size=12, symbol='star'),
                            name='Target Portfolio'))

    # Update layout
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Risk (Standard Deviation) (%)",
        yaxis_title="Return (%)",
        # yaxis_tickformat='%',
        yaxis_tickformat=".2%", 
        xaxis_tickformat=".2%", 
        showlegend=True
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Create two columns for layout
    col1, col2 = st.columns(2)

    # Display Min Variance Portfolio in the first column
    with col1:
        st.write("Min Variance Portfolio:")
        st.dataframe(pd.DataFrame(MVP, index=stocks_, columns=["Weights"]), width=160)  # Adjust the width as needed
        st.write(f"Portfolio Return: {'&nbsp;'*3} {'&nbsp;'*3} {round(pf_return * 100, 2)} %")  # Convert back to percentage for display
        st.write(f"Portfolio Std Dev: {'&nbsp;'*3} {'&nbsp;'*3} {round(pf_std * 100, 2)} %")  # Convert back to percentage for display

    # Display Target Return Portfolio Weights in the second column
    with col2:
        st.write("Target Return Portfolio Weights:")
        st.dataframe(pd.DataFrame(optimal_pf, index=stocks_, columns=["Weights"]), width=160)  # Adjust the width as needed
        st.write(f"Portfolio Return: {'&nbsp;'*3} {'&nbsp;'*3} {round(target_return * 100, 2)} %")  # Convert back to percentage for display
        st.write(f"Portfolio Std Dev: {'&nbsp;'*3} {'&nbsp;'*3} {round(np.sqrt(np.matmul(optimal_pf, np.matmul(stock_cov, optimal_pf.T))) * 100, 2)} %")  
                # Convert back to percentage for display

    if st.session_state.asset_class == 'Cash':
        st.title("Black-Litterman")

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

        st.header("Input Vector Q (Expected Returns)")
        Q_data = []
        for i in range(rows):
            q_value = st.number_input(f"Expected return for view {i+1}", value=0.0)
            Q_data.append(q_value)

        Q_vector = pd.Series(Q_data)

        if st.button("Submit"):
            if len(P_data) == rows and all(len(row) == cols for row in P_data):
                # st.subheader("Matrix P")
                # st.write(P_matrix)
                # st.subheader("Vector Q")
                # st.write(Q_vector)

                # Optional: Display the result
                st.success("Matrices successfully entered!")
                market_cap_weights, pf_return, pf_std, optimal_pf = black_litterman(stocks_, mkt_, start_date, end_date, target_return, P_matrix, Q_vector)

                # Calculate portfolio risk
                portfolio_std_dev = np.sqrt(np.dot(optimal_pf.T, np.dot(stock_cov, optimal_pf)))

                st.write("Optimal Weights:")
                st.dataframe(pd.DataFrame(optimal_pf, index=stocks_, columns=["Weights"]), width=160)  # Adjust the width as needed
                st.write(f"Portfolio Return: {'&nbsp;'*3} {'&nbsp;'*3} {round(pf_return * 100, 2)} %")  # Convert back to percentage for display
                st.write(f"Portfolio Std Dev: {'&nbsp;'*3} {'&nbsp;'*3} {round(portfolio_std_dev * 100, 2)} %")  # Convert back to percentage for display
                # st.dataframe(pd.DataFrame(market_cap_weights, index=stocks_, columns=["Weights"]), width=160)  # Adjust the width as needed

            else:
                st.error("Error: Please check the dimensions of Matrix P and the number of entries in Vector Q.")


def page_one():
    st.title("hi")
def page_two():
    st.title("Hi")


st.sidebar.title("Portfolio Construction and Management")
page = st.sidebar.radio("Select a page:", ["Optimizer: MVO (+Black Litterman)", "Covariance Shrinkage (Ledoit and Wolf", "Page Two"])

# Render the selected page
if page == "Optimizer: MVO (+Black Litterman)":
    home()
elif page == "Page One":
    page_one()
elif page == "Page Two":
    page_two()
