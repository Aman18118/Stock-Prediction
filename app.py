import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# New dependency: A dedicated library for fetching stock ticker symbols.
# You'll need to install it: pip install pytickersymbols
from pytickersymbols import PyTickerSymbols

# --- Page Configuration ---
st.set_page_config(
    page_title="QuantEdge | AI Forecast",
    page_icon="🤖",
    layout="wide"
)

# --- Custom CSS for Professional UI ---
st.markdown("""
<style>
    /* Main container and background */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    body {
        background-color: #0E1117;
    }
    /* Base card style */
    .metric-card, .news-article, .stTabs [data-baseweb="tab-panel"], .mover-card {
        background-color: #1B2133; /* Darker card background */
        border-radius: 10px;
        border: 1px solid #2C344B;
        padding: 25px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.2);
        color: #FAFAFA;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    /* Metric Card specific styles */
    .metric-card {
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center; /* Center content vertically */
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px 0 rgba(0, 163, 255, 0.2);
    }
    .metric-card h4 {
        color: #888;
        margin-bottom: 12px;
        font-size: 1.1rem;
        font-weight: 500;
        text-align: center;
    }
    .metric-card p {
        font-size: 1.6rem;
        font-weight: 600;
        text-align: center;
    }
    [data-testid="stMetricLabel"] { color: #FFFFFF !important; }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; }
    
    /* Company Header */
    .company-header { display: flex; align-items: center; gap: 20px; margin-bottom: 2rem; }
    .company-logo { width: 70px; height: 70px; border-radius: 50%; object-fit: contain; border: 2px solid #2C344B; }
    .company-name h2 { font-size: 2.5rem; font-weight: bold; margin: 0; }
    .company-name p { font-size: 1.2rem; color: #888; margin: 0; }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 2px solid #2C344B; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: transparent; border-radius: 8px 8px 0 0; padding: 10px 15px; border: none; color: #B0B0B0; }
    .stTabs [aria-selected="true"] { background-color: #1B2133; color: #4A90E2; border-bottom: 3px solid #4A90E2; }
    
    /* --- NEW: Market Mover Card UI --- */
    .mover-card {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100%;
        padding: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .mover-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0, 163, 255, 0.2);
    }
    .mover-symbol { font-weight: bold; font-size: 1.2em; margin-bottom: 8px; text-align: center; }
    .gainer { color: #28a745; font-size: 1.4em; font-weight: 600; }
    .loser { color: #dc3545; font-size: 1.4em; font-weight: 600; }
    
    /* Footer and Buttons */
    .stButton>button { border: 1px solid #4A90E2; border-radius: 10px; color: #FFFFFF; background-color: #4A90E2; padding: 10px 25px; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #FFFFFF; color: #4A90E2; border-color: #FFFFFF; }
    [data-testid="stAlert"] { background-color: #1B2133; border: 1px solid #2C344B; border-radius: 10px; }
    [data-testid="stAlert"] *, [data-testid="stAlert"] p, [data-testid="stAlert"] div { color: #FFFFFF !important; }
    .footer { margin-top: 3rem; padding: 2rem; background-color: #1B2133; border-top: 1px solid #2C344B; text-align: center; color: #FAFAFA; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# --- Constants & Global Variables ---
START_DATE = "2015-01-01"
TODAY = date.today()
TIME_STEP = 60
# A user agent is required for the NSE India fetch to work correctly
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}


# --- Initialize Session State ---
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'evaluation_data' not in st.session_state:
    st.session_state.evaluation_data = None
if 'moving_averages' not in st.session_state:
    st.session_state.moving_averages = []
if 'ma_id_counter' not in st.session_state:
    st.session_state.ma_id_counter = 0
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = ""

# --- Dynamic Index Ticker Fetching ---
@st.cache_data(ttl=86400) # Cache for 24 hours
def get_nifty50_tickers():
    """Fetches NIFTY 50 tickers from the official NSE data file."""
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
        # The storage_options dictionary is used to pass the User-Agent header
        df = pd.read_csv(url, storage_options={'User-Agent': HEADERS['User-Agent']})
        # Append ".NS" as required by yfinance for Indian stocks
        return (df['Symbol'] + ".NS").tolist()
    except Exception as e:
        st.error(f"Could not fetch NIFTY 50 tickers: {e}", icon="🚨")
        return []

@st.cache_data(ttl=86400) # Cache for 24 hours
def get_nasdaq100_tickers():
    """Fetches NASDAQ 100 tickers using the pytickersymbols library."""
    try:
        stock_data = PyTickerSymbols()
        nasdaq_100 = list(stock_data.get_stocks_by_index('NASDAQ 100'))
        return [stock['symbol'] for stock in nasdaq_100]
    except Exception as e:
        st.error(f"Could not fetch NASDAQ 100 tickers: {e}", icon="🚨")
        return []

@st.cache_data(ttl=86400) # Cache for 24 hours
def get_sp500_tickers():
    """Fetches S&P 500 tickers using the pytickersymbols library."""
    try:
        stock_data = PyTickerSymbols()
        sp_500 = list(stock_data.get_stocks_by_index('S&P 500'))
        return [stock['symbol'] for stock in sp_500]
    except Exception as e:
        st.error(f"Could not fetch S&P 500 tickers: {e}", icon="🚨")
        return []

# --- Data Loading & Processing ---
@st.cache_data(ttl=3600)
def load_data(ticker_input: str):
    """
    Loads historical data and company info for a given ticker.
    This version is more robust against API failures.
    """
    tickers_to_try = [ticker_input]
    if not ticker_input.endswith(".NS"):
        tickers_to_try.append(f"{ticker_input}.NS")

    df = None
    final_ticker = None

    for ticker in tickers_to_try:
        try:
            stock_data = yf.Ticker(ticker)
            history_df = stock_data.history(start=START_DATE, end=TODAY)
            if not history_df.empty:
                df = history_df
                final_ticker = ticker
                break
        except Exception:
            continue
    
    if df is None or df.empty:
        return None, None, None

    info = {}
    try:
        info = yf.Ticker(final_ticker).info
    except Exception:
        pass # It's okay if info fails, we can proceed without it

    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    return df, info, final_ticker

@st.cache_data(ttl=60)
def get_latest_price(ticker: str):
    """Fetches the most recent price data for a stock."""
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return data['Close'].iloc[-1]
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_market_movers(tickers):
    """Fetches data for a list of tickers and returns top 5 gainers and losers."""
    if not tickers:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    data = yf.download(tickers, period="2d", progress=False)['Close']
    if data.empty or len(data) < 2:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    
    pct_change = data.pct_change().iloc[-1] * 100
    gainers = pct_change.nlargest(5)
    losers = pct_change.nsmallest(5)
    return gainers, losers

# --- Model & Technical Analysis ---
@st.cache_resource
def get_trained_model(df):
    """Builds, trains, and returns a cached LSTM model."""
    close_data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]

    X_train, y_train = [], []
    for i in range(TIME_STEP, len(train_data)):
        X_train.append(train_data[i-TIME_STEP:i, 0])
        y_train.append(train_data[i, 0])
    
    if not X_train:
        return None, None

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(TIME_STEP, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=25, verbose=0)
    
    return model, scaler

def add_technical_indicators(df, moving_averages):
    """Calculates and adds technical indicators to the dataframe."""
    for ma in moving_averages:
        period = ma['period']
        ma_type = ma['type']
        column_name = f"{ma_type}{period}"
        if ma_type == 'SMA':
            df[column_name] = df['Close'].rolling(window=period).mean()
        elif ma_type == 'EMA':
            df[column_name] = df['Close'].ewm(span=period, adjust=False).mean()

    # RSI
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['StdDev'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA20'] + (df['StdDev'] * 2)
    df['BB_Lower'] = df['SMA20'] - (df['StdDev'] * 2)

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

# --- UI Helper Functions ---
def render_metric_card(title, value):
    """Renders a styled metric card."""
    st.markdown(f'<div class="metric-card"><h4>{title}</h4><p>{value}</p></div>', unsafe_allow_html=True)

def render_mover_card(ticker, change, is_gainer):
    """Renders a market mover card."""
    symbol = "▲" if is_gainer else "▼"
    css_class = "gainer" if is_gainer else "loser"
    clean_ticker = ticker.replace(".NS", "")
    st.markdown(f'<div class="mover-card"><p class="mover-symbol">{clean_ticker}</p><p class="{css_class}">{symbol} {change:.2f}%</p></div>', unsafe_allow_html=True)
    
def create_prediction_gauge(df_forecast, latest_price):
    """Creates a Plotly gauge chart based on the AI's forecast."""
    if df_forecast.empty or latest_price is None:
        return None

    final_predicted_price = df_forecast['Forecast'].iloc[-1]
    pct_change = ((final_predicted_price - latest_price) / latest_price) * 100

    rating, value = "Hold", 3
    if pct_change > 5: rating, value = "Strong Buy", 5
    elif pct_change > 1: rating, value = "Buy", 4
    elif pct_change < -5: rating, value = "Strong Sell", 1
    elif pct_change < -1: rating, value = "Sell", 2

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{rating}</b>", 'font': {'size': 24, 'color': '#FAFAFA'}},
        gauge={
            'axis': {'range': [0, 6], 'tickwidth': 0, 'showticklabels': False},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "#1B2133",
            'borderwidth': 2,
            'bordercolor': "#2C344B",
            'steps': [
                {'range': [0, 2], 'color': '#d9534f'},
                {'range': [2, 4], 'color': '#f0ad4e'},
                {'range': [4, 6], 'color': '#5cb85c'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.8,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, margin={'t':50, 'b':0, 'l':0, 'r':0}, paper_bgcolor="#1B2133")
    return fig

# --- Sidebar UI ---
with st.sidebar:
    st.title("QuantEdge AI")

    user_ticker_input = st.text_input(
        "Search for a Stock:", "",
        placeholder="e.g., AAPL or RELIANCE",
        help="Enter any global ticker (e.g., AAPL) or Indian stock symbol (e.g., RELIANCE)."
    ).upper()

    if user_ticker_input and user_ticker_input != st.session_state.current_ticker:
        st.session_state.current_ticker = user_ticker_input
        # Reset data when ticker changes
        st.session_state.forecast_data = None
        st.session_state.evaluation_data = None

    prediction_days = st.slider("Forecast Horizon (Days):", 1, 30, 15)
    
    st.subheader("Technical Indicators")
    with st.expander("Add Moving Average"):
        ma_type = st.selectbox("Type", ["SMA", "EMA"], key="ma_type")
        ma_period = st.selectbox("Period", [20, 50, 100, 150, 200], key="ma_period")
        if st.button("Add MA"):
            st.session_state.ma_id_counter += 1
            st.session_state.moving_averages.append({
                "id": st.session_state.ma_id_counter,
                "type": ma_type, "period": ma_period
            })
            st.rerun()

    if st.session_state.moving_averages:
        st.write("Active Moving Averages:")
        for ma in st.session_state.moving_averages:
            cols = st.columns([0.8, 0.2])
            cols[0].info(f"{ma['period']}-Day {ma['type']}")
            if cols[1].button("X", key=f"remove_ma_{ma['id']}"):
                st.session_state.moving_averages = [m for m in st.session_state.moving_averages if m['id'] != ma['id']]
                st.rerun()
    
    st.write("Other Indicators")
    show_bbands = st.checkbox("Show Bollinger Bands", value=False)
    show_volume = st.checkbox("Show Volume", value=False)
    show_rsi = st.checkbox("Show RSI", value=False)
    show_macd = st.checkbox("Show MACD", value=False)
    
    st.markdown("---")
    market_choice = st.selectbox("Market Movers:", ["S&P 500", "NASDAQ 100", "NIFTY 50"])
    st.markdown("---")
    st.info("Disclaimer: All data and predictions are for informational purposes only and should not be considered financial advice.")

# --- Main App Logic ---
if not user_ticker_input:
    st.title("Welcome to the QuantEdge AI Dashboard")
    st.info("📈 Enter a stock ticker in the sidebar to load its financial data and AI forecast.")
else:
    data, info, backend_ticker = load_data(user_ticker_input)

    if data is None:
        st.error(f"Failed to load data for '{user_ticker_input}'. Please check the symbol or try another.")
    else:
        data = add_technical_indicators(data, st.session_state.moving_averages)

        logo_url = info.get('logo_url', '')
        website_url = info.get('website', '')
        if not logo_url and website_url:
            logo_url = f"https://www.google.com/s2/favicons?sz=128&domain_url={website_url}"
        
        company_name = info.get('longName', user_ticker_input)
        
        st.markdown(f"""
            <div class="company-header">
                <img class="company-logo" src="{logo_url}">
                <div class="company-name">
                    <h2>{company_name}</h2>
                    <p>{backend_ticker}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 Forecast & Analysis", "🏢 Company Profile", "🤖 Model Performance"])
        
        with tab1:
            # --- Key Metrics ---
            st.subheader("Key Metrics")
            latest_price = get_latest_price(backend_ticker) or data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2]
            price_change = latest_price - prev_close
            pct_change = (price_change / prev_close) * 100
            
            currency_symbol = "₹" if backend_ticker.endswith(".NS") else "$"
            
            cols = st.columns(4)
            with cols[0]:
                render_metric_card("Current Price", f"{currency_symbol}{latest_price:,.2f}")
            with cols[1]:
                render_metric_card("Daily Change", f'{"▲" if price_change >= 0 else "▼"} {price_change:,.2f} ({pct_change:.2f}%)')
            with cols[2]:
                 render_metric_card("Market Cap", f"${info.get('marketCap', 0)/1e9:,.2f}B")
            with cols[3]:
                 render_metric_card("Volume", f"{data['Volume'].iloc[-1]:,}")

            # --- Interactive Price Chart ---
            st.subheader("Interactive Price Chart")
            num_subplots = 1 + sum([show_volume, show_rsi, show_macd])
            row_heights = [0.6] + [0.4 / (num_subplots - 1)] * (num_subplots - 1) if num_subplots > 1 else [1.0]
            
            fig_price = make_subplots(rows=num_subplots, cols=1, shared_xaxes=True, 
                                      vertical_spacing=0.03, row_heights=row_heights)
            
            fig_price.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
            
            for ma in st.session_state.moving_averages:
                col_name = f"{ma['type']}{ma['period']}"
                fig_price.add_trace(go.Scatter(x=data['Date'], y=data[col_name], name=f"{ma['period']}-Day {ma['type']}"), row=1, col=1)

            if show_bbands:
                fig_price.add_trace(go.Scatter(x=data['Date'], y=data['BB_Upper'], name='Upper Band', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
                fig_price.add_trace(go.Scatter(x=data['Date'], y=data['BB_Lower'], name='Lower Band', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.2)'), row=1, col=1)

            subplot_row = 2
            if show_volume:
                fig_price.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume', marker_color='royalblue'), row=subplot_row, col=1)
                fig_price.update_yaxes(title_text="Volume", row=subplot_row, col=1)
                subplot_row += 1
            if show_rsi:
                fig_price.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name='RSI'), row=subplot_row, col=1)
                fig_price.add_hline(y=70, line_dash="dash", line_color="red", row=subplot_row, col=1)
                fig_price.add_hline(y=30, line_dash="dash", line_color="green", row=subplot_row, col=1)
                fig_price.update_yaxes(title_text="RSI", row=subplot_row, col=1)
                subplot_row += 1
            if show_macd:
                fig_price.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], name='MACD', line=dict(color='blue')), row=subplot_row, col=1)
                fig_price.add_trace(go.Scatter(x=data['Date'], y=data['MACD_Signal'], name='Signal Line', line=dict(color='red')), row=subplot_row, col=1)
                fig_price.update_yaxes(title_text="MACD", row=subplot_row, col=1)
            
            fig_price.update_layout(height=800, xaxis_rangeslider_visible=False, showlegend=True, yaxis_title=f'Price ({currency_symbol})')
            fig_price.update_xaxes(showticklabels=False, row=1, col=1)
            st.plotly_chart(fig_price, use_container_width=True)

            # --- AI Forecast Section ---
            if st.button("Run AI Forecast", key="forecast_button"):
                with st.spinner("Generating AI forecast... This may take a moment."):
                    model, scaler = get_trained_model(data)
                    if model is None:
                        st.warning("Could not train model (not enough data).")
                        st.session_state.forecast_data = None
                    else:
                        close_data_full = data.filter(['Close']).values
                        scaled_data_full = scaler.transform(close_data_full)
                        last_sequence = scaled_data_full[-TIME_STEP:]
                        future_predictions = []
                        current_batch = last_sequence.reshape(1, TIME_STEP, 1)
                        for _ in range(prediction_days):
                            next_pred = model.predict(current_batch, verbose=0)[0][0]
                            future_predictions.append(next_pred)
                            current_batch = np.append(current_batch[:, 1:, :], [[[next_pred]]], axis=1)
                        
                        forecasted_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                        last_date = data['Date'].iloc[-1]
                        future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
                        df_forecast = pd.DataFrame({'Date': future_dates, 'Forecast': forecasted_prices.flatten()})

                        st.session_state.forecast_data = {
                            'df_forecast': df_forecast,
                            'latest_price': latest_price,
                            'currency_symbol': currency_symbol
                        }
            
            if st.session_state.forecast_data:
                forecast_info = st.session_state.forecast_data
                df_forecast = forecast_info['df_forecast']
                
                st.subheader("AI Forecast Summary")
                col1, col2 = st.columns([2, 1])
                with col1:
                    sub_cols = st.columns(3)
                    with sub_cols[0]:
                        trend = "Upward 📈" if df_forecast['Forecast'].iloc[-1] > forecast_info['latest_price'] else "Downward 📉"
                        render_metric_card("Predicted Trend", trend)
                    with sub_cols[1]:
                        render_metric_card("Forecast High", f"{forecast_info['currency_symbol']}{df_forecast['Forecast'].max():,.2f}")
                    with sub_cols[2]:
                        render_metric_card("Forecast Low", f"{forecast_info['currency_symbol']}{df_forecast['Forecast'].min():,.2f}")

                with col2:
                    st.subheader("AI Rating")
                    gauge_fig = create_prediction_gauge(df_forecast, forecast_info['latest_price'])
                    if gauge_fig:
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    else:
                        st.info("AI rating could not be generated.")
                
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Historical Price', line=dict(color='#007bff', width=2)))
                fig_forecast.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast['Forecast'], name='Forecasted Price', line=dict(color='#FFA500', width=2, dash='dash')))
                fig_forecast.update_layout(title="Historical vs. Forecasted Stock Price", yaxis_title=f"Price ({forecast_info['currency_symbol']})", xaxis_rangeslider_visible=True)
                st.plotly_chart(fig_forecast, use_container_width=True)
            
            # --- Market Movers ---
            st.markdown("---")
            st.subheader(f"Today's Market Movers ({market_choice})")

            market_map = {
                "S&P 500": get_sp500_tickers,
                "NASDAQ 100": get_nasdaq100_tickers,
                "NIFTY 50": get_nifty50_tickers
            }
            tickers_to_fetch = market_map[market_choice]()
            gainers, losers = get_market_movers(tickers_to_fetch)

            if not gainers.empty and not losers.empty:
                st.markdown("<h5>Top 5 Gainers</h5>", unsafe_allow_html=True)
                g_cols = st.columns(5)
                for i, (ticker, change) in enumerate(gainers.items()):
                    with g_cols[i]:
                        render_mover_card(ticker, change, is_gainer=True)
                
                st.markdown("<h5 style='margin-top: 25px;'>Top 5 Losers</h5>", unsafe_allow_html=True)
                l_cols = st.columns(5)
                for i, (ticker, change) in enumerate(losers.items()):
                    with l_cols[i]:
                        render_mover_card(ticker, change, is_gainer=False)
            else:
                st.info("Market data is currently being updated. Please check back shortly.")

        with tab2:
            st.subheader(f"About {company_name}")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                st.markdown(f"**Website:** [{info.get('website', 'N/A')}]({info.get('website', '#')})")
            with col2:
                st.info(f"**Business Summary:** {info.get('longBusinessSummary', 'No summary available.')}")
            
            st.markdown("<hr style='border-color: #2C344B;'>", unsafe_allow_html=True)
            st.subheader("Key Financials")

            def format_metric(value, prefix="", suffix="", is_percent=False):
                if value is None or not isinstance(value, (int, float)): return "N/A"
                if is_percent: return f"{value:.2%}"
                if abs(value) > 1e12: return f"{prefix}{value/1e12:.2f}T{suffix}"
                if abs(value) > 1e9: return f"{prefix}{value/1e9:.2f}B{suffix}"
                if abs(value) > 1e6: return f"{prefix}{value/1e6:.2f}M{suffix}"
                return f"{prefix}{value:,.2f}{suffix}"

            m_cols = st.columns(4)
            with m_cols[0]:
                st.metric("Market Cap", format_metric(info.get('marketCap'), prefix="$"))
                st.metric("Enterprise Value", format_metric(info.get('enterpriseValue'), prefix="$"))
            with m_cols[1]:
                st.metric("Trailing P/E", format_metric(info.get('trailingPE')))
                st.metric("Forward P/E", format_metric(info.get('forwardPE')))
            with m_cols[2]:
                st.metric("Dividend Yield", format_metric(info.get('dividendYield'), is_percent=True))
                st.metric("Trailing EPS", format_metric(info.get('trailingEps'), prefix="$"))
            with m_cols[3]:
                st.metric("Beta", format_metric(info.get('beta')))
                st.metric("52 Week High/Low", f"{format_metric(info.get('fiftyTwoWeekHigh'))} / {format_metric(info.get('fiftyTwoWeekLow'))}")

        with tab3:
            st.subheader("Understanding the Model's Performance")
            st.markdown("To ensure the model is reliable, we test its ability to predict on historical data it has never seen before (the last 20% of the dataset).")
            
            if st.button("Evaluate Model Performance", key="eval_button"):
                with st.spinner("Training and evaluating the model..."):
                    model, scaler = get_trained_model(data)
                    if model is None:
                        st.warning("Could not evaluate model (not enough data).")
                    else:
                        close_data = data.filter(['Close']).values
                        scaled_data = scaler.transform(close_data)
                        train_size = int(len(scaled_data) * 0.8)
                        test_data = scaled_data[train_size - TIME_STEP:]
                        X_test, y_test = [], []
                        for i in range(TIME_STEP, len(test_data)):
                            X_test.append(test_data[i-TIME_STEP:i, 0])
                            y_test.append(test_data[i, 0])
                        
                        if not X_test:
                            st.warning("Not enough data for evaluation.")
                        else:
                            X_test, y_test = np.array(X_test), np.array(y_test)
                            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                            
                            test_predictions = scaler.inverse_transform(model.predict(X_test, verbose=0))
                            actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
                            
                            st.session_state.evaluation_data = {
                                'rmse': np.sqrt(mean_squared_error(actual_prices, test_predictions)),
                                'r2': r2_score(actual_prices, test_predictions),
                                'actual_prices': actual_prices,
                                'test_predictions': test_predictions,
                                'currency_symbol': currency_symbol
                            }
            
            if st.session_state.evaluation_data:
                eval_info = st.session_state.evaluation_data
                st.markdown("#### Evaluation Metrics")
                cols = st.columns(2)
                cols[0].metric("Root Mean Squared Error (RMSE)", f"{eval_info['currency_symbol']}{eval_info['rmse']:.2f}", help="Lower is better. Average prediction error in currency.")
                cols[1].metric("R-squared (R²) Score", f"{eval_info['r2']:.2%}", help="Higher is better (max 100%). How well predictions fit actual data.")

                fig_eval = go.Figure()
                fig_eval.add_trace(go.Scatter(y=eval_info['actual_prices'].flatten(), name='Actual Price', line=dict(color='red')))
                fig_eval.add_trace(go.Scatter(y=eval_info['test_predictions'].flatten(), name='Predicted Price', line=dict(color='blue')))
                fig_eval.update_layout(title="Model Evaluation: Actual vs. Predicted Prices (Test Data)", yaxis_title=f"Price ({eval_info['currency_symbol']})")
                st.plotly_chart(fig_eval, use_container_width=True)

# --- Footer ---
st.markdown("""
<div class="footer">
    <p>QuantEdge AI | Advanced Stock Analysis & Forecasting</p>
    <p>All data is provided for informational purposes only, and does not constitute investment advice. 
    Past performance is not indicative of future results. Please consult with a qualified financial advisor before making any investment decisions.</p>
</div>
""", unsafe_allow_html=True)