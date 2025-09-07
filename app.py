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
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="QuantEdge | AI Forecast",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Optimized CSS ---
st.markdown("""
<style>
    /* Optimized CSS with better performance */
    :root {
        --bg-primary: #0E1117;
        --bg-secondary: #1B2133;
        --border-color: #2C344B;
        --text-primary: #FAFAFA;
        --text-secondary: #888;
        --accent-blue: #4A90E2;
        --accent-green: #28a745;
        --accent-red: #dc3545;
    }
    
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 100%;
    }
    
    /* Universal card styling */
    .card {
        background-color: var(--bg-secondary);
        border-radius: 10px;
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        color: var(--text-primary);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 163, 255, 0.15);
    }
    
    /* Metric cards */
    .metric-card {
        text-align: center;
        height: 100%;
    }
    
    .metric-card h4 {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Market mover cards */
    .mover-card {
        cursor: pointer;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, var(--bg-secondary) 0%, #252B42 100%);
    }
    
    .mover-card:hover {
        background: linear-gradient(135deg, #252B42 0%, var(--bg-secondary) 100%);
    }
    
    .mover-symbol {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .gainer { color: var(--accent-green); }
    .loser { color: var(--accent-red); }
    
    /* Company header */
    .company-header {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .company-logo {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        object-fit: contain;
        background: white;
        padding: 2px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, var(--accent-blue) 0%, #5BA0F2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(74, 144, 226, 0.3);
    }
    
    /* Tabs optimization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        border-bottom: 2px solid var(--border-color);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-secondary);
        border-bottom: 3px solid var(--accent-blue);
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
START_DATE = "2015-01-01"
TODAY = date.today()
TIME_STEP = 60

# --- Session State Management ---
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'forecast_data': None,
        'evaluation_data': None,
        'moving_averages': [],
        'ma_id_counter': 0,
        'current_ticker': "",
        'selected_from_movers': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Optimized Data Fetching ---
@st.cache_data(ttl=86400)
def get_market_tickers(market_name):
    """Unified function to get market tickers with fallback options"""
    tickers = {

    "S&P 500": [

    'A', 'AAL', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI',

    'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG',

    'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME',

    'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD',

    'APH', 'APO', 'APP', 'APTV', 'ARE', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON',

    'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN',

    'BF.B', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BLDR', 'BMY',

    'BR', 'BRK.B', 'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH',

    'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG',

    'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA',

    'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COIN', 'COO',

    'COP', 'COST', 'CPAY', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSCO',

    'CSGP', 'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CVS', 'CVX', 'CZR', 'D',

    'DAL', 'DASH', 'DD', 'DDOG', 'DE', 'DECK', 'DFS', 'DG', 'DGX', 'DHI',

    'DHR', 'DIS', 'DLR', 'DLTR', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK',

    'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL',

    'ELV', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ERJ',

    'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVA', 'EW', 'EWBC', 'EXC', 'EXPD',

    'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV',

    'FI', 'FICO', 'FIS', 'FITB', 'FLT', 'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR',

    'FTNT', 'FTV', 'GD', 'GE', 'GEHC', 'GEV', 'GEN', 'GILD', 'GIS', 'GL',

    'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW',

    'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX',

    'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HWM', 'IBM',

    'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP',

    'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBL',

    'JCI', 'JCP', 'JD', 'JEF', 'JNJ', 'JPM', 'K', 'KDP', 'KEX', 'KEY',

    'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KSU',

    'L', 'LAD', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT',

    'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA',

    'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET',

    'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO',

    'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI',

    'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NI',

    'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR',

    'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OGN', 'OKE', 'OMC', 'ON', 'ORCL',

    'ORLY', 'OXY', 'PANW', 'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PDD', 'PEG',

    'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR',

    'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA',

    'PSX', 'PTC', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'REG', 'REGN',

    'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG',

    'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SCL', 'SCRM', 'SEE', 'SHW', 'SJM',

    'SLB', 'SLG', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STT',

    'STX', 'STZ', 'SUI', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP',

    'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO',

    'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT',

    'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UNH',

    'UNP', 'UPS', 'URI', 'USB', 'V', 'VEEV', 'VFC', 'VICI', 'VLO', 'VMC',

    'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD',

    'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WRK',

    'WST', 'WY', 'WYNN', 'X', 'XEL', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH',

    'ZBRA', 'ZION', 'ZTS'

],

    "NASDAQ 100": [

    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'AVGO', 'TSLA', 'COST',

    'AMD', 'PEP', 'ADBE', 'CSCO', 'NFLX', 'INTC', 'TMUS', 'CMCSA', 'QCOM', 'INTU',

    'TXN', 'AMGN', 'HON', 'AMAT', 'BKNG', 'SBUX', 'ISRG', 'GILD', 'LRCX', 'ADI',

    'ADP', 'VRTX', 'REGN', 'MDLZ', 'PYPL', 'PANW', 'SNPS', 'CDNS', 'ASML', 'MAR',

    'KLAC', 'MU', 'MNST', 'CSX', 'MELI', 'CTAS', 'ORLY', 'AEP', 'FTNT', 'CRWD',

    'PCAR', 'DXCM', 'KDP', 'MRNA', 'ABNB', 'KHC', 'PAYX', 'IDXX', 'WDAY', 'LULU',

    'CEG', 'EXC', 'ROST', 'ODFL', 'BIIB', 'GEHC', 'DDOG', 'CPRT', 'FAST', 'GFS',

    'BKR', 'CTSH', 'VRSK', 'ON', 'CSGP', 'SGEN', 'XEL', 'MCHP', 'AZN', 'SIRI',

    'ENPH', 'EA', 'ILMN', 'ADSK', 'WBD', 'PDD', 'JD', 'ZM', 'TEAM', 'MRVL',

    'WBA', 'ALGN', 'DLTR', 'FANG', 'EBAY', 'ZM', 'ANSS', 'ZS', 'CHTR', 'PCAR'

],

    "NIFTY 50": [

    'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',

    'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',

    'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',

    'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',

    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS',

    'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LTIM.NS',

    'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS',

    'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SUNPHARMA.NS',

    'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS',

    'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'

]

}
    
    try:
        # Try to fetch dynamically if you have the libraries
        if market_name == "NIFTY 50":
            url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
            df = pd.read_csv(url)
            return (df['Symbol'] + ".NS").tolist()
    except:
        pass
    
    return tickers.get(market_name, [])

@st.cache_data(ttl=3600)
def load_data(ticker_input: str):
    """Optimized data loading with better error handling"""
    try:
        # Handle ticker variations
        tickers_to_try = [ticker_input]
        if not ticker_input.endswith(".NS") and len(ticker_input) <= 10:
            tickers_to_try.append(f"{ticker_input}.NS")
        
        for ticker in tickers_to_try:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=START_DATE, end=TODAY)
                
                if not df.empty:
                    df.reset_index(inplace=True)
                    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                    
                    # Get info with timeout
                    info = stock.info if hasattr(stock, 'info') else {}
                    
                    return df, info, ticker
            except:
                continue
                
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data(ttl=300)
def get_market_movers(tickers):
    """Optimized market movers fetching"""
    if not tickers:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    
    try:
        # Fetch in smaller batches for better performance
        batch_size = 10
        all_changes = {}
        
        for i in range(0, len(tickers[:20]), batch_size):  # Limit to 20 tickers
            batch = tickers[i:i+batch_size]
            data = yf.download(batch, period="2d", progress=False, threads=True)['Close']
            
            if not data.empty and len(data) >= 2:
                if isinstance(data, pd.DataFrame):
                    for col in data.columns:
                        pct_change = (data[col].iloc[-1] - data[col].iloc[-2]) / data[col].iloc[-2] * 100
                        all_changes[col] = pct_change
                else:
                    ticker = batch[0]
                    pct_change = (data.iloc[-1] - data.iloc[-2]) / data.iloc[-2] * 100
                    all_changes[ticker] = pct_change
        
        changes_series = pd.Series(all_changes)
        gainers = changes_series.nlargest(5)
        losers = changes_series.nsmallest(5)
        
        return gainers, losers
    except:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64')

# --- Optimized Model Training ---
@st.cache_resource
def get_trained_model(_df):  # Note the underscore to prevent hashing issues
    """Optimized LSTM model with early stopping"""
    try:
        close_data = _df[['Close']].values
        
        if len(close_data) < TIME_STEP * 2:
            return None, None
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data)
        
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        
        X_train, y_train = [], []
        for i in range(TIME_STEP, len(train_data)):
            X_train.append(train_data[i-TIME_STEP:i, 0])
            y_train.append(train_data[i, 0])
        
        if len(X_train) < 10:
            return None, None
        
        X_train = np.array(X_train).reshape(-1, TIME_STEP, 1)
        y_train = np.array(y_train)
        
        # Optimized model architecture
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(TIME_STEP, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        model.fit(X_train, y_train, 
                 batch_size=32, 
                 epochs=50, 
                 verbose=0,
                 callbacks=[early_stop],
                 validation_split=0.1)
        
        return model, scaler
    except Exception as e:
        st.error(f"Model training error: {e}")
        return None, None

def add_technical_indicators(df, moving_averages):
    """Optimized technical indicators calculation"""
    df = df.copy()
    
    # Moving averages
    for ma in moving_averages:
        col_name = f"{ma['type']}{ma['period']}"
        if ma['type'] == 'SMA':
            df[col_name] = df['Close'].rolling(window=ma['period'], min_periods=1).mean()
        elif ma['type'] == 'EMA':
            df[col_name] = df['Close'].ewm(span=ma['period'], adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma20 + (std20 * 2)
    df['BB_Lower'] = sma20 - (std20 * 2)
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# --- UI Components ---
def render_metric_card(title, value):
    """Render metric card"""
    st.markdown(f'''
        <div class="card metric-card">
            <h4>{title}</h4>
            <p>{value}</p>
        </div>
    ''', unsafe_allow_html=True)

def render_mover_card(ticker, change, is_gainer):
    """Render clickable market mover card"""
    symbol = "▲" if is_gainer else "▼"
    css_class = "gainer" if is_gainer else "loser"
    clean_ticker = ticker.replace(".NS", "")
    
    # Create unique key for button
    button_key = f"mover_{clean_ticker}_{change:.2f}".replace(".", "_")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'''
            <div class="card mover-card">
                <div class="mover-symbol">{clean_ticker}</div>
                <div class="{css_class}">{symbol} {change:.2f}%</div>
            </div>
        ''', unsafe_allow_html=True)
    with col2:
        if st.button("📊", key=button_key, help=f"Load {clean_ticker}"):
            st.session_state.current_ticker = clean_ticker
            st.session_state.selected_from_movers = True
            st.rerun()

def create_prediction_gauge(df_forecast, latest_price):
    """Create prediction gauge chart"""
    if df_forecast.empty or latest_price is None:
        return None
    
    final_price = df_forecast['Forecast'].iloc[-1]
    pct_change = ((final_price - latest_price) / latest_price) * 100
    
    # Determine rating
    if pct_change > 5:
        rating, value = "Strong Buy", 5
    elif pct_change > 1:
        rating, value = "Buy", 4
    elif pct_change < -5:
        rating, value = "Strong Sell", 1
    elif pct_change < -1:
        rating, value = "Sell", 2
    else:
        rating, value = "Hold", 3
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={'reference': 3, 'relative': False},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{rating}</b>", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 6], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 2], 'color': '#ff4444'},
                {'range': [2, 4], 'color': '#ffaa00'},
                {'range': [4, 6], 'color': '#00ff00'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin={'t': 50, 'b': 0, 'l': 0, 'r': 0},
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    
    return fig

# --- Main Sidebar ---
with st.sidebar:
    st.title("🤖 QuantEdge AI")
    st.markdown("---")
    
    # Check if ticker was selected from movers
    if st.session_state.selected_from_movers:
        default_value = st.session_state.current_ticker
        st.session_state.selected_from_movers = False
    else:
        default_value = st.session_state.current_ticker
    
    # Search input
    user_ticker_input = st.text_input(
        "🔍 Search Stock",
        value=default_value,
        placeholder="e.g., AAPL, RELIANCE",
        help="Enter any stock symbol"
    ).upper()
    
    if user_ticker_input != st.session_state.current_ticker:
        st.session_state.current_ticker = user_ticker_input
        st.session_state.forecast_data = None
        st.session_state.evaluation_data = None
    
    st.markdown("---")
    
    # Prediction settings
    st.subheader("⚙️ Forecast Settings")
    prediction_days = st.slider("Forecast Days", 1, 30, 15)
    
    # Technical indicators
    st.markdown("---")
    st.subheader("📊 Technical Indicators")
    
    with st.expander("Moving Averages"):
        col1, col2 = st.columns(2)
        with col1:
            ma_type = st.selectbox("Type", ["SMA", "EMA"])
        with col2:
            ma_period = st.selectbox("Period", [20, 50, 100, 200])
        
        if st.button("Add MA", use_container_width=True):
            st.session_state.ma_id_counter += 1
            st.session_state.moving_averages.append({
                "id": st.session_state.ma_id_counter,
                "type": ma_type,
                "period": ma_period
            })
            st.rerun()
    
    # Display active MAs
    if st.session_state.moving_averages:
        for ma in st.session_state.moving_averages:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"{ma['period']}-Day {ma['type']}")
            with col2:
                if st.button("❌", key=f"del_ma_{ma['id']}"):
                    st.session_state.moving_averages = [
                        m for m in st.session_state.moving_averages 
                        if m['id'] != ma['id']
                    ]
                    st.rerun()
    
    # Other indicators
    show_bbands = st.checkbox("Bollinger Bands")
    show_volume = st.checkbox("Volume")
    show_rsi = st.checkbox("RSI")
    show_macd = st.checkbox("MACD")
    
    # Market movers
    st.markdown("---")
    st.subheader("🌍 Market Movers")
    market_choice = st.selectbox(
        "Select Market",
        ["S&P 500", "NASDAQ 100", "NIFTY 50"]
    )
    
    # Footer
    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not financial advice.")

# --- Main Content Area ---
if not st.session_state.current_ticker:
    # Welcome screen
    st.title("Welcome to QuantEdge AI 🚀")
    st.markdown("### AI-Powered Stock Analysis & Forecasting")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📈 **Real-time Data**\nLive market prices and indicators")
    with col2:
        st.info("🤖 **AI Predictions**\nLSTM neural network forecasts")
    with col3:
        st.info("📊 **Technical Analysis**\nComprehensive indicators")
    
    st.markdown("---")
    st.markdown("### 👈 Enter a stock symbol in the sidebar to begin")
    
else:
    # Load data
    data, info, backend_ticker = load_data(st.session_state.current_ticker)
    
    if data is None:
        st.error(f"❌ Unable to load data for '{st.session_state.current_ticker}'")
        st.info("Please check the symbol and try again")
    else:
        # Add indicators
        data = add_technical_indicators(data, st.session_state.moving_averages)
        
        # Company header
        company_name = info.get('longName', st.session_state.current_ticker)
        logo_url = info.get('logo_url', '')
        if not logo_url:
            website = info.get('website', '')
            if website:
                logo_url = f"https://www.google.com/s2/favicons?sz=64&domain={website}"
        
        st.markdown(f'''
            <div class="company-header">
                <img class="company-logo" src="{logo_url}" onerror="this.style.display='none'">
                <div>
                    <h2 style="margin:0">{company_name}</h2>
                    <p style="margin:0;color:#888">{backend_ticker}</p>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Analysis", 
            "🤖 AI Forecast", 
            "🏢 Company Info", 
            "📈 Model Performance"
        ])
        
        with tab1:
            # Key metrics
            st.subheader("Key Metrics")
            
            latest_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else latest_price
            price_change = latest_price - prev_close
            pct_change = (price_change / prev_close * 100) if prev_close != 0 else 0
            
            currency = "₹" if backend_ticker.endswith(".NS") else "$"
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Current Price",
                    f"{currency}{latest_price:,.2f}",
                    f"{price_change:+.2f} ({pct_change:+.2f}%)"
                )
            with col2:
                volume = data['Volume'].iloc[-1]
                st.metric("Volume", f"{volume:,.0f}")
            with col3:
                high_52w = data['High'].tail(252).max() if len(data) > 252 else data['High'].max()
                st.metric("52W High", f"{currency}{high_52w:,.2f}")
            with col4:
                low_52w = data['Low'].tail(252).min() if len(data) > 252 else data['Low'].min()
                st.metric("52W Low", f"{currency}{low_52w:,.2f}")
            
            # Interactive chart
            st.subheader("Price Chart")
            
            # Calculate number of subplots
            num_subplots = 1 + sum([show_volume, show_rsi, show_macd])
            heights = [0.6] + [0.4/(num_subplots-1)]*(num_subplots-1) if num_subplots > 1 else [1]
            
            fig = make_subplots(
                rows=num_subplots, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=heights,
                subplot_titles=(['Price'] + 
                               (['Volume'] if show_volume else []) +
                               (['RSI'] if show_rsi else []) +
                               (['MACD'] if show_macd else []))
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data['Date'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Moving averages
            for ma in st.session_state.moving_averages:
                col_name = f"{ma['type']}{ma['period']}"
                if col_name in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data['Date'],
                            y=data[col_name],
                            name=f"{ma['period']}-Day {ma['type']}",
                            line=dict(width=1)
                        ),
                        row=1, col=1
                    )
            
            # Bollinger Bands
            if show_bbands and 'BB_Upper' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data['BB_Upper'],
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data['BB_Lower'],
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)'
                    ),
                    row=1, col=1
                )
            
            current_row = 2
            
            # Volume
            if show_volume:
                colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                         for i in range(len(data))]
                fig.add_trace(
                    go.Bar(
                        x=data['Date'],
                        y=data['Volume'],
                        name='Volume',
                        marker_color=colors
                    ),
                    row=current_row, col=1
                )
                current_row += 1
            
            # RSI
            if show_rsi and 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data['RSI'],
                        name='RSI',
                        line=dict(color='purple')
                    ),
                    row=current_row, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
                current_row += 1
            
            # MACD
            if show_macd and 'MACD' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data['MACD'],
                        name='MACD',
                        line=dict(color='blue')
                    ),
                    row=current_row, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data['MACD_Signal'],
                        name='Signal',
                        line=dict(color='red')
                    ),
                    row=current_row, col=1
                )
            
            # Update layout
            fig.update_layout(
                height=700,
                xaxis_rangeslider_visible=False,
                showlegend=True,
                hovermode='x unified',
                template='plotly_dark'
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2C344B')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2C344B')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Market Movers Section
            st.markdown("---")
            st.subheader(f"📊 {market_choice} Market Movers")
            
            with st.spinner("Loading market data..."):
                market_tickers = get_market_tickers(market_choice)
                gainers, losers = get_market_movers(market_tickers)
            
            if not gainers.empty and not losers.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🚀 Top Gainers")
                    for ticker, change in gainers.items():
                        render_mover_card(ticker, change, is_gainer=True)
                
                with col2:
                    st.markdown("### 📉 Top Losers")
                    for ticker, change in losers.items():
                        render_mover_card(ticker, change, is_gainer=False)
            else:
                st.info("Market data is updating. Please refresh in a moment.")
        
        with tab2:
            st.subheader("🤖 AI-Powered Price Forecast")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"📊 Forecast period: **{prediction_days} days** | Model: **LSTM Neural Network**")
            with col2:
                if st.button("🚀 Generate Forecast", use_container_width=True):
                    with st.spinner("Training AI model and generating predictions..."):
                        model, scaler = get_trained_model(data)
                        
                        if model is None:
                            st.error("Insufficient data for model training")
                        else:
                            # Prepare data for prediction
                            close_data = data[['Close']].values
                            scaled_data = scaler.transform(close_data)
                            
                            # Generate predictions
                            last_sequence = scaled_data[-TIME_STEP:]
                            predictions = []
                            current_batch = last_sequence.reshape(1, TIME_STEP, 1)
                            
                            for _ in range(prediction_days):
                                next_pred = model.predict(current_batch, verbose=0)[0][0]
                                predictions.append(next_pred)
                                current_batch = np.append(current_batch[:, 1:, :], [[[next_pred]]], axis=1)
                            
                            # Inverse transform predictions
                            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                            
                            # Create forecast dataframe
                            last_date = data['Date'].iloc[-1]
                            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                                        periods=prediction_days, freq='D')
                            
                            df_forecast = pd.DataFrame({
                                'Date': future_dates,
                                'Forecast': predictions.flatten()
                            })
                            
                            st.session_state.forecast_data = {
                                'df_forecast': df_forecast,
                                'latest_price': latest_price,
                                'currency': currency
                            }
                            st.success("✅ Forecast generated successfully!")
            
            # Display forecast if available
            if st.session_state.forecast_data:
                forecast_info = st.session_state.forecast_data
                df_forecast = forecast_info['df_forecast']
                
                # Forecast metrics
                st.markdown("### Forecast Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    target_price = df_forecast['Forecast'].iloc[-1]
                    st.metric(
                        "Target Price",
                        f"{forecast_info['currency']}{target_price:,.2f}",
                        f"{(target_price/forecast_info['latest_price']-1)*100:+.1f}%"
                    )
                
                with col2:
                    max_price = df_forecast['Forecast'].max()
                    st.metric(
                        "Forecast High",
                        f"{forecast_info['currency']}{max_price:,.2f}"
                    )
                
                with col3:
                    min_price = df_forecast['Forecast'].min()
                    st.metric(
                        "Forecast Low",
                        f"{forecast_info['currency']}{min_price:,.2f}"
                    )
                
                with col4:
                    trend = "📈 Bullish" if target_price > forecast_info['latest_price'] else "📉 Bearish"
                    st.metric("Trend", trend)
                
                # AI Rating Gauge
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Forecast chart
                    fig_forecast = go.Figure()
                    
                    # Historical prices
                    fig_forecast.add_trace(go.Scatter(
                        x=data['Date'],
                        y=data['Close'],
                        name='Historical',
                        line=dict(color='#4A90E2', width=2)
                    ))
                    
                    # Forecast
                    fig_forecast.add_trace(go.Scatter(
                        x=df_forecast['Date'],
                        y=df_forecast['Forecast'],
                        name='AI Forecast',
                        line=dict(color='#FFA500', width=2, dash='dash')
                    ))
                    
                    # Add connection line
                    connection_x = [data['Date'].iloc[-1], df_forecast['Date'].iloc[0]]
                    connection_y = [data['Close'].iloc[-1], df_forecast['Forecast'].iloc[0]]
                    fig_forecast.add_trace(go.Scatter(
                        x=connection_x,
                        y=connection_y,
                        line=dict(color='#FFA500', width=1, dash='dot'),
                        showlegend=False
                    ))
                    
                    fig_forecast.update_layout(
                        title="Price Forecast",
                        yaxis_title=f"Price ({forecast_info['currency']})",
                        height=400,
                        template='plotly_dark',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                
                with col2:
                    st.markdown("### AI Recommendation")
                    gauge_fig = create_prediction_gauge(df_forecast, forecast_info['latest_price'])
                    if gauge_fig:
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Confidence metrics
                    st.markdown("### Confidence Factors")
                    confidence_score = min(95, 70 + len(data) / 100)  # Simple confidence calculation
                    st.progress(confidence_score / 100)
                    st.caption(f"Model Confidence: {confidence_score:.0f}%")
        
        with tab3:
            st.subheader("🏢 Company Information")
            
            if info:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### Overview")
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Country:** {info.get('country', 'N/A')}")
                    st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}" 
                            if info.get('fullTimeEmployees') else "**Employees:** N/A")
                    
                    website = info.get('website', '')
                    if website:
                        st.write(f"**Website:** [{website}]({website})")
                
                with col2:
                    summary = info.get('longBusinessSummary', 'No description available.')
                    st.markdown("### Business Summary")
                    st.write(summary[:500] + "..." if len(summary) > 500 else summary)
                
                st.markdown("---")
                st.markdown("### Financial Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    market_cap = info.get('marketCap', 0)
                    if market_cap:
                        st.metric("Market Cap", f"${market_cap/1e9:.2f}B" if market_cap > 1e9 else f"${market_cap/1e6:.2f}M")
                    pe_ratio = info.get('trailingPE', 0)
                    if pe_ratio:
                        st.metric("P/E Ratio", f"{pe_ratio:.2f}")
                
                with col2:
                    eps = info.get('trailingEps', 0)
                    if eps:
                        st.metric("EPS", f"${eps:.2f}")
                    div_yield = info.get('dividendYield', 0)
                    if div_yield:
                        st.metric("Dividend Yield", f"{div_yield*100:.2f}%")
                
                with col3:
                    beta = info.get('beta', 0)
                    if beta:
                        st.metric("Beta", f"{beta:.2f}")
                    revenue = info.get('totalRevenue', 0)
                    if revenue:
                        st.metric("Revenue", f"${revenue/1e9:.2f}B" if revenue > 1e9 else f"${revenue/1e6:.2f}M")
                
                with col4:
                    profit_margin = info.get('profitMargins', 0)
                    if profit_margin:
                        st.metric("Profit Margin", f"{profit_margin*100:.1f}%")
                    roe = info.get('returnOnEquity', 0)
                    if roe:
                        st.metric("ROE", f"{roe*100:.1f}%")
            else:
                st.info("Company information not available")
        
        with tab4:
            st.subheader("📈 Model Performance Evaluation")
            
            st.markdown("""
            Evaluate the AI model's accuracy by testing it on historical data. 
            The model will be trained on 80% of the data and tested on the remaining 20%.
            """)
            
            if st.button("🧪 Evaluate Model", use_container_width=True):
                with st.spinner("Evaluating model performance..."):
                    model, scaler = get_trained_model(data)
                    
                    if model is None:
                        st.error("Insufficient data for evaluation")
                    else:
                        # Prepare test data
                        close_data = data[['Close']].values
                        scaled_data = scaler.transform(close_data)
                        
                        train_size = int(len(scaled_data) * 0.8)
                        test_data = scaled_data[train_size - TIME_STEP:]
                        
                        X_test, y_test = [], []
                        for i in range(TIME_STEP, len(test_data)):
                            X_test.append(test_data[i-TIME_STEP:i, 0])
                            y_test.append(test_data[i, 0])
                        
                        if len(X_test) > 0:
                            X_test = np.array(X_test).reshape(-1, TIME_STEP, 1)
                            y_test = np.array(y_test)
                            
                            # Make predictions
                            predictions = model.predict(X_test, verbose=0)
                            
                            # Inverse transform
                            predictions = scaler.inverse_transform(predictions)
                            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                            
                            # Calculate metrics
                            rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
                            r2 = r2_score(y_test_actual, predictions)
                            mae = np.mean(np.abs(y_test_actual - predictions))
                            
                            # Calculate directional accuracy
                            if len(predictions) > 1:
                                actual_direction = np.diff(y_test_actual.flatten()) > 0
                                pred_direction = np.diff(predictions.flatten()) > 0
                                dir_accuracy = np.mean(actual_direction == pred_direction) * 100
                            else:
                                dir_accuracy = 0
                            
                            st.session_state.evaluation_data = {
                                'rmse': rmse,
                                'r2': r2,
                                'mae': mae,
                                'dir_accuracy': dir_accuracy,
                                'predictions': predictions,
                                'actual': y_test_actual,
                                'currency': currency
                            }
                            st.success("✅ Evaluation complete!")
            
            # Display evaluation results
            if st.session_state.evaluation_data:
                eval_data = st.session_state.evaluation_data
                
                st.markdown("### Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "RMSE",
                        f"{eval_data['currency']}{eval_data['rmse']:.2f}",
                        help="Root Mean Square Error - Lower is better"
                    )
                
                with col2:
                    st.metric(
                        "R² Score",
                        f"{eval_data['r2']:.3f}",
                        help="Coefficient of determination - Closer to 1 is better"
                    )
                
                with col3:
                    st.metric(
                        "MAE",
                        f"{eval_data['currency']}{eval_data['mae']:.2f}",
                        help="Mean Absolute Error - Lower is better"
                    )
                
                with col4:
                    st.metric(
                        "Direction Accuracy",
                        f"{eval_data['dir_accuracy']:.1f}%",
                        help="Percentage of correct trend predictions"
                    )
                
                # Actual vs Predicted chart
                st.markdown("### Actual vs Predicted Prices")
                
                fig_eval = go.Figure()
                
                fig_eval.add_trace(go.Scatter(
                    y=eval_data['actual'].flatten(),
                    name='Actual',
                    line=dict(color='#4A90E2', width=2)
                ))
                
                fig_eval.add_trace(go.Scatter(
                    y=eval_data['predictions'].flatten(),
                    name='Predicted',
                    line=dict(color='#FFA500', width=2)
                ))
                
                fig_eval.update_layout(
                    title="Model Performance on Test Data",
                    yaxis_title=f"Price ({eval_data['currency']})",
                    xaxis_title="Time Steps",
                    height=400,
                    template='plotly_dark',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_eval, use_container_width=True)
                
                # Error distribution
                st.markdown("### Prediction Error Distribution")
                
                errors = eval_data['predictions'].flatten() - eval_data['actual'].flatten()
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=errors,
                    nbinsx=30,
                    name='Error Distribution',
                    marker_color='#4A90E2'
                ))
                
                fig_hist.update_layout(
                    title="Distribution of Prediction Errors",
                    xaxis_title=f"Error ({eval_data['currency']})",
                    yaxis_title="Frequency",
                    height=300,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)