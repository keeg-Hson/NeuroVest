# indicators_shim.py
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def rsi(df: pd.DataFrame, close="close", length=14, col="rsi"):
    df[col] = RSIIndicator(close=df[close], window=length).rsi(); return df

def sma(df: pd.DataFrame, close="close", length=50, col="sma"):
    df[col] = SMAIndicator(close=df[close], window=length).sma_indicator(); return df

def ema(df: pd.DataFrame, close="close", length=21, col="ema"):
    df[col] = EMAIndicator(close=df[close], window=length).ema_indicator(); return df

def macd(df: pd.DataFrame, close="close", fast=12, slow=26, signal=9,
         macd_col="macd", signal_col="macd_signal", diff_col="macd_diff"):
    m = MACD(close=df[close], window_fast=fast, window_slow=slow, window_sign=signal)
    df[macd_col], df[signal_col], df[diff_col] = m.macd(), m.macd_signal(), m.macd_diff()
    return df

def bollinger(df: pd.DataFrame, close="close", length=20, ndev=2.0,
              high_col="bb_high", low_col="bb_low", mid_col="bb_mid", width_col="bb_width"):
    bb = BollingerBands(close=df[close], window=length, window_dev=ndev)
    df[high_col], df[low_col], df[mid_col], df[width_col] = (
        bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg(), bb.bollinger_wband()
    )
    return df

def stoch(df: pd.DataFrame, high="high", low="low", close="close", k=14, d=3,
          k_col="stoch_k", d_col="stoch_d"):
    st = StochasticOscillator(high=df[high], low=df[low], close=df[close], window=k, smooth_window=d)
    df[k_col], df[d_col] = st.stoch(), st.stoch_signal(); return df

def atr(df: pd.DataFrame, high="high", low="low", close="close", length=14, col="atr"):
    a = AverageTrueRange(high=df[high], low=df[low], close=df[close], window=length)
    df[col] = a.average_true_range(); return df

def adx(df: pd.DataFrame, high="high", low="low", close="close", length=14, col="adx"):
    a = ADXIndicator(high=df[high], low=df[low], close=df[close], window=length)
    df[col] = a.adx(); return df
