# -----------------------------
# Import libraries
# -----------------------------
import numpy as np
import streamlit as st
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import time
import re
import requests

from newsapi import NewsApiClient
from newspaper import Article
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as si

import nltk
from nltk.corpus import stopwords

import yfinance as yf
from pandas_datareader import data as web
import plotly.graph_objects as go
import plotly.express as px

# Ensure stopwords are available
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# -----------------------------
# Helper Functions
# -----------------------------

def clean_text(text, stop):
    """Clean article text for sentiment analysis."""
    REPLACE_NO_SPACE = re.compile(r"(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\n)|(—)|(\")|(\()|(\))|(\[)|(\])")
    text = REPLACE_NO_SPACE.sub("", text)
    text = re.sub(r"[0-9]+", "", text)
    text = " ".join(word for word in text.split() if word not in stop)
    return text

def analyze_articles(articles):
    """Run sentiment analysis on a batch of articles."""
    df = pd.DataFrame(articles)
    if df.empty:
        return pd.DataFrame()

    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df["polarity"] = ""
    df["subjectivity"] = ""
    df["vader_sentiment"] = ""

    stop = stopwords.words("english")
    analyser = si()

    for i in df.index:
        try:
            article = Article(df.loc[i, "url"])
            article.download()
            time.sleep(1)  # polite delay
            article.parse()
            text = clean_text(article.text, stop)
            df.loc[i, "polarity"] = TextBlob(text).sentiment.polarity
            df.loc[i, "subjectivity"] = TextBlob(text).sentiment.subjectivity
            df.loc[i, "vader_sentiment"] = analyser.polarity_scores(text)["compound"]
        except Exception:
            continue

    df["publishedAt"] = df["publishedAt"].dt.tz_localize(None)
    df["Date"] = df["publishedAt"].dt.date
    return df

def build_scores(df):
    """Aggregate sentiment scores into min/max and extreme score."""
    s = df[["Date", "polarity", "subjectivity", "vader_sentiment"]].copy()
    min_s = s.groupby("Date", as_index=False).min().rename(
        columns={"polarity": "polarity_min", "subjectivity": "subjectivity_min", "vader_sentiment": "vader_min"}
    )
    max_s = s.groupby("Date", as_index=False).max().rename(
        columns={"polarity": "polarity_max", "subjectivity": "subjectivity_max", "vader_sentiment": "vader_max"}
    )
    comb = pd.merge(max_s, min_s, on="Date")
    comb["extreme_score"] = comb["vader_max"] + comb["vader_min"]
    return comb[["Date", "extreme_score"]]

def trading_strategy(ticker, scores, start, end):
    """Combine sentiment with SMA crossover strategy."""
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return pd.DataFrame()

    df["2_SMA"] = df["Close"].rolling(window=2).mean()
    df["5_SMA"] = df["Close"].rolling(window=5).mean()
    df = df.dropna().reset_index()
    df["returns"] = df["Close"].pct_change()

    scores["Date"] = pd.to_datetime(scores["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    data = pd.merge(df, scores, on="Date", how="left").fillna(0)

    lt = data["extreme_score"].expanding().mean()
    st_thr = data["extreme_score"].expanding().mean()

    data["signal"] = np.where(data.extreme_score > lt, 1,
                       np.where(data.extreme_score < st_thr, -1, 0))
    data["position"] = data["signal"].shift(1)
    data["strategy_returns"] = data["returns"] * data["position"]

    # Sharpe Ratios
    data["excess_returns"] = data.strategy_returns - (0.04 / 252)
    sharpe = np.sqrt(252) * data.excess_returns.mean() / data.strategy_returns.std()

    data["excess_returns_benchmark"] = data.returns - (0.04 / 252)
    sharpe_bench = np.sqrt(252) * data.excess_returns_benchmark.mean() / data.returns.std()

    return data, sharpe, sharpe_bench

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Trading Strategy and Signal Indicator For Indian and American Stock Market")
st.header("Dynamic strategy report and dashboards, one click away!") 
st.subheader("Created by: Mithesh R, Vidhi K, Kartikay L")

st.write("Don't have a Ticker? You can find it [here](https://in.finance.yahoo.com)")
st.write("Input example: `Indian,sensex,^BSESN` or `NASDAQ,apple` (no ticker needed for US).")

input1 = st.text_input("Enter Market, Keyword and Ticker", "Indian,sensex,^BSESN")
list1 = [x.strip() for x in input1.split(",")]

load_state = st.text("Preparing your report...")

try:
    market = list1[0]
    keyword = list1[1]
    ticker = list1[2] if len(list1) > 2 else None
except Exception:
    st.error("Invalid input format. Please use `Market,Keyword,Ticker`.")
    st.stop()

# -----------------------------
# Fetch and analyze news
# -----------------------------
newsapi = NewsApiClient(api_key=st.secrets["NEWSAPI_KEY"])
tod = datetime.datetime.now()
a = tod - datetime.timedelta(days=29)

sources = "google-news-in,the-hindu" if market.lower() == "indian" else "google-news-in,cbs-news,cnn"
articles = newsapi.get_everything(
    q=keyword,
    sources=sources,
    from_param=str(a)[:10],
    to=str(tod)[:10],
    language="en",
    sort_by="relevancy"
)

df_articles = analyze_articles(articles["articles"])
if df_articles.empty:
    st.warning("No news articles found.")
    st.stop()

scores = build_scores(df_articles)

# -----------------------------
# Trading strategy
# -----------------------------
if market.lower() == "indian":
    if not ticker:
        st.error("Please provide a Yahoo Finance ticker for Indian markets.")
        st.stop()
    data, sharpe, sharpe_bench = trading_strategy(ticker, scores, str(a)[:10], str(tod)[:10])
else:
    # default to Apple if no ticker
    data, sharpe, sharpe_bench = trading_strategy("AAPL", scores, str(a)[:10], str(tod)[:10])

if data.empty:
    st.warning("No market data found for given ticker.")
    st.stop()

# -----------------------------
# Plots
# -----------------------------

st.write(f"Report for stock: **{ticker or 'AAPL'}**")
st.write("### Sentiment Trends")
st.bar_chart(df_articles["polarity"])
st.bar_chart(df_articles["vader_sentiment"])

st.write("### Strategy vs Benchmark Returns")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data["returns"].cumsum(), mode="lines", name="Returns"))
fig.add_trace(go.Scatter(x=data["Date"], y=data["strategy_returns"].cumsum(), mode="lines", name="Strategy Returns"))
st.plotly_chart(fig)

st.metric("Annualised Sharpe Ratio (Strategy)", f"{sharpe:.2f}")
st.metric("Annualised Sharpe Ratio (Benchmark)", f"{sharpe_bench:.2f}")
