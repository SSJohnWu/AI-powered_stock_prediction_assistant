import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import traceback

st.set_page_config(page_title="AI 個股預測小幫手", layout="wide")


class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def normalize(series):
    min_val = np.min(series)
    max_val = np.max(series)
    normalized = (series - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def denormalize(normalized, min_val, max_val):
    return normalized * (max_val - min_val) + min_val


def create_windows(data, window_size=30):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i])
        y.append(data[i])
    return np.array(X), np.array(y)


# ── Sidebar ──
with st.sidebar:
    st.title("🧭 控制面板")

    st.header("🎯 選擇要分析的股票")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("台積電 2330.TW", use_container_width=True):
            st.session_state.stock_code = "2330.TW"
        if st.button("AAPL", use_container_width=True):
            st.session_state.stock_code = "AAPL"
    with col2:
        if st.button("鴻海 2317.TW", use_container_width=True):
            st.session_state.stock_code = "2317.TW"
        if st.button("NVDA", use_container_width=True):
            st.session_state.stock_code = "NVDA"

    stock_code = st.text_input(
        "股票代碼",
        value=st.session_state.get("stock_code", "2330.TW"),
    )
    st.caption("💡 台股請加上 .TW 後綴，例如 2330.TW")

    st.header("🔑 Gemini API Key")
    api_key = st.text_input(
        "API Key", type="password", placeholder="輸入你的 Gemini API Key"
    )

    col_start, col_clear = st.columns(2)
    with col_start:
        start_btn = st.button("🚀 開始預測", type="primary", use_container_width=True)
    with col_clear:
        if st.button("⏹️ 清除快取", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

# ── Main ──
if not start_btn:
    st.stop()


# ── Block 1: Fetch Data ──
st.header(f"📊 區塊 1：抓取歷史資料 - {stock_code}")


@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    info = stock.info
    return df, info


try:
    df, info = fetch_stock_data(stock_code)
except Exception as e:
    st.error(f"無法取得股票資料：{e}")
    st.stop()

if df.empty or len(df) < 30:
    st.error(
        f"資料筆數不足 ({len(df)} 筆)，至少需要 30 筆資料才能進行預測。"
    )
    st.stop()

company_name = info.get("longName", info.get("shortName", stock_code))
industry = info.get("industry", info.get("sector", "N/A"))
latest_close = df["Close"].iloc[-1]
week52_high = info.get("fiftyTwoWeekHigh", df["Close"].max())
week52_low = info.get("fiftyTwoWeekLow", df["Close"].min())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("公司名稱", company_name)
col2.metric("產業類型", industry)
col3.metric("最新收盤價", f"{latest_close:.2f}")
col4.metric("52 週最高價", f"{week52_high:.2f}")
col5.metric("52 週最低價", f"{week52_low:.2f}")

# ── Prepare data ──
close_prices = df["Close"].values
dates = df.index

norm_prices, min_val, max_val = normalize(close_prices)

window_size = 30
X, y = create_windows(norm_prices, window_size)

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_dates = dates[window_size : window_size + len(X_train)]
test_dates = dates[window_size + len(X_train) : window_size + len(X)]

# ── Block 2: AI Model Prediction ──
st.header("🤖 區塊 2：AI 模型預測")

# ─ Linear Regression ─
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

lr_model = LinearRegression()
lr_model.fit(X_train_flat, y_train)
lr_test_pred = lr_model.predict(X_test_flat)

# ─ LSTM ─
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)

lstm_model = LSTMPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

epochs = 100
lstm_model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = lstm_model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

lstm_model.eval()
with torch.no_grad():
    lstm_test_pred = lstm_model(X_test_tensor).squeeze(-1).numpy()

# ─ Denormalize ─
lr_test_pred_denorm = denormalize(lr_test_pred, min_val, max_val)
lstm_test_pred_denorm = denormalize(lstm_test_pred, min_val, max_val)
y_test_denorm = denormalize(y_test, min_val, max_val)

# ─ Future 20 days (autoregressive recursive forecast) ─
last_window = norm_prices[-window_size:]
lr_future = []
lstm_future = []

lr_current_window = last_window.copy()
lstm_current_window = last_window.copy()

for _ in range(20):
    lr_next = lr_model.predict(lr_current_window.reshape(1, -1))[0]
    lr_future.append(lr_next)
    lr_current_window = np.append(lr_current_window[1:], lr_next)

    lstm_input = torch.FloatTensor(lstm_current_window).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        lstm_next = lstm_model(lstm_input).item()
    lstm_future.append(lstm_next)
    lstm_current_window = np.append(lstm_current_window[1:], lstm_next)

lr_future_denorm = denormalize(np.array(lr_future), min_val, max_val)
lstm_future_denorm = denormalize(np.array(lstm_future), min_val, max_val)

# ─ Future dates (handle timezone) ─
last_date = dates[-1]
if df.index.tz is not None:
    tz = df.index.tz
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=20, freq="B", tz=tz
    )
else:
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=20, freq="B"
    )

# ─ Plotly Chart ─
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=dates,
        y=df["Close"],
        mode="lines",
        name="實際歷史股價",
        line=dict(color="#1f77b4", width=2.5),
        hovertemplate="日期: %{x|%Y-%m-%d}<br>價格: %{y:.2f}<extra>實際歷史股價</extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=lr_test_pred_denorm,
        mode="lines",
        name="線性迴歸測試預測",
        line=dict(color="#d62728", width=2, dash="dash"),
        hovertemplate="日期: %{x|%Y-%m-%d}<br>預測: %{y:.2f}<extra>線性迴歸</extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=lstm_test_pred_denorm,
        mode="lines",
        name="LSTM 測試預測",
        line=dict(color="#2ca02c", width=2, dash="dash"),
        hovertemplate="日期: %{x|%Y-%m-%d}<br>預測: %{y:.2f}<extra>LSTM</extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=future_dates,
        y=lr_future_denorm,
        mode="lines+markers",
        name="線性迴歸未來預測",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
        marker=dict(size=5),
        hovertemplate="日期: %{x|%Y-%m-%d}<br>預測: %{y:.2f}<extra>線性迴歸未來</extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=future_dates,
        y=lstm_future_denorm,
        mode="lines+markers",
        name="LSTM 未來預測",
        line=dict(color="#9467bd", width=2, dash="dot"),
        marker=dict(size=5),
        hovertemplate="日期: %{x|%Y-%m-%d}<br>預測: %{y:.2f}<extra>LSTM 未來</extra>",
    )
)

fig.add_vrect(
    x0=test_dates[0],
    x1=test_dates[-1],
    fillcolor="rgba(255, 0, 0, 0.05)",
    layer="below",
    line_width=0,
    annotation_text="測試評估區",
    annotation_position="top left",
    annotation_font_size=12,
    annotation_font_color="red",
)

fig.add_vrect(
    x0=future_dates[0],
    x1=future_dates[-1],
    fillcolor="rgba(0, 0, 255, 0.05)",
    layer="below",
    line_width=0,
    annotation_text="未來預測區",
    annotation_position="top left",
    annotation_font_size=12,
    annotation_font_color="blue",
)

fig.update_layout(
    template="plotly_white",
    title=f"{stock_code} 股價預測圖",
    xaxis_title="日期",
    yaxis_title="股價",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
    xaxis=dict(gridcolor="#eef0f2"),
    yaxis=dict(gridcolor="#eef0f2"),
    height=600,
)

st.plotly_chart(fig, use_container_width=True)

# ── Block 3: Model Evaluation ──
st.header("📈 區塊 3：模型表現評估與未來走勢")

lr_mae = mean_absolute_error(y_test_denorm, lr_test_pred_denorm)
lr_rmse = np.sqrt(mean_squared_error(y_test_denorm, lr_test_pred_denorm))
lstm_mae = mean_absolute_error(y_test_denorm, lstm_test_pred_denorm)
lstm_rmse = np.sqrt(mean_squared_error(y_test_denorm, lstm_test_pred_denorm))


def direction_accuracy(actual, predicted):
    actual_dir = np.diff(actual) > 0
    pred_dir = np.diff(predicted) > 0
    return np.mean(actual_dir == pred_dir) * 100


lr_dir_acc = direction_accuracy(y_test_denorm, lr_test_pred_denorm)
lstm_dir_acc = direction_accuracy(y_test_denorm, lstm_test_pred_denorm)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📉 線性迴歸")
    lr_cols = st.columns(3)
    lr_cols[0].metric("MAE", f"{lr_mae:.4f}")
    lr_cols[1].metric("RMSE", f"{lr_rmse:.4f}")
    lr_cols[2].metric("方向準確率", f"{lr_dir_acc:.2f}%")
with col2:
    st.subheader("🧠 LSTM")
    lstm_cols = st.columns(3)
    lstm_cols[0].metric("MAE", f"{lstm_mae:.4f}")
    lstm_cols[1].metric("RMSE", f"{lstm_rmse:.4f}")
    lstm_cols[2].metric("方向準確率", f"{lstm_dir_acc:.2f}%")

# ─ Last 20 days comparison ─
st.subheader("📋 測試集最後 20 天對比")
last_20_idx = -20 if len(test_dates) >= 20 else -len(test_dates)

comparison_df = pd.DataFrame(
    {
        "日期": test_dates[last_20_idx:],
        "實際": y_test_denorm[last_20_idx:],
        "線性迴歸預測": lr_test_pred_denorm[last_20_idx:],
        "線性迴歸誤差%": (
            (y_test_denorm[last_20_idx:] - lr_test_pred_denorm[last_20_idx:])
            / y_test_denorm[last_20_idx:]
            * 100
        ).round(2),
        "LSTM 預測": lstm_test_pred_denorm[last_20_idx:],
        "LSTM 誤差%": (
            (y_test_denorm[last_20_idx:] - lstm_test_pred_denorm[last_20_idx:])
            / y_test_denorm[last_20_idx:]
            * 100
        ).round(2),
    }
)
comparison_df["日期"] = comparison_df["日期"].dt.strftime("%Y-%m-%d")
st.dataframe(comparison_df, hide_index=True, use_container_width=True)

# ─ Future 20 days table ─
st.subheader("🔮 未來 20 天預測值")
future_df = pd.DataFrame(
    {
        "日期": future_dates,
        "線性迴歸預測值": lr_future_denorm,
        "LSTM 預測值": lstm_future_denorm,
    }
)
future_df["日期"] = future_df["日期"].dt.strftime("%Y-%m-%d")
st.dataframe(future_df, hide_index=True, use_container_width=True)

# ── Block 4: Gemini Financial Analyst Report ──
st.header("🤖 區塊 4：Gemini 金融分析師報告")

if not api_key:
    st.warning("⚠️ 請在側邊欄輸入 Gemini API Key 以產生 AI 分析報告。")
else:
    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        prompt = f"""你是一位資深全球股市分析師，請根據以下數據進行分析：

## 測試集表現指標
### 線性迴歸
- MAE: {lr_mae:.4f}
- RMSE: {lr_rmse:.4f}
- 方向準確率: {lr_dir_acc:.2f}%

### LSTM
- MAE: {lstm_mae:.4f}
- RMSE: {lstm_rmse:.4f}
- 方向準確率: {lstm_dir_acc:.2f}%

## 測試集最後 20 天對比
{comparison_df.to_string(index=False)}

## 未來 20 天預測
{future_df.to_string(index=False)}

請在 200 字內分析與比較兩個模型在測試集上的表現、未來 20 天走勢以及提供具體的買賣建議。"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        st.markdown("### 📝 AI 分析報告")
        st.write(response.text)

    except Exception as e:
        st.error(f"Gemini API 呼叫失敗：{e}")
        traceback.print_exc()

# ── Block 5: Learning Notes ──
st.header("💡 區塊 5：學習筆記")
with st.expander("📖 線性迴歸 vs LSTM 模型原理與區別"):
    st.markdown("""
### 線性迴歸 (Linear Regression)
線性迴歸是一種監督式學習演算法，透過建立自變數（X）與應變數（y）之間的線性關係來進行預測。其數學形式為 \\( y = wX + b \\)，其中 \\( w \\) 為權重、\\( b \\) 為偏置。在股價預測中，我們使用過去 30 天的股價作為特徵，透過最小平方法（OLS）找出最佳擬合直線來預測隔天股價。

### LSTM (長短期記憶網路)
LSTM 是一種特殊的循環神經網路（RNN），專門設計用來解決長期依賴問題。它透過遺忘閘、輸入閘、輸出閘三種閘門機制，控制資訊的保留與遺忘，有效避免梯度消失或爆炸問題。在股價預測中，LSTM 能捕捉時間序列中的長期依賴關係與非線性模式。

### 主要區別

| 特性 | 線性迴歸 | LSTM |
|------|---------|------|
| 模型類型 | 統計模型 | 深度學習 |
| 非線性能力 | 僅能捕捉線性關係 | 可捕捉複雜非線性模式 |
| 序列記憶 | 無記憶能力 | 具長期記憶能力 |
| 訓練速度 | 極快（秒級） | 較慢（需多個 epochs） |
| 資料需求 | 少量即可 | 需較多資料 |
| 可解釋性 | 高（權重可直接解釋） | 低（黑箱模型） |
| 過擬合風險 | 低 | 高（需正則化） |
    """)
