import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import traceback

import torch
import torch.nn as nn
import torch.optim as optim

from google import genai

st.set_page_config(layout="wide", page_title="AI 個股預測小幫手")
st.title("🤖 AI 個股預測小幫手")

if "stock_code" not in st.session_state:
    st.session_state.stock_code = "2330.TW"

with st.sidebar:
    st.header("🎯 選擇要分析的股票")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("台積電 2330.TW", use_container_width=True):
            st.session_state.stock_code = "2330.TW"
            st.rerun()
        if st.button("AAPL", use_container_width=True):
            st.session_state.stock_code = "AAPL"
            st.rerun()
    with col2:
        if st.button("鴻海 2317.TW", use_container_width=True):
            st.session_state.stock_code = "2317.TW"
            st.rerun()
        if st.button("NVDA", use_container_width=True):
            st.session_state.stock_code = "NVDA"
            st.rerun()

    stock_code = st.text_input("股票代碼", key="stock_code")
    st.caption("💡 台股請加 .TW (如 2330.TW)，美股直接輸入代號")

    st.divider()
    api_key = st.text_input("🔑 Gemini API Key", type="password")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("🚀 開始預測", type="primary", use_container_width=True)
    with col2:
        if st.button("⏹️ 清除快取", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

if not start_btn:
    st.info("👈 請在側邊欄選擇股票代碼，輸入 Gemini API Key（選填），然後點擊「🚀 開始預測」")
    st.stop()


@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    info = stock.info
    return df, info


def create_window_data(data, window=30):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window : i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train_test_split(X, y, train_ratio=0.8):
    n = len(X)
    n_train = int(n * train_ratio)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def linear_regression_fit(X, y):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    coeffs, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
    return coeffs


def linear_regression_predict(X, coeffs):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    return X_bias @ coeffs


class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm(model, X_train, y_train, epochs=150, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def direction_accuracy(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    true_dir = (y_true[1:] > y_true[:-1]).astype(int)
    pred_dir = (y_pred[1:] > y_pred[:-1]).astype(int)
    return float(np.mean(true_dir == pred_dir))


with st.spinner("📥 正在抓取歷史資料..."):
    try:
        df, info = fetch_stock_data(stock_code)
    except Exception as e:
        st.error(f"❌ 無法取得股票資料：{e}")
        st.stop()

min_required = 60
if len(df) < min_required:
    st.error(f"❌ 資料筆數不足（僅 {len(df)} 筆），至少需要 {min_required} 筆資料。")
    st.stop()

company_name = info.get("longName", stock_code)
industry = info.get("industry", "N/A")
latest_close = df["Close"].iloc[-1]
high_52w = df["Close"].max()
low_52w = df["Close"].min()

st.divider()
st.header("📊 歷史資料概覽")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🏢 公司名稱", company_name)
col2.metric("🏭 產業", industry)
col3.metric("📈 最新收盤價", f"${latest_close:.2f}")
col4.metric("📊 52週最高", f"${high_52w:.2f}")
col5.metric("📉 52週最低", f"${low_52w:.2f}")

prices = df["Close"].values.astype(np.float64)
dates = df.index

max_price = prices.max()
min_price = prices.min()
norm_prices = (prices - min_price) / (max_price - min_price + 1e-10)

window = 30
X, y = create_window_data(norm_prices, window)
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.8)

test_dates = dates[window:][len(X_train) :]

if len(y_test) < 5:
    st.error("❌ 測試集資料不足，無法進行預測。")
    st.stop()

st.divider()
st.header("🤖 AI 模型預測")

progress_bar = st.progress(0, text="訓練線性迴歸...")

lr_coeffs = linear_regression_fit(X_train, y_train)
lr_pred_norm = linear_regression_predict(X_test, lr_coeffs)
lr_pred = lr_pred_norm * (max_price - min_price) + min_price

progress_bar.progress(50, text="訓練 LSTM...")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

torch.manual_seed(42)
lstm_model = LSTMPredictor()
lstm_model = train_lstm(lstm_model, X_train_tensor, y_train_tensor, epochs=150)

lstm_model.eval()
with torch.no_grad():
    lstm_pred_norm = lstm_model(X_test_tensor).squeeze().numpy()
lstm_pred = lstm_pred_norm * (max_price - min_price) + min_price

progress_bar.progress(100, text="完成！")
progress_bar.empty()

y_test_actual = y_test * (max_price - min_price) + min_price

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=dates,
        y=prices,
        mode="lines",
        name="實際收盤價",
        line=dict(color="blue", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=lr_pred,
        mode="lines",
        name="線性迴歸預測",
        line=dict(color="red", width=2, dash="dash"),
    )
)
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=lstm_pred,
        mode="lines",
        name="LSTM 預測",
        line=dict(color="green", width=2, dash="dash"),
    )
)
fig.update_layout(
    title=f"{company_name} ({stock_code}) - 股價預測",
    xaxis_title="日期",
    yaxis_title="收盤價",
    height=500,
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.header("📈 模型表現評估")

lr_mae = mae(y_test_actual, lr_pred)
lr_rmse = rmse(y_test_actual, lr_pred)
lr_dir = direction_accuracy(y_test_actual, lr_pred)

lstm_mae = mae(y_test_actual, lstm_pred)
lstm_rmse = rmse(y_test_actual, lstm_pred)
lstm_dir = direction_accuracy(y_test_actual, lstm_pred)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📉 線性迴歸")
    st.metric("MAE", f"{lr_mae:.4f}")
    st.metric("RMSE", f"{lr_rmse:.4f}")
    st.metric("方向準確率", f"{lr_dir:.2%}")

with col2:
    st.subheader("🧠 LSTM")
    st.metric("MAE", f"{lstm_mae:.4f}")
    st.metric("RMSE", f"{lstm_rmse:.4f}")
    st.metric("方向準確率", f"{lstm_dir:.2%}")

with col3:
    st.subheader("🏆 比較")
    better_mae = "LSTM" if lstm_mae <= lr_mae else "線性迴歸"
    better_rmse = "LSTM" if lstm_rmse <= lr_rmse else "線性迴歸"
    better_dir = "LSTM" if lstm_dir >= lr_dir else "線性迴歸"
    st.metric("MAE 較優", better_mae)
    st.metric("RMSE 較優", better_rmse)
    st.metric("方向準確率較優", better_dir)

st.subheader("📋 最後 20 天預測對比")

n_last = min(20, len(y_test_actual))
last20_actual = y_test_actual[-n_last:]
last20_lr = lr_pred[-n_last:]
last20_lstm = lstm_pred[-n_last:]
last20_dates = test_dates[-n_last:]

comparison_df = pd.DataFrame(
    {
        "日期": [d.strftime("%Y-%m-%d") for d in last20_dates],
        "實際": [f"${v:.2f}" for v in last20_actual],
        "線性迴歸預測": [f"${v:.2f}" for v in last20_lr],
        "線性迴歸誤差%": [
            f"{abs(a - p) / a * 100:.2f}%" if a != 0 else "N/A"
            for a, p in zip(last20_actual, last20_lr)
        ],
        "LSTM 預測": [f"${v:.2f}" for v in last20_lstm],
        "LSTM 誤差%": [
            f"{abs(a - p) / a * 100:.2f}%" if a != 0 else "N/A"
            for a, p in zip(last20_actual, last20_lstm)
        ],
    }
)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.divider()
st.header("🤖 Gemini 金融分析師報告")

if not api_key:
    st.warning("⚠️ 請在側邊欄填入 Gemini API Key 以取得 AI 分析報告。")
else:
    with st.spinner("🤔 正在詢問 Gemini 分析師..."):
        metrics_text = (
            f"| 指標 | 線性迴歸 | LSTM |\n"
            f"|------|---------|------|\n"
            f"| MAE | {lr_mae:.4f} | {lstm_mae:.4f} |\n"
            f"| RMSE | {lr_rmse:.4f} | {lstm_rmse:.4f} |\n"
            f"| 方向準確率 | {lr_dir:.2%} | {lstm_dir:.2%} |\n"
        )
        prompt = f"""你是一位資深全球股市分析師。請根據以下資料，在 200 字內分析與比較兩個模型的表現、股票強弱與買進建議。

股票：{company_name} ({stock_code})
最新收盤價：${latest_close:.2f}

模型評估指標：
{metrics_text}

最後 20 天預測對比（日期、實際、線性迴歸預測、LSTM 預測）：
{comparison_df.to_string(index=False)}

請提供你的專業分析與買賣建議。"""

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            st.success("📄 分析結果：")
            st.write(response.text)
        except Exception as e:
            st.error(f"❌ Gemini API 呼叫失敗：{e}")
            traceback.print_exc()

st.divider()
st.header("💡 學習筆記")

with st.expander("📖 線性迴歸 vs LSTM 模型原理與區別"):
    st.markdown("""
### 線性迴歸 (Linear Regression)
- **原理**：透過最小平方法，尋找自變數（過去 30 天股價）與應變數（當天股價）之間的最佳線性關係。
- **優點**：計算快速、容易解釋。
- **缺點**：無法捕捉股價中的非線性與時間序列模式。

### LSTM (長短期記憶網路)
- **原理**：一種特殊的循環神經網路 (RNN)，透過「遺忘閘」、「輸入閘」、「輸出閘」的機制，解決傳統 RNN 的長期依賴問題，適合處理時間序列資料。
- **優點**：能學習長期依賴關係、捕捉非線性模式。
- **缺點**：訓練較慢、需要較多資料、超參數調校複雜。

### 主要區別
| 項目 | 線性迴歸 | LSTM |
|------|---------|------|
| 模型類型 | 統計模型 | 深度學習 |
| 非線性 | ❌ 無法 | ✅ 可以 |
| 時間序列 | ❌ 不擅長 | ✅ 擅長 |
| 訓練速度 | ⚡ 極快 | 🐢 較慢 |
| 解釋性 | ✅ 高 | ❌ 低 |
    """)
