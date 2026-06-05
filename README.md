# 🤖 AI 個股預測小幫手

一個基於 Python 與 Streamlit 打造的互動式網頁應用程式，結合了傳統統計模型與現代深度學習技術，用來預測個股股價走勢，並透過 Google Gemini AI 產出專業的金融分析師報告。

---

## 🌟 核心功能

1. **📊 歷史資料即時抓取**：使用 `yfinance` 抓取美股或台股近一年的歷史交易資料，並顯示關鍵指標（公司名稱、產業類型、最新收盤價、52 週最高/最低價）。
2. **🤖 雙模型預測與對比**：
   - **線性迴歸 (Linear Regression)**：使用原生 NumPy 矩陣公式（`np.linalg.lstsq`）進行無庫（no-library）模型訓練。
   - **LSTM (長短期記憶網路)**：基於 `PyTorch` 建立輕量化循環神經網路，捕捉時間序列中的非線性特徵。
3. **📈 互動式圖表可視化**：採用 `Plotly` 繪製精美的互動式折線圖，直觀對比實際收盤價、線性迴歸預測值與 LSTM 預測值。
4. **📉 模型表現精準評估**：
   - 計算並呈現兩大模型的 **MAE (平均絕對誤差)**、**RMSE (均方根誤差)** 與 **方向準確率**。
   - 提供直觀的「優勝模型」對比與最後 20 天預測細節表格。
5. **💬 Gemini 金融分析師報告**：透過最新版 `google-genai` SDK，串接 `gemini-2.5-flash` 模型，依據預測數據自動生成 200 字內的專業股市分析與買賣建議。
6. **💡 學習筆記**：內建摺疊區塊，科普線性迴歸與 LSTM 的模型原理、優缺點及適用場景。

---

## 🛠️ 技術棧

- **網頁框架**：[Streamlit](https://streamlit.io/)
- **數據源**：[yfinance](https://github.com/ranaroussi/yfinance)
- **數據處理**：Pandas, NumPy
- **可視化**：Plotly
- **深度學習**：PyTorch
- **AI 報告**：Google GenAI SDK (`google-genai`)

---

## 🚀 快速開始

### 1. 複製專案並安裝依賴

請確保您的環境中已安裝 Python 3.8+。

```bash
# 建立並啟動虛擬環境 (選填)
python -m venv .venv
# Windows 啟動虛擬環境：
.venv\Scripts\activate
# macOS/Linux 啟動虛擬環境：
source .venv/bin/activate

# 安裝所需套件
pip install -r requirements.txt
```

### 2. 運行應用程式

在專案目錄下執行以下指令啟動 Streamlit 服務：

```bash
streamlit run app.py
```
啟動後，瀏覽器會自動開啟應用程式頁面（預設為 `http://localhost:8501`）。

---

## 🔑 Gemini API 金鑰設定

1. 前往 [Google AI Studio](https://aistudio.google.com/) 申請免費的 Gemini API Key。
2. 啟動本程式後，在左側側邊欄的 **🔑 Gemini API Key** 輸入框中貼上金鑰。
3. 點擊 **🚀 開始預測** 後，系統除預測股價外，亦會自動生成 Gemini AI 分析師報告。
> 💡 *若未填寫 API 金鑰，仍可正常使用股價預測與指標評估功能，僅「區塊 4：Gemini 金融分析師報告」會顯示提示。*

---

## 📁 專案結構

```text
opencode/
├── app.py              # Streamlit 應用程式主程式碼
├── requirements.txt    # 專案套件依賴清單
└── README.md           # 本說明文件
```

---

## 💡 免責聲明

**本專案所有預測與報告僅供學習與學術研究用途，不構成任何實際的投資與交易建議。股市投資存在風險，請讀者謹慎評估並自行承擔風險。**
