# 📧 HW3 — Spam/Ham Email Classifier

本專案改編自 Packt 出版社的《Hands-On Artificial Intelligence for Cybersecurity》第 3 章，  
並延伸加入 **前處理 (preprocessing)**、**可視化 (visualization)** 與 **Streamlit 互動介面**，  
用以訓練與分析垃圾郵件（Spam）與正常郵件（Ham）分類器。

---

## 📚 專案來源 (Source Reference)

原始範例：  
👉 [PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity.git)

本專案為延伸實作版本，使用 OpenSpec 與 AI Coding CLI 完成開發。

---
---

## ☁️ Demo 網址 (Streamlit Cloud)

🔗 **線上展示**：  
👉 [https://lic924-hw3-spam-email-app-streamlit-tkfzpe.streamlit.app/](https://lic924-hw3-spam-email-app-streamlit-tkfzpe.streamlit.app/)  

💡 建議使用桌面瀏覽器開啟以獲得完整互動體驗。

---

## 🧩 功能說明 (Features)

- 📊 **資料前處理**：正則表達式清理文字、替換 URL / Email / Phone / 數字為統一標記。
- 🧹 **文字特徵化**：使用 `TF-IDF` 與 `CountVectorizer` 向量化。
- 🧠 **模型訓練**：採用 `LogisticRegression` 進行二元分類。
- 🎨 **互動式視覺化**：
  - 類別分佈圖 (Class Distribution)
  - Top-N 常見詞彙分析
  - 混淆矩陣 (Confusion Matrix)
  - ROC / PR 曲線
  - Threshold sweep (Precision / Recall / F1)
- 🌐 **Streamlit 介面**：支援資料選擇、訓練比例調整、門檻 (threshold) 視覺化與即時預測。

---

## 🧰 專案架構 (Project Structure)

```bash
Chapter03/
├── datasets/              # 原始與處理後的資料集
│   ├── sms_spam_no_header.csv
│   ├── processed_spam.csv
│   └── ...
├── models/                # 儲存訓練後模型與向量器
│   ├── model.joblib
│   └── vectorizer.joblib
├── sources/               # 截圖或說明圖
│   ├── overview.png
│   ├── roc_pr.png
│   └── ...
├── app_streamlit.py       # Streamlit 主應用程式
├── prepare_dataset.py     # 前處理程式
├── requirements.txt       # 相依套件清單
└── README.md              # 專案說明文件
```

---

## 🧪 使用方式 (Usage)

1. 左側選擇資料集（例如 `sms_spam_no_header.csv`）  
2. 指定 **Label** 與 **Text** 欄位  
3. 調整 **測試集比例 (Test Size)**、**Seed**、**Threshold**  
4. 檢視下列結果：
   - 📊 類別分佈 (Class Distribution)
   - 🔠 Top Tokens by Class
   - 🧮 混淆矩陣、ROC / PR 曲線
   - 🎯 Threshold Sweep (Precision / Recall / F1)
5. 在 **「Live Inference」** 區輸入文字訊息，可即時分類為 **spam / ham**

---

## ⚙️ 安裝與執行 (Installation & Run)

```bash
# 1️⃣ 建立虛擬環境
python -m venv .venv
source .venv/bin/activate        # Windows 用 .venv\Scripts\activate

# 2️⃣ 安裝相依套件
pip install -r requirements.txt

# 3️⃣ 執行 Streamlit 應用程式
streamlit run app_streamlit.py

---