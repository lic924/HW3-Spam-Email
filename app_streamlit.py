# app_streamlit.py
# Spam/Ham Classifier — Phase 4 Visualizations (pickle-safe, final)

import os, re, glob, string, html
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

import streamlit as st

# ============ PAGE CONFIG ============
st.set_page_config(page_title="Spam/Ham Classifier — Phase 4 Visualizations", layout="wide")
TITLE = "Spam/Ham Classifier — Phase 4 Visualizations"
SUBTITLE = "Interactive dashboard for data distribution, token patterns, and model performance"

# ============ REGEX & CONSTANTS ============
URL_RE   = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-\s.]*)?(?:\(?\d{2,4}\)?[-\s.]*)?\d{3,4}[-\s.]?\d{3,4}\b")
NUM_RE   = re.compile(r"\b\d+(?:[.,]\d+)?\b")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# 讓 <URL> / <EMAIL> / <PHONE> / <NUM> 也會被視為 token，同時保留一般單字
# 這個 token_pattern 只是一個字串（內建可序列化），避免自訂函式被 pickle
TOKEN_PATTERN = r'(?u)<[^>\s]+>|\b\w+\b'

SPECIAL_TAGS = {"<url>", "<email>", "<phone>", "<num>"}

# ============ CLEANING (僅在 pandas 端執行，不塞進向量器) ============
def clean_and_mask_text(s: str) -> str:
    if s is None:
        s = ""
    s = html.unescape(str(s))   # 先處理 &lt; &amp; ...
    s = s.lower()
    s = URL_RE.sub(" <URL> ", s)
    s = EMAIL_RE.sub(" <EMAIL> ", s)
    s = PHONE_RE.sub(" <PHONE> ", s)
    s = NUM_RE.sub(" <NUM> ", s)
    tokens = []
    for tok in s.split():
        if tok in SPECIAL_TAGS:
            tokens.append(tok)
        else:
            tok = tok.translate(PUNCT_TABLE)
            # 丟掉 html entity 殘片或空 token
            if not tok or tok.startswith("&") or ";" in tok:
                continue
            tokens.append(tok)
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()

def count_special_tokens(series: pd.Series) -> pd.DataFrame:
    def counter(txt):
        t = str(txt)
        return {
            "<URL>": t.count("<url>"),
            "<EMAIL>": t.count("<email>"),
            "<PHONE>": t.count("<phone>"),
            "<NUM>": t.count("<num>"),
        }
    agg = Counter()
    for t in series:
        agg.update(counter(t))
    return (
        pd.DataFrame({"count": pd.Series(agg)})
        .reindex(["<URL>", "<EMAIL>", "<PHONE>", "<NUM>"])
        .fillna(0).astype(int)
    )

@st.cache_data(show_spinner=False)
def list_csvs(base_dirs=(".", "data", "datasets")):
    paths = []
    for d in base_dirs:
        if os.path.isdir(d):
            paths += glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
    return sorted(set(paths), key=lambda p: (p.count(os.sep), p))

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

# ============ VECTORIZERS（純內建參數，pickle-safe） ============
@st.cache_resource(show_spinner=False)
def get_tfidf_vectorizer():
    return TfidfVectorizer(
        # 我們已在 pandas 端清洗過了
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
    )

def make_count_vectorizer(remove_stopwords: bool):
    return CountVectorizer(
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 1),
        min_df=1,
        max_df=1.0,
        stop_words=("english" if remove_stopwords else None),
    )

def compute_top_tokens_by_class(cleaned_texts: pd.Series, labels: pd.Series, topn: int,
                                remove_stopwords: bool, exclude_special: bool):
    cv = make_count_vectorizer(remove_stopwords)
    try:
        X = cv.fit_transform(cleaned_texts.astype(str))
    except ValueError:
        empty = pd.DataFrame({"token": [], "frequency": []})
        return empty, empty

    vocab = np.array(cv.get_feature_names_out())

    # 過濾：排除特殊 token、去掉單字母或非 ASCII（讓圖更乾淨）
    mask = np.ones(len(vocab), dtype=bool)
    if exclude_special:
        mask &= ~np.isin(vocab, list(SPECIAL_TAGS))
    mask &= np.array([len(t) > 1 and t.isascii() for t in vocab])

    if not mask.any():
        empty = pd.DataFrame({"token": [], "frequency": []})
        return empty, empty

    vocab = vocab[mask]
    X = X[:, mask]

    y = pd.Series(labels).astype(int).values
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    c0 = np.asarray(X[idx0].sum(axis=0)).ravel() if len(idx0) else np.zeros(len(vocab), dtype=int)
    c1 = np.asarray(X[idx1].sum(axis=0)).ravel() if len(idx1) else np.zeros(len(vocab), dtype=int)

    df0 = pd.DataFrame({"token": vocab, "frequency": c0}).sort_values("frequency", ascending=False).head(topn)
    df1 = pd.DataFrame({"token": vocab, "frequency": c1}).sort_values("frequency", ascending=False).head(topn)
    return df0, df1

def plot_simple_bar(df: pd.DataFrame, x: str, y: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df[x], df[y])
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.set_xticklabels(df[x], rotation=90)
    fig.tight_layout()
    return fig

def plot_confusion_table(cm: np.ndarray):
    return pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])

def decision_scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    raise ValueError("Classifier has no continuous score (predict_proba/decision_function).")

# ============ SIDEBAR ============
with st.sidebar:
    st.title("Inputs")
    dataset_path = st.selectbox("Dataset CSV", list_csvs(), index=0, placeholder="Select a CSV…")
    models_dir   = st.text_input("Models dir", value="models")
    test_size    = st.slider("Test size", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    seed         = st.number_input("Seed", min_value=0, value=42, step=1)
    decision_th  = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
    remove_sw    = st.checkbox("Top Tokens 移除英文停用詞（只影響圖表）", value=False)
    exclude_special = st.checkbox("Top Tokens 排除特殊標記（<URL>/<EMAIL>/<PHONE>/<NUM>）", value=True)

# ============ LOAD DATA ============
@st.cache_data(show_spinner=True)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_csv(dataset_path)

with st.sidebar:
    label_col = st.selectbox("Label column", options=list(df.columns), index=0)
    text_col  = st.selectbox("Text column",  options=list(df.columns), index=1 if len(df.columns) > 1 else 0)

# ============ HEADER ============
st.markdown(f"# {TITLE}")
st.markdown(SUBTITLE)

# ============ PREPARE ============
df = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"}).dropna()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

# label → 0/1
if not pd.api.types.is_numeric_dtype(df["label"]):
    uniq = list(df["label"].astype(str).unique())
    mapping = {v: i for i, v in enumerate(uniq)}
    df["label"] = df["label"].astype(str).map(mapping)
else:
    df["label"] = df["label"].astype(int)

# 清洗文本（僅在 pandas 端）
cleaned_text = df["text"].astype(str).apply(clean_and_mask_text)

# ============ DATA OVERVIEW ============
st.markdown("## Data Overview")
c1, c2 = st.columns([1.2, 1.0])

with c1:
    cls_cnt = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["ham", "spam"], [cls_cnt.get(0, 0), cls_cnt.get(1, 0)])
    ax.set_title("Class distribution")
    fig.tight_layout()
    st.pyplot(fig)

with c2:
    st.caption("Token replacements in cleaned text (approximate)")
    st.dataframe(count_special_tokens(cleaned_text))

# ============ TOP TOKENS ============
st.markdown("## Top Tokens by Class")
topn = st.slider("Top-N tokens", min_value=5, max_value=50, value=20, step=1, key="topn_tokens")

df_ham, df_spam = compute_top_tokens_by_class(
    cleaned_text, df["label"], topn=topn, remove_stopwords=remove_sw, exclude_special=exclude_special
)

c3, c4 = st.columns(2)
with c3:
    st.markdown("**Class: ham**")
    if df_ham.empty:
        st.info("No tokens found for ham（可取消『移除停用詞』或『排除特殊標記』看看）")
    else:
        st.pyplot(plot_simple_bar(df_ham.sort_values("frequency", ascending=True), "token", "frequency", "Top tokens (ham)"))
with c4:
    st.markdown("**Class: spam**")
    if df_spam.empty:
        st.info("No tokens found for spam（可取消『移除停用詞』或『排除特殊標記』看看）")
    else:
        st.pyplot(plot_simple_bar(df_spam.sort_values("frequency", ascending=True), "token", "frequency", "Top tokens (spam)"))

# ============ TRAIN / EVAL ============
st.markdown("## Model Performance (Test)")

@st.cache_resource(show_spinner=True)
def train_and_eval(cleaned_series: pd.Series, labels: pd.Series, test_size: float, seed: int, save_dir: str):
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_series.astype(str), labels.astype(int),
        test_size=test_size, stratify=labels, random_state=seed
    )
    vec = get_tfidf_vectorizer()
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=300)
    clf.fit(Xtr, y_train)

    ensure_dir(save_dir)
    # 向量器不含自訂函式，joblib 可正常序列化
    joblib.dump(vec, os.path.join(save_dir, "vectorizer.joblib"))
    joblib.dump(clf, os.path.join(save_dir, "model.joblib"))

    scores = decision_scores(clf, Xte)
    fpr, tpr, _ = roc_curve(y_test, scores)
    prec, rec, thr = precision_recall_curve(y_test, scores)

    return {"y_test": y_test, "scores": scores, "fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec, "thr": thr}

res = train_and_eval(cleaned_text, df["label"], test_size, seed, models_dir)

# Confusion matrix（依 threshold，固定 2x2）
y_test = res["y_test"]; scores = res["scores"]
y_pred = (scores >= decision_th).astype(int)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
st.markdown("### Confusion matrix")
st.dataframe(plot_confusion_table(cm))

# ROC & PR
cc1, cc2 = st.columns(2)
with cc1:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(res["fpr"], res["tpr"])
    ax.plot([0,1],[0,1],'--', linewidth=1)
    ax.set_title("ROC"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); fig.tight_layout()
    st.pyplot(fig)
with cc2:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(res["rec"], res["prec"])
    ax.set_title("Precision-Recall"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); fig.tight_layout()
    st.pyplot(fig)

# Threshold sweep（0.30~0.80、步長 0.05）
st.markdown("### Threshold sweep (precision/recall/f1)")
def sweep_threshold(y_true, scores, start=0.30, end=0.80, step=0.05):
    y_true = pd.Series(y_true).astype(int).values
    thresholds = np.round(np.arange(start, end + 1e-9, step), 2)
    rows = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        cm2 = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm2[0, 0], cm2[0, 1], cm2[1, 0], cm2[1, 1]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({"threshold": float(t), "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)})
    return pd.DataFrame(rows)

st.dataframe(sweep_threshold(y_test, scores, 0.30, 0.80, 0.05))

# ============ LIVE INFERENCE ============
st.markdown("## Live Inference")

@st.cache_resource(show_spinner=False)
def load_artifacts(path: str):
    vec = joblib.load(os.path.join(path, "vectorizer.joblib"))
    clf = joblib.load(os.path.join(path, "model.joblib"))
    return vec, clf

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Use spam example"):
        st.session_state["live_text"] = "Congratulations! You won a prize. Click http://bit.ly/free-cash to claim now."
with col_b:
    if st.button("Use ham example"):
        st.session_state["live_text"] = "Hi, are we still on for the meeting tomorrow at 10am? Let me know."

live_text = st.text_area("Enter a message to classify", value=st.session_state.get("live_text", ""), height=160)
if st.button("Predict"):
    try:
        vec, clf = load_artifacts(models_dir)
        X = vec.transform([clean_and_mask_text(live_text)])  # 與訓練一致：先清洗再向量化
        if hasattr(clf, "predict_proba"):
            sc = float(clf.predict_proba(X)[:, 1][0])
        elif hasattr(clf, "decision_function"):
            sc = 1.0 / (1.0 + np.exp(-float(clf.decision_function(X)[0])))
        else:
            raise ValueError("Model has no score output.")
        pred = int(sc >= decision_th)
        label = "spam" if pred == 1 else "ham"
        st.success(f"Prediction: **{label}**  |  score={sc:.3f}  |  threshold={decision_th:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("此版本不把自訂函式塞進向量器，完全避免 PicklingError。Top Tokens 可移除英文停用詞以及排除 <URL>/<EMAIL>/<PHONE>/<NUM>。")