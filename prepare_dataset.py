import os, re, argparse, string
import pandas as pd

URL_RE   = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-\s.]*)?(?:\(?\d{2,4}\)?[-\s.]*)?\d{3,4}[-\s.]?\d{3,4}\b")
NUM_RE   = re.compile(r"\b\d+(?:[.,]\d+)?\b")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def clean_and_mask(text: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = URL_RE.sub(" <URL> ", text)
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = PHONE_RE.sub(" <PHONE> ", text)
    text = NUM_RE.sub(" <NUM> ", text)
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def read_table(path, no_header=False):
    sep = "\t" if path.endswith(".tsv") else ","
    if no_header:
        return pd.read_csv(path, sep=sep, header=None)
    return pd.read_csv(path, sep=sep)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-header", action="store_true", help="來源檔沒有欄位名時使用")
    ap.add_argument("--label-col", help="標籤欄位名（有 header 時用）")
    ap.add_argument("--text-col", help="文字欄位名（有 header 時用）")
    ap.add_argument("--label-idx", type=int, help="標籤欄位索引（無 header 時用）")
    ap.add_argument("--text-idx", type=int, help="文字欄位索引（無 header 時用）")
    args = ap.parse_args()

    # 讀檔
    df = read_table(args.input, no_header=args.no_header)

    # 取得 label/text 欄
    if args.no_header:
        # ✅ 修正：用底線屬性名稱
        if args.label_idx is None or args.text_idx is None:
            raise SystemExit("❌ 無 header 時請加 --label-idx 與 --text-idx（例如 0 與 1）")
        label_series = df.iloc[:, args.label_idx]
        text_series  = df.iloc[:, args.text_idx]
    else:
        if args.label_col and args.text_col:
            label_series = df[args.label_col]
            text_series  = df[args.text_col]
        else:
            # 嘗試自動偵測常見欄位名
            possible_label = ["label","Category","class","v1","spam"]
            possible_text  = ["text","Message","content","v2","email","url"]
            label_name = next((c for c in df.columns if c in possible_label), None)
            text_name  = next((c for c in df.columns if c in possible_text), None)
            if label_name is None or text_name is None:
                raise SystemExit(
                    f"❌ 找不到常見欄位，請用 --label-col 與 --text-col 指定。現有欄位：{list(df.columns)}"
                )
            label_series = df[label_name]
            text_series  = df[text_name]

    out = pd.DataFrame({"label": label_series, "text": text_series})

    # 去除全空白/NaN
    out["text"] = out["text"].astype(str).str.strip()
    out = out.dropna(subset=["label", "text"])
    out = out[out["text"] != ""]

    # label 規一化：spam/ham -> 1/0；如果不是，則自動編碼 0/1
    out["label"] = out["label"].astype(str).str.lower().map({"spam":1, "ham":0}).fillna(out["label"])
    if not pd.api.types.is_numeric_dtype(out["label"]):
        uniq = list(pd.Series(out["label"]).astype(str).unique())
        mapping = {v:i for i,v in enumerate(uniq)}
        out["label"] = out["label"].astype(str).map(mapping)
        print(f"ℹ️ 自動對應 label：{mapping}")

    # 文字清洗
    out["text"] = out["text"].apply(clean_and_mask)

    # 去重
    out = out.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)

    # 確保輸出資料夾存在
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 輸出
    out.to_csv(args.output, index=False)

    print(f"✅ 輸出 {len(out)} 筆到 {args.output}")
    print(out["label"].value_counts())
    print("👀 範例：")
    print(out.head(3))

if __name__ == "__main__":
    main()