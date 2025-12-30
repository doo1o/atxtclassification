import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from typing import List, Tuple, Dict

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]  # AGNews标准4类


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize(text: str) -> List[str]:
    text = clean_text(text)
    # 更稳健一些：只保留字母数字和'，其余当分隔
    return re.findall(r"[a-z0-9']+", text)


def build_vocab(texts: List[str], min_freq: int = 2, max_vocab: int = 50000) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in counter.most_common():
        if c < min_freq:
            break
        if w in vocab:
            continue
        vocab[w] = len(vocab)
        if len(vocab) >= max_vocab:
            break
    return vocab


def numericalize(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]


def load_agnews_csv(train_csv: str, test_csv: str) -> Tuple[List[str], List[int], List[str], List[int], List[str]]:

    def read_one(path: str):

        df = pd.read_csv(path)

        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols

        if "class index" in cols:

            label_col = "class index"
            title_col = "title" if "title" in cols else None
            desc_col = "description" if "description" in cols else None

            labels = df[label_col].astype(int).tolist()  # 1..4
            if title_col and desc_col:
                texts = (df[title_col].fillna("") + " " + df[desc_col].fillna("")).astype(str).tolist()
            elif desc_col:
                texts = df[desc_col].fillna("").astype(str).tolist()
            else:

                others = [c for c in cols if c != label_col]
                texts = df[others].fillna("").astype(str).agg(" ".join, axis=1).tolist()

        elif "class_index" in cols:
            labels = df["class_index"].astype(int).tolist()
            title = df["title"].fillna("").astype(str) if "title" in cols else ""
            desc = df["description"].fillna("").astype(str) if "description" in cols else ""
            texts = (title + " " + desc).astype(str).tolist()

        elif "label" in cols and "text" in cols:
            labels = df["label"].astype(int).tolist()
            texts = df["text"].fillna("").astype(str).tolist()

        elif "category" in cols and "text" in cols:

            cat = df["category"].astype(str).tolist()
            name2id = {n.lower(): i for i, n in enumerate(LABEL_NAMES)}
            labels = [name2id.get(x.lower(), 0) for x in cat]
            texts = df["text"].fillna("").astype(str).tolist()

        else:

            df2 = pd.read_csv(path, header=None)
            if df2.shape[1] >= 3:
                labels = df2.iloc[:, 0].astype(int).tolist()
                texts = (df2.iloc[:, 1].fillna("").astype(str) + " " + df2.iloc[:, 2].fillna("").astype(str)).tolist()
            elif df2.shape[1] == 2:
                labels = df2.iloc[:, 0].astype(int).tolist()
                texts = df2.iloc[:, 1].fillna("").astype(str).tolist()
            else:
                raise ValueError(f"Unrecognized CSV format: {path}")


        if min(labels) == 1 and max(labels) == 4:
            labels = [x - 1 for x in labels]

        return texts, labels

    tr_texts, tr_labels = read_one(train_csv)
    te_texts, te_labels = read_one(test_csv)
    return tr_texts, tr_labels, te_texts, te_labels, LABEL_NAMES


class TextClsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = numericalize(self.texts[idx], self.vocab)[: self.max_len]
        y = self.labels[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_batch(batch, pad_id: int = 0):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    maxlen = int(lengths.max().item()) if len(lengths) else 1
    padded = torch.full((len(xs), maxlen), pad_id, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, : x.size(0)] = x
    return padded, lengths, torch.stack(ys)
