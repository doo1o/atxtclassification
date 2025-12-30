# predict.py
import os, json, argparse
import torch
import torch.nn.functional as F

from model import AdvancedTransformerClassifier

def tokenize(text: str):
    return text.lower().split()

def numericalize(text: str, vocab):
    toks = tokenize(text)
    unk = vocab.get("<unk>", 1)
    return [vocab.get(t, unk) for t in toks]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_id2label(label2id: dict):

    max_id = max(label2id.values())
    id2label = [""] * (max_id + 1)
    for lab, i in label2id.items():
        id2label[int(i)] = lab
    return id2label

def load_model(ckpt_dir: str, device: str):
    vocab = load_json(os.path.join(ckpt_dir, "vocab.json"))
    label2id = load_json(os.path.join(ckpt_dir, "label2id.json"))
    id2label = build_id2label(label2id)


    config_path = os.path.join(ckpt_dir, "config.json")
    config = load_json(config_path) if os.path.exists(config_path) else {}
    max_len = int(config.get("max_len", 256))


    d_model = int(config.get("d_model", 384))
    nhead = int(config.get("nhead", 8))
    num_layers = int(config.get("num_layers", 6))
    d_ff = int(config.get("d_ff", 1536))
    dropout = float(config.get("dropout", 0.1))
    attn_dropout = float(config.get("attn_dropout", 0.1))
    use_attention_pooling = bool(config.get("use_attention_pooling", True))

    model = AdvancedTransformerClassifier(
        vocab_size=len(vocab),
        num_classes=len(id2label),
        pad_id=vocab["<pad>"],
        max_len=max_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        attn_dropout=attn_dropout,
        use_attention_pooling=use_attention_pooling,
    ).to(device)


    model_path1 = os.path.join(ckpt_dir, "model.pt")
    model_path2 = os.path.join(ckpt_dir, "best.pt")

    if os.path.exists(model_path1):
        state = torch.load(model_path1, map_location=device)
    elif os.path.exists(model_path2):
        obj = torch.load(model_path2, map_location=device)
        state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    else:
        raise FileNotFoundError("找不到 model.pt 或 best.pt")

    model.load_state_dict(state, strict=True)
    model.eval()

    return model, vocab, id2label, max_len

@torch.no_grad()
def predict_one(model, vocab, id2label, text: str, max_len: int, device: str):
    ids = numericalize(text, vocab)[:max_len]
    if len(ids) == 0:
        return "EMPTY", 0.0

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1,T]
    logits = model(x)[0]  # [C]
    prob = F.softmax(logits, dim=-1)
    pred_id = int(prob.argmax().item())
    return id2label[pred_id], float(prob[pred_id].item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True, help="包含 model.pt/vocab.json/label2id.json 的目录")
    ap.add_argument("--text", type=str, required=True, help="要分类的文本")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, id2label, max_len = load_model(args.ckpt_dir, device)

    label, score = predict_one(model, vocab, id2label, args.text, max_len, device)
    print(f"pred: {label}\nscore: {score:.4f}")

if __name__ == "__main__":
    main()
