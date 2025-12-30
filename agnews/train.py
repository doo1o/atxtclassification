import os
import json
import math
import random
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataset import (
    load_agnews_csv, build_vocab, TextClsDataset, collate_batch
)
from model import AdvancedTransformerClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        # logits: [B,C], target: [B]
        n_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y = []
    all_pred = []
    for input_ids, lengths, y in loader:
        input_ids = input_ids.to(device)
        y = y.to(device)
        logits = model(input_ids)
        pred = logits.argmax(dim=-1)
        all_y.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
    acc = accuracy_score(all_y, all_pred)
    macro_f1 = f1_score(all_y, all_pred, average="macro")
    return acc, macro_f1, all_y, all_pred

def save_artifacts(save_dir, model, vocab, label2id, config):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(save_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/du/code/txtClassification/data/agnews")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_vocab", type=int, default=50000)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)


    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--use_attention_pooling", action="store_true")

    parser.add_argument("--save_dir", type=str, default="./runs/agnews_adv_tf1")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_csv = os.path.join(args.data_root, "train.csv")
    test_csv = os.path.join(args.data_root, "test.csv")
    tr_texts_full, tr_labels_full, te_texts, te_labels, label_names = load_agnews_csv(train_csv, test_csv)
    num_classes = len(label_names)

    # train/val split
    tr_texts, va_texts, tr_labels, va_labels = train_test_split(
        tr_texts_full, tr_labels_full, test_size=0.1, random_state=args.seed, stratify=tr_labels_full
    )
    print(f"split => train {len(tr_texts)}, val {len(va_texts)}, test {len(te_texts)}")
    print("labels:", label_names)

    # vocab on train only
    vocab = build_vocab(tr_texts, min_freq=args.min_freq, max_vocab=args.max_vocab)
    pad_id = vocab["<pad>"]
    print("vocab_size:", len(vocab), "pad_id:", pad_id)

    # dataset + loader
    train_ds = TextClsDataset(tr_texts, tr_labels, vocab, max_len=args.max_len)
    val_ds   = TextClsDataset(va_texts, va_labels, vocab, max_len=args.max_len)
    test_ds  = TextClsDataset(te_texts, te_labels, vocab, max_len=args.max_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id=pad_id),
        num_workers=2, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id=pad_id),
        num_workers=2, pin_memory=(device == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id=pad_id),
        num_workers=2, pin_memory=(device == "cuda")
    )

    # model
    model = AdvancedTransformerClassifier(
        vocab_size=len(vocab),
        num_classes=num_classes,
        pad_id=pad_id,
        max_len=args.max_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        use_attention_pooling=args.use_attention_pooling,
    ).to(device)

    # loss/opt/sched
    criterion = LabelSmoothingCE(args.label_smoothing) if args.label_smoothing > 0 else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # train loop with early stopping
    best_val_f1 = -1.0
    bad_epochs = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for input_ids, lengths, y in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(input_ids)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            global_step += 1
            losses.append(loss.item())

        tr_acc, tr_f1, _, _ = evaluate(model, train_loader, device)
        va_acc, va_f1, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | loss {np.mean(losses):.4f} | train acc {tr_acc:.4f} f1 {tr_f1:.4f} | val acc {va_acc:.4f} f1 {va_f1:.4f}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            bad_epochs = 0

            print("  new best, saving artifacts...")

            ckpt_path = os.path.join(args.save_dir, "best.pt")
            torch.save({
                "model": model.state_dict(),
                "vocab": vocab,
                "label_names": label_names,
                "args": vars(args),
            }, ckpt_path)

            label2id = {name: i for i, name in enumerate(label_names)}
            config = {
                "vocab_size": len(vocab),
                "num_classes": num_classes,
                "pad_id": pad_id,
                "max_len": args.max_len,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "d_ff": args.d_ff,
                "dropout": args.dropout,
                "attn_dropout": args.attn_dropout,
                "use_attention_pooling": args.use_attention_pooling,
            }

            save_artifacts(
                save_dir=os.path.join(args.save_dir, "artifacts"),
                model=model,
                vocab=vocab,
                label2id=label2id,
                config=config,
            )

            print("  saved artifacts ->", os.path.join(args.save_dir, "artifacts"))

        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    # load best and test
    ckpt_path = os.path.join(args.save_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    te_acc, te_f1, y_true, y_pred = evaluate(model, test_loader, device)
    print("\n==== TEST RESULT ====")
    print(f"test acc: {te_acc:.4f} | test macro-F1: {te_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    out = {
        "test_acc": float(te_acc),
        "test_macro_f1": float(te_f1),
        "confusion_matrix": cm.tolist(),
        "label_names": label_names,
    }
    with open(os.path.join(args.save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved metrics ->", os.path.join(args.save_dir, "test_metrics.json"))


if __name__ == "__main__":
    main()
    