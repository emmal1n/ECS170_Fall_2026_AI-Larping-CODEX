"""Stage 4 report runs: IMDB sentiment classification and joke generation.

Trains a plain RNN, an LSTM, and a GRU for each task, plus two architecture
ablations. Writes metrics, loss curves, and generation examples under result/stage_4_result/.
"""

import csv
import json
import os
import random
import re
import time
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

SEED = 2
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.environ.get(
    "STAGE4_DATA", os.path.join(REPO_ROOT, "data", "stage_4_data")
)
OUT_DIR = os.path.join(REPO_ROOT, "result", "stage_4_result")
MAX_LEN = 120
VOCAB_SIZE = 12000
EMBED_DIM = 64
BATCH = 128
LR = 1e-3
CLS_EPOCHS = 8
GEN_EPOCHS = 30

# Negation words stay. Sentiment flips on "not" and "never".
STOP = set(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself just me more most my myself
    of off on once only or other our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what
    when where which while who whom why will with you your yours yourself
    yourselves""".split()
)

# Examples shown in the report skip jokes that contain these words.
DISPLAY_BLOCK = {
    "slave", "slaves", "rape", "rapist", "nazi", "porn", "suicide",
    "murder", "sex", "sexual", "fuck", "fucking", "shit", "bitch",
    "nigger", "faggot", "cunt",
}

COLORS = {
    "RNN": "#1d4e89",
    "LSTM": "#e07a1f",
    "GRU": "#0f766e",
    "LSTM hidden 64": "#9a3412",
    "LSTM 2 layers": "#7c3aed",
    "GRU 2 layers": "#0369a1",
}


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)


def clean_review(text):
    text = text.replace("<br />", " ").replace("<br/>", " ")
    text = text.lower()
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return [t for t in text.split() if len(t) > 1 and t not in STOP]


def clean_joke(text):
    text = text.lower().replace("\n", " ")
    return re.findall(r"[a-z0-9']+|[.!?]", text)


def load_reviews():
    root = os.path.join(DATA_ROOT, "text_classification")
    data = {}
    for split in ("train", "test"):
        xs, ys = [], []
        for label, y in (("neg", 0), ("pos", 1)):
            folder = os.path.join(root, split, label)
            for name in os.listdir(folder):
                if not name.endswith(".txt"):
                    continue
                with open(os.path.join(folder, name), encoding="utf-8", errors="replace") as f:
                    tokens = clean_review(f.read())
                if not tokens:
                    tokens = ["unk"]
                xs.append(tokens[:MAX_LEN])
                ys.append(y)
        data[split] = (xs, ys)
    return data


def load_jokes():
    path = os.path.join(DATA_ROOT, "text_generation", "data")
    jokes = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            tokens = clean_joke(row["Joke"])
            if len(tokens) >= 4:
                jokes.append(tokens[:40])
    return jokes


class Vocab:
    def __init__(self, token_lists, max_size, specials):
        counts = Counter(t for seq in token_lists for t in seq)
        self.pad, self.unk = "<pad>", "<unk>"
        words = [self.pad, self.unk] + [s for s in specials if s not in (self.pad, self.unk)]
        most = [w for w, _ in counts.most_common(max_size) if w not in words]
        words.extend(most)
        self.stoi = {w: i for i, w in enumerate(words)}
        self.itos = words

    def encode(self, tokens):
        unk = self.stoi[self.unk]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids):
        out = []
        for i in ids:
            if i <= 0:
                continue
            w = self.itos[i] if i < len(self.itos) else self.unk
            if w in ("<unk>", "<eos>"):
                if w == "<eos>":
                    break
                continue
            out.append(w)
        return detok(out)

    def __len__(self):
        return len(self.itos)


def detok(words):
    text = ""
    for w in words:
        if w in ".!?":
            text += w
        elif text:
            text += " " + w
        else:
            text = w
    return text


class ReviewSet(Dataset):
    def __init__(self, encoded, labels):
        self.x = encoded
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class JokeSet(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_cls(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([max(len(x), 1) for x in xs], dtype=torch.long)
    width = int(lengths.max())
    padded = torch.zeros(len(xs), width, dtype=torch.long)
    for i, x in enumerate(xs):
        if x:
            padded[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        else:
            padded[i, 0] = 1
    return padded, lengths, torch.tensor(ys, dtype=torch.long)


def collate_gen(batch):
    inputs = [s[:-1] if len(s) > 1 else s for s in batch]
    targets = [s[1:] if len(s) > 1 else s for s in batch]
    lengths = torch.tensor([len(x) for x in inputs], dtype=torch.long)
    width = int(lengths.max())
    xpad = torch.zeros(len(batch), width, dtype=torch.long)
    ypad = torch.zeros(len(batch), width, dtype=torch.long)
    for i, (x, y) in enumerate(zip(inputs, targets)):
        xpad[i, : len(x)] = torch.tensor(x)
        ypad[i, : len(y)] = torch.tensor(y[:width])
    return xpad, lengths, ypad


class Classifier(nn.Module):
    def __init__(self, n_vocab, cell, hidden, layers, pad_idx):
        super().__init__()
        self.cell_name = cell
        self.emb = nn.Embedding(n_vocab, EMBED_DIM, padding_idx=pad_idx)
        rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(EMBED_DIM, hidden, layers, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            self.drop(self.emb(x)), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        if self.cell_name == "lstm":
            h = h[0]
        return self.fc(h[-1])


class Generator(nn.Module):
    def __init__(self, n_vocab, cell, hidden, layers, pad_idx):
        super().__init__()
        self.cell_name = cell
        self.emb = nn.Embedding(n_vocab, EMBED_DIM, padding_idx=pad_idx)
        rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(EMBED_DIM, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, n_vocab)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.fc(out)


def run_classifier(model, loader, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()
    total, correct, n = 0.0, 0, 0
    preds, trues = [], []
    for x, lengths, y in loader:
        if train:
            optimizer.zero_grad()
        logits = model(x, lengths)
        loss = loss_fn(logits, y)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        total += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        n += y.size(0)
        if not train:
            preds.append(pred.detach())
            trues.append(y)
    row = {"loss": total / n, "accuracy": correct / n}
    if preds:
        p = torch.cat(preds).numpy()
        t = torch.cat(trues).numpy()
        pr, rc, f1, _ = precision_recall_fscore_support(t, p, average="macro", zero_division=0)
        row.update({"precision": float(pr), "recall": float(rc), "f1": float(f1),
                    "accuracy": float(accuracy_score(t, p))})
    return row


def run_generator(model, loader, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total, correct, n = 0.0, 0, 0
    for x, lengths, y in loader:
        if train:
            optimizer.zero_grad()
        logits = model(x, lengths)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        mask = y != 0
        total += loss.item() * mask.sum().item()
        correct += ((logits.argmax(-1) == y) & mask).sum().item()
        n += mask.sum().item()
    return {"loss": total / max(n, 1), "token_accuracy": correct / max(n, 1)}


def generate(model, vocab, prompt, max_new=24):
    model.eval()
    ids = vocab.encode(clean_joke(prompt))
    eos = vocab.stoi.get("<eos>")
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids], dtype=torch.long)
            lengths = torch.tensor([len(ids)])
            logits = model(x, lengths)[0, -1]
            nxt = int(logits.argmax())
            ids.append(nxt)
            if nxt == eos:
                break
    return vocab.decode(ids)


def safe_text(text):
    words = set(re.findall(r"[a-z0-9']+", text.lower()))
    return BLOCK_FREE(words)


def BLOCK_FREE(words):
    return words.isdisjoint(DISPLAY_BLOCK)


def word_overlap(reference, generated):
    ref = clean_joke(reference)
    gen = clean_joke(generated)
    if not ref:
        return 0.0
    # Share of reference words that also appear in the generation, order ignored.
    bag = Counter(gen)
    hit = 0
    for w in ref:
        if bag[w]:
            hit += 1
            bag[w] -= 1
    return hit / len(ref)


def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.28)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(frameon=False)


def plot_curves(histories, key, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for name, hist in histories:
        epochs = list(range(1, len(hist) + 1))
        mark = 1 if len(epochs) <= 20 else 5
        ax.plot(epochs, [h[key] for h in hist], marker="o", linewidth=2, markersize=5,
                markevery=mark, color=COLORS.get(name, "#334155"), label=name)
    style_ax(ax, title, "Epoch", ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ablation(rows, path):
    labels = [r["name"] for r in rows]
    acc = [r["test"]["accuracy"] for r in rows]
    f1 = [r["test"]["f1"] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - 0.18, acc, 0.36, label="Accuracy", color="#1d4e89")
    ax.bar(x + 0.18, f1, 0.36, label="Macro F1", color="#e07a1f")
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax.text(i - 0.18, a + 0.004, f"{a:.3f}", ha="center", va="bottom", fontsize=8, color="#1d4e89")
        ax.text(i + 0.18, f + 0.004, f"{f:.3f}", ha="center", va="bottom", fontsize=8, color="#9a3412")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.70, 0.92)
    style_ax(ax, "Classification ablation on the IMDB test set", "Model", "Score")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_architecture(path, kind):
    fig, ax = plt.subplots(figsize=(10.2, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    if kind == "classifier":
        boxes = [
            (0.2, "Review\ntokens"),
            (2.2, "Embedding\n12k x 64"),
            (4.2, "RNN, LSTM,\nor GRU"),
            (6.2, "Last hidden\nstate"),
            (8.2, "Linear\npos / neg"),
        ]
        title = "Sentiment classifier"
    else:
        boxes = [
            (0.2, "Three starting\nwords"),
            (2.2, "Embedding\n64-d"),
            (4.2, "RNN, LSTM,\nor GRU"),
            (6.2, "Next-word\nsoftmax"),
            (8.2, "Append word\nand repeat"),
        ]
        title = "Joke generator"
    for x, text in boxes:
        ax.add_patch(plt.Rectangle((x, 0.7), 1.7, 1.25, facecolor="#f8fafc",
                                   edgecolor="#1e293b", linewidth=1.2))
        ax.text(x + 0.85, 1.32, text, ha="center", va="center", fontsize=9)
    for x, _ in boxes[:-1]:
        ax.annotate("", xy=(x + 2.05, 1.32), xytext=(x + 1.75, 1.32),
                    arrowprops=dict(arrowstyle="->", color="#334155", lw=1.2))
    ax.set_title(title, loc="left", fontsize=12, pad=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_classifiers(train_loader, test_loader, n_vocab, pad_idx):
    configs = [
        {"name": "RNN", "cell": "rnn", "hidden": 128, "layers": 1},
        {"name": "LSTM", "cell": "lstm", "hidden": 128, "layers": 1},
        {"name": "GRU", "cell": "gru", "hidden": 128, "layers": 1},
        {"name": "LSTM hidden 64", "cell": "lstm", "hidden": 64, "layers": 1},
        {"name": "LSTM 2 layers", "cell": "lstm", "hidden": 128, "layers": 2},
    ]
    runs = []
    for cfg in configs:
        set_seed()
        model = Classifier(n_vocab, cfg["cell"], cfg["hidden"], cfg["layers"], pad_idx)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        hist = []
        print(f"\n=== classifier {cfg['name']} params={count_params(model)} ===", flush=True)
        t0 = time.time()
        for epoch in range(1, CLS_EPOCHS + 1):
            tr = run_classifier(model, train_loader, opt)
            te = run_classifier(model, test_loader)
            hist.append({"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["accuracy"], **
                         {f"test_{k}": v for k, v in te.items()}})
            print(f"  epoch {epoch} train_loss={tr['loss']:.4f} train_acc={tr['accuracy']:.4f} "
                  f"test_acc={te['accuracy']:.4f} f1={te['f1']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        final = hist[-1]
        runs.append({
            "name": cfg["name"], "cell": cfg["cell"], "hidden": cfg["hidden"],
            "layers": cfg["layers"], "params": count_params(model), "history": hist,
            "test": {"accuracy": final["test_accuracy"], "precision": final["test_precision"],
                     "recall": final["test_recall"], "f1": final["test_f1"],
                     "loss": final["test_loss"]},
        })
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "models", f"cls_{cfg['cell']}_{cfg['hidden']}_{cfg['layers']}.pt"))
        save_partial(runs, "classification")
    return runs


def train_generators(loader, n_vocab, pad_idx):
    configs = [
        {"name": "RNN", "cell": "rnn", "hidden": 128, "layers": 1},
        {"name": "LSTM", "cell": "lstm", "hidden": 128, "layers": 1},
        {"name": "GRU", "cell": "gru", "hidden": 128, "layers": 1},
        {"name": "LSTM hidden 64", "cell": "lstm", "hidden": 64, "layers": 1},
        {"name": "GRU 2 layers", "cell": "gru", "hidden": 128, "layers": 2},
    ]
    runs, models = [], {}
    for cfg in configs:
        set_seed()
        model = Generator(n_vocab, cfg["cell"], cfg["hidden"], cfg["layers"], pad_idx)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        hist = []
        print(f"\n=== generator {cfg['name']} params={count_params(model)} ===", flush=True)
        t0 = time.time()
        for epoch in range(1, GEN_EPOCHS + 1):
            tr = run_generator(model, loader, opt)
            hist.append({"epoch": epoch, "train_loss": tr["loss"], "token_accuracy": tr["token_accuracy"]})
            if epoch == 1 or epoch % 5 == 0 or epoch == GEN_EPOCHS:
                print(f"  epoch {epoch} loss={tr['loss']:.4f} token_acc={tr['token_accuracy']:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        final = hist[-1]
        runs.append({
            "name": cfg["name"], "cell": cfg["cell"], "hidden": cfg["hidden"],
            "layers": cfg["layers"], "params": count_params(model), "history": hist,
            "train": {"loss": final["train_loss"], "token_accuracy": final["token_accuracy"]},
        })
        models[cfg["name"]] = model
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "models", f"gen_{cfg['cell']}_{cfg['hidden']}_{cfg['layers']}.pt"))
        save_partial(runs, "generation")
    return runs, models


def save_partial(runs, key):
    path = os.path.join(OUT_DIR, "partial.json")
    blob = {}
    if os.path.exists(path):
        with open(path) as f:
            blob = json.load(f)
    blob[key] = runs
    with open(path, "w") as f:
        json.dump(blob, f)


def build_examples(models, vocab, jokes):
    originals = []
    for tokens in jokes:
        text = detok(tokens)
        if safe_text(text):
            originals.append((tokens[:3], text))
    # Prefer the assignment's example prefix, then a spread of other openings.
    chosen = []
    wanted = [("what", "did", "the")]
    for prefix, text in originals:
        if tuple(prefix) == wanted[0]:
            chosen.append((prefix, text))
            break
    seen = {tuple(c[0]) for c in chosen}
    for prefix, text in originals:
        key = tuple(prefix)
        if key in seen:
            continue
        if len(text.split()) < 6 or len(text) > 140:
            continue
        chosen.append((prefix, text))
        seen.add(key)
        if len(chosen) >= 6:
            break
    free_prompts = ["why did the", "a horse walked"]
    main_names = ["RNN", "LSTM", "GRU"]
    rows = []
    for prefix, reference in chosen:
        prompt = " ".join(prefix)
        by_model = {name: generate(models[name], vocab, prompt) for name in main_names}
        rows.append({
            "prompt": prompt,
            "reference": reference,
            "in_training": True,
            "generations": by_model,
            "overlap": {name: round(word_overlap(reference, text), 3) for name, text in by_model.items()},
        })
    for prompt in free_prompts:
        by_model = {name: generate(models[name], vocab, prompt) for name in main_names}
        rows.append({
            "prompt": prompt,
            "reference": None,
            "in_training": False,
            "generations": by_model,
            "overlap": {},
        })
    return rows


def plot_examples(rows, path):
    shown = [r for r in rows if r["in_training"]][:3]
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "Joke completions from three-word prefixes", fontsize=14, color="#0f172a")
    top = 0.88
    height = 0.26
    for i, row in enumerate(shown):
        y = top - i * height
        ax.add_patch(plt.Rectangle((0.02, y - 0.22), 0.96, 0.24, facecolor="#f8fafc", edgecolor="#cbd5e1"))
        ax.text(0.05, y - 0.02, "Prefix: " + row["prompt"], fontsize=11, color="#1d4e89", fontweight="medium")
        ax.text(0.05, y - 0.08, "Training joke: " + row["reference"], fontsize=9, color="#334155", wrap=True)
        ax.text(0.05, y - 0.16, "LSTM: " + row["generations"]["LSTM"], fontsize=9, color="#0f172a", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def wrap(text, width):
    words, line, lines = text.split(), "", []
    for w in words:
        trial = (line + " " + w).strip()
        if len(trial) > width:
            lines.append(line)
            line = w
        else:
            line = trial
    if line:
        lines.append(line)
    return "\n".join(lines[:2])


def main():
    set_seed()
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)
    print("loading reviews...", flush=True)
    reviews = load_reviews()
    print("train", len(reviews["train"][1]), "test", len(reviews["test"][1]), flush=True)
    vocab = Vocab(reviews["train"][0], VOCAB_SIZE, [])
    train_x = [vocab.encode(s) for s in reviews["train"][0]]
    test_x = [vocab.encode(s) for s in reviews["test"][0]]
    train_loader = DataLoader(ReviewSet(train_x, reviews["train"][1]), batch_size=BATCH, shuffle=True, collate_fn=collate_cls)
    test_loader = DataLoader(ReviewSet(test_x, reviews["test"][1]), batch_size=BATCH, shuffle=False, collate_fn=collate_cls)

    cls_runs = train_classifiers(train_loader, test_loader, len(vocab), vocab.stoi["<pad>"])
    plot_curves(
        [(r["name"], [{"loss": h["train_loss"]} for h in r["history"]]) for r in cls_runs if r["layers"] == 1 and r["hidden"] == 128],
        "loss", "Training loss on IMDB reviews", "Cross-entropy loss",
        os.path.join(OUT_DIR, "figures", "classification_loss.png"),
    )
    plot_curves(
        [(r["name"], [{"accuracy": h["test_accuracy"]} for h in r["history"]]) for r in cls_runs if r["layers"] == 1 and r["hidden"] == 128],
        "accuracy", "Test accuracy on IMDB reviews", "Accuracy",
        os.path.join(OUT_DIR, "figures", "classification_accuracy.png"),
    )
    plot_ablation(cls_runs, os.path.join(OUT_DIR, "figures", "classification_ablation.png"))

    print("loading jokes...", flush=True)
    jokes = load_jokes()
    gen_vocab = Vocab(jokes, 8000, ["<eos>"])
    sequences = [gen_vocab.encode(s + ["<eos>"]) for s in jokes]
    gen_loader = DataLoader(JokeSet(sequences), batch_size=64, shuffle=True, collate_fn=collate_gen)
    gen_runs, gen_models = train_generators(gen_loader, len(gen_vocab), gen_vocab.stoi["<pad>"])
    plot_curves(
        [(r["name"], [{"loss": h["train_loss"]} for h in r["history"]]) for r in gen_runs if r["name"] in ("RNN", "LSTM", "GRU")],
        "loss", "Training loss on joke generation", "Next-word cross-entropy",
        os.path.join(OUT_DIR, "figures", "generation_loss.png"),
    )
    plot_architecture(os.path.join(OUT_DIR, "figures", "classifier_architecture.png"), "classifier")
    plot_architecture(os.path.join(OUT_DIR, "figures", "generator_architecture.png"), "generator")

    examples = build_examples(gen_models, gen_vocab, jokes)
    plot_examples(examples, os.path.join(OUT_DIR, "figures", "generation_examples.png"))

    lengths = [len(s) for s in reviews["train"][0]]
    payload = {
        "settings": {
            "seed": SEED,
            "max_tokens": MAX_LEN,
            "classification_vocab": len(vocab),
            "embedding_dim": EMBED_DIM,
            "batch_size": BATCH,
            "learning_rate": LR,
            "optimizer": "Adam",
            "classification_epochs": CLS_EPOCHS,
            "generation_epochs": GEN_EPOCHS,
            "generation_vocab": len(gen_vocab),
            "n_jokes": len(jokes),
            "train_reviews": len(reviews["train"][1]),
            "test_reviews": len(reviews["test"][1]),
            "mean_train_tokens_after_cleaning": round(float(np.mean(lengths)), 1),
            "embeddings": "learned with the model, not pretrained GloVe",
        },
        "classification": [{k: v for k, v in r.items()} for r in cls_runs],
        "generation": gen_runs,
        "examples": examples,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote", os.path.join(OUT_DIR, "metrics.json"), flush=True)


if __name__ == "__main__":
    main()
