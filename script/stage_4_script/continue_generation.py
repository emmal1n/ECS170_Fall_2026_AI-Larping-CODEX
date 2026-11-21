"""Keep training the joke generators and refresh the generation figures."""

import json
import os

import torch
from torch.utils.data import DataLoader

import train_stage4 as t

EXTRA_EPOCHS = 90
OUT = t.OUT_DIR


def main():
    t.set_seed()
    jokes = t.load_jokes()
    vocab = t.Vocab(jokes, 8000, ["<eos>"])
    sequences = [vocab.encode(s + ["<eos>"]) for s in jokes]
    loader = DataLoader(t.JokeSet(sequences), batch_size=64, shuffle=True, collate_fn=t.collate_gen)
    metrics_path = os.path.join(OUT, "metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)

    models = {}
    for run in metrics["generation"]:
        model = t.Generator(len(vocab), run["cell"], run["hidden"], run["layers"], vocab.stoi["<pad>"])
        path = os.path.join(OUT, "models", f"gen_{run['cell']}_{run['hidden']}_{run['layers']}.pt")
        model.load_state_dict(torch.load(path, map_location="cpu"))
        opt = torch.optim.Adam(model.parameters(), lr=t.LR)
        print(f"\n=== continue {run['name']} from epoch {len(run['history'])} ===", flush=True)
        for step in range(1, EXTRA_EPOCHS + 1):
            tr = t.run_generator(model, loader, opt)
            epoch = len(run["history"]) + 1
            run["history"].append({"epoch": epoch, "train_loss": tr["loss"], "token_accuracy": tr["token_accuracy"]})
            if step == 1 or step % 15 == 0 or step == EXTRA_EPOCHS:
                print(f"  epoch {epoch} loss={tr['loss']:.4f} token_acc={tr['token_accuracy']:.4f}", flush=True)
        final = run["history"][-1]
        run["train"] = {"loss": final["train_loss"], "token_accuracy": final["token_accuracy"]}
        torch.save(model.state_dict(), path)
        models[run["name"]] = model

    metrics["settings"]["generation_epochs"] = len(metrics["generation"][0]["history"])
    main_names = ["RNN", "LSTM", "GRU"]
    t.plot_curves(
        [(r["name"], [{"loss": h["train_loss"]} for h in r["history"]]) for r in metrics["generation"] if r["name"] in main_names],
        "loss", "Training loss on joke generation", "Next-word cross-entropy",
        os.path.join(OUT, "figures", "generation_loss.png"),
    )
    # Token accuracy comparison, including the two generation ablations.
    fig_rows = metrics["generation"]
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [r["name"] for r in fig_rows]
    vals = [r["train"]["token_accuracy"] for r in fig_rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=[t.COLORS.get(n, "#334155") for n in labels])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    t.style_ax(ax, "Joke generation token accuracy after training", "Model", "Training token accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "generation_ablation.png"), dpi=160)
    plt.close(fig)

    examples = pick_examples(models, vocab, jokes)
    metrics["examples"] = examples
    t.plot_examples(examples, os.path.join(OUT, "figures", "generation_examples.png"))
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("updated", metrics_path, flush=True)


def pick_examples(models, vocab, jokes):
    originals = []
    for tokens in jokes:
        text = t.detok(tokens)
        if t.safe_text(text) and len(tokens) >= 6:
            originals.append((tokens[:3], text))
    main_names = ["RNN", "LSTM", "GRU"]

    def row(prefix, reference, in_training):
        prompt = " ".join(prefix) if isinstance(prefix, (list, tuple)) else prefix
        by_model = {name: t.generate(models[name], vocab, prompt) for name in main_names}
        overlap = {}
        if reference:
            overlap = {name: round(t.word_overlap(reference, text), 3) for name, text in by_model.items()}
        return {"prompt": prompt, "reference": reference, "in_training": in_training,
                "generations": by_model, "overlap": overlap}

    rows = []
    for prefix, text in originals:
        if tuple(prefix) == ("what", "did", "the"):
            rows.append(row(prefix, text, True))
            break
    scored = []
    for prefix, text in originals:
        if tuple(prefix) == ("what", "did", "the"):
            continue
        gen = t.generate(models["GRU"], vocab, " ".join(prefix))
        scored.append((t.word_overlap(text, gen), prefix, text, gen))
    scored.sort(reverse=True)
    used = {rows[0]["prompt"]} if rows else set()
    for _, prefix, text, _ in scored:
        prompt = " ".join(prefix)
        if prompt in used or len(text) > 120:
            continue
        rows.append(row(prefix, text, True))
        used.add(prompt)
        if len([r for r in rows if r["in_training"]]) >= 4:
            break
    for prompt in ("a horse walked", "my professor said"):
        rows.append(row(prompt, None, False))
    return rows


if __name__ == "__main__":
    main()
