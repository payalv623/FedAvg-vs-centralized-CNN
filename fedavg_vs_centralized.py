"""
fedavg_vs_centralized.py  --  Federated Averaging vs centralized CNN training
============================================================================

"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
# use a shared ../data folder if one exists, otherwise download Fashion-MNIST into ./data
DATA_DIR = HERE.parent / "data" if (HERE.parent / "data").exists() else HERE / "data"
RESULTS = HERE / "results"
FIG_DIR = RESULTS / "figures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Experimental configuration (fixed a priori; NOT tuned on the test set)
# ---------------------------------------------------------------------------
CFG = {
    "dataset": "Fashion-MNIST",
    "num_clients": 10,
    "budget_epochs": 30,            # passes over the training data, for every method
    "local_epochs": [1, 2, 5],      # E; FedAvg rounds R = budget_epochs / E
    "partitions": ["iid", "dir1.0", "dir0.3", "dir0.1", "patho2"],
    "seeds": [0, 1, 2],
    "batch_size": 64,
    "lr": 0.05,                     # plain SGD (no momentum) for both central and client training,
    "momentum": 0.0,                # so the only difference between methods is federation itself
    "weight_decay": 0.0,
    # all 10 clients participate in every round (no client sampling)
}


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
def load_fashion_mnist():
    """Return (x_train, y_train, x_test, y_test) as normalised GPU tensors."""
    from torchvision.datasets import FashionMNIST
    tr = FashionMNIST(DATA_DIR, train=True, download=True)
    te = FashionMNIST(DATA_DIR, train=False, download=True)
    # identical preprocessing for every experiment: scale to [0,1] then
    # standardise with the training-set mean/std
    xtr = tr.data.float().div(255.0).unsqueeze(1)
    xte = te.data.float().div(255.0).unsqueeze(1)
    mean, std = xtr.mean().item(), xtr.std().item()
    xtr, xte = (xtr - mean) / std, (xte - mean) / std
    return (xtr.to(DEVICE), tr.targets.to(DEVICE), xte.to(DEVICE), te.targets.to(DEVICE))


# ---------------------------------------------------------------------------
# 2. PARTITIONING THE TRAINING SET ACROSS CLIENTS
# ---------------------------------------------------------------------------
def partition(labels, kind, num_clients, seed, num_classes=10):
    """
    Split training indices into `num_clients` disjoint shards.
      iid      : uniform random split (equal sizes, same label mix).
      dirA     : for every class, draw client proportions p ~ Dirichlet(A * 1_K)
                 and split that class accordingly. Small A -> strong label skew
                 (Hsu et al., 2019). Re-drawn until each client has >= 50 samples.
      pathoC   : sort by label, cut into num_clients*C shards, give each client C
                 shards -> each client sees only ~C classes (McMahan et al., 2017).
    The union of all shards is always exactly the full training set, so the
    centralized baseline trains on "the union of the same training examples".
    """
    rng = np.random.default_rng(seed)
    y = labels.cpu().numpy()
    n = len(y)
    if kind == "iid":
        perm = rng.permutation(n)
        return [np.sort(s) for s in np.array_split(perm, num_clients)]
    if kind.startswith("dir"):
        alpha = float(kind[3:])
        while True:
            parts = [[] for _ in range(num_clients)]
            for c in range(num_classes):
                idx = rng.permutation(np.where(y == c)[0])
                p = rng.dirichlet(alpha * np.ones(num_clients))
                cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
                for k, chunk in enumerate(np.split(idx, cuts)):
                    parts[k].extend(chunk.tolist())
            if min(len(p) for p in parts) >= 50:
                return [np.sort(np.array(p)) for p in parts]
    if kind.startswith("patho"):
        per_client = int(kind[5:])
        order = np.argsort(y, kind="stable")
        shards = np.array_split(order, num_clients * per_client)
        # Each shard holds a single class. Re-draw the shard assignment until no
        # client receives two shards of the SAME class, so every client really has
        # `per_client` distinct classes.
        while True:
            shard_ids = rng.permutation(len(shards))
            parts = [np.sort(np.concatenate([shards[s] for s in shard_ids[k::num_clients]]))
                     for k in range(num_clients)]
            if all(len(np.unique(y[p])) == per_client for p in parts):
                return parts
    raise ValueError(kind)


def label_histogram(labels, parts, num_classes=10):
    """K x C matrix of per-client class counts (used for plots and the privacy probe)."""
    y = labels.cpu().numpy()
    return np.stack([np.bincount(y[p], minlength=num_classes) for p in parts])


# ---------------------------------------------------------------------------
# 3. MODEL: one fixed small CNN for all experiments
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """
    conv(1->32, 3x3) -> ReLU -> maxpool 2   : 28x28 -> 14x14
    conv(32->64, 3x3) -> ReLU -> maxpool 2  : 14x14 -> 7x7
    fc(64*7*7 -> 128) -> ReLU -> fc(128 -> 10)
    ~421k parameters. No BatchNorm on purpose: BN running statistics are not
    model *parameters* and averaging them across non-IID clients is a known
    confounder, which would blur the FedAvg-vs-central comparison.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 4. TRAINING / EVALUATION PRIMITIVES (shared by central and federated code)
# ---------------------------------------------------------------------------
def train_epochs(model, x, y, epochs, gen):
    """
    Plain mini-batch SGD for `epochs` passes over (x, y). The torch.Generator
    `gen` controls the shuffling so every run is reproducible. Returns the mean
    training loss of the last epoch.
    """
    opt = torch.optim.SGD(model.parameters(), lr=CFG["lr"], momentum=CFG["momentum"],
                          weight_decay=CFG["weight_decay"])
    model.train()
    bs, n = CFG["batch_size"], len(y)
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen, device="cpu").to(x.device)
        tot, cnt = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        last = tot / cnt
    return last


@torch.no_grad()
def evaluate(model, x, y, bs=2000):
    """Accuracy and mean cross-entropy loss on (x, y)."""
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, len(y), bs):
        out = model(x[i:i + bs])
        loss += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(y), loss / len(y)


def init_model(seed):
    """Same seed -> same initial weights for central and federated runs."""
    torch.manual_seed(seed)
    return SmallCNN().to(DEVICE)


# ---------------------------------------------------------------------------
# 5. CENTRALIZED BASELINE
# ---------------------------------------------------------------------------
def run_centralized(data, seed):
    xtr, ytr, xte, yte = data
    model = init_model(seed)
    gen = torch.Generator().manual_seed(seed)
    log, t0 = [], time.time()
    for ep in range(1, CFG["budget_epochs"] + 1):
        tr_loss = train_epochs(model, xtr, ytr, 1, gen)
        acc, te_loss = evaluate(model, xte, yte)
        log.append({"method": "centralized", "partition": "-", "local_epochs": "-", "seed": seed,
                    "round": ep, "epochs_equiv": ep, "train_loss": tr_loss, "test_acc": acc,
                    "test_loss": te_loss, "client_drift": 0.0, "wall_s": time.time() - t0})
    return log


# ---------------------------------------------------------------------------
# 6. FEDERATED AVERAGING
# ---------------------------------------------------------------------------
def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def fedavg_aggregate(global_model, client_states, weights):
    """
    Weighted parameter averaging:  w <- sum_k (n_k / n) * w_k.
    `weights` are the client sample counts n_k (normalised here).
    """
    w = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    w = w / w.sum()
    new_state = {}
    for name in client_states[0]:
        stacked = torch.stack([s[name].float() for s in client_states])     # K x ...
        shape = [-1] + [1] * (stacked.dim() - 1)
        new_state[name] = (w.view(shape) * stacked).sum(0)
    global_model.load_state_dict(new_state)


def run_fedavg(data, parts, local_epochs, seed, partition_name, privacy_log=None):
    """
    One FedAvg run with the full client set participating every round.
    Logs, per round: global test accuracy/loss, mean client training loss and
    'client drift' = mean L2 distance between client models and their average
    (a direct measure of how far heterogeneous clients pull apart).
    """
    xtr, ytr, xte, yte = data
    K = len(parts)
    rounds = CFG["budget_epochs"] // local_epochs
    global_model = init_model(seed)
    sizes = [len(p) for p in parts]
    client_x = [xtr[torch.as_tensor(p, device=DEVICE)] for p in parts]
    client_y = [ytr[torch.as_tensor(p, device=DEVICE)] for p in parts]
    # one RNG per client for reproducible local shuffling
    gens = [torch.Generator().manual_seed(seed * 1000 + k) for k in range(K)]
    log, t0 = [], time.time()
    for r in range(1, rounds + 1):
        g_flat = flat(global_model)
        states, losses, flats = [], [], []
        for k in range(K):
            local = copy.deepcopy(global_model)          # client starts from the global model
            losses.append(train_epochs(local, client_x[k], client_y[k], local_epochs, gens[k]))
            states.append(copy.deepcopy(local.state_dict()))
            flats.append(flat(local))
            # ---- privacy probe: the server only sees the update (w_k - w_global)
            if privacy_log is not None and r == 1:
                privacy_log.append(label_inference(global_model, local, k))
        fedavg_aggregate(global_model, states, sizes)
        avg_flat = flat(global_model)
        drift = float(np.mean([(f - avg_flat).norm().item() for f in flats]))
        acc, te_loss = evaluate(global_model, xte, yte)
        log.append({"method": "fedavg", "partition": partition_name, "local_epochs": local_epochs,
                    "seed": seed, "round": r, "epochs_equiv": r * local_epochs,
                    "train_loss": float(np.average(losses, weights=sizes)), "test_acc": acc,
                    "test_loss": te_loss, "client_drift": drift,
                    "update_norm": float(np.mean([(f - g_flat).norm().item() for f in flats])),
                    "wall_s": time.time() - t0})
    return log, global_model


# ---------------------------------------------------------------------------
# 7. PRIVACY PROBE: label-distribution inference from a model update
# ---------------------------------------------------------------------------
def label_inference(global_model, local_model, client_id):
    """
    Honest-but-curious server attack. For softmax + cross-entropy, the gradient
    of the loss w.r.t. the output-layer bias b_c is (p_c - y_c), averaged over
    the batch. Classes PRESENT in a client's data therefore push b_c UP (negative
    gradient, y_c = 1 for those samples) and absent classes push b_c DOWN.
    So the server can read the client's label distribution from
        delta_b = b_local - b_global
    without ever seeing a single image. The server ranks the classes by delta_b
    (larger = more likely present / more frequent); score_label_inference compares
    that ranking with the true class counts.
    """
    db = (local_model.fc2.bias - global_model.fc2.bias).detach().cpu().numpy()
    return {"client": client_id, "delta_bias": db.tolist()}


def score_label_inference(priv_rows, hist):
    """
    Compare inferred vs true label distributions for each client:
      - Spearman rank correlation between delta_b and true class counts
      - top-k precision: of the k classes with the largest delta_b, how many are
        among the client's k most frequent classes (k = number of classes with
        >= 5% of the client's data)
      - presence AUC: can delta_b separate classes the client has (>= 1%) from
        classes it (almost) does not have?
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    out = []
    for row in priv_rows:
        k = row["client"]
        db = np.array(row["delta_bias"])
        counts = hist[k]
        share = counts / counts.sum()
        present = share >= 0.01
        kk = max(1, int((share >= 0.05).sum()))
        top_pred = set(np.argsort(-db)[:kk])
        top_true = set(np.argsort(-counts)[:kk])
        rho = spearmanr(db, counts).correlation
        auc = roc_auc_score(present, db) if 0 < present.sum() < len(present) else np.nan
        # if all classes are "top" classes (IID), top-k precision is trivially 1 -> report N/A
        topk = len(top_pred & top_true) / kk if kk < len(counts) else np.nan
        out.append({"client": k, "spearman": rho, "topk_precision": topk,
                    "k": kk, "presence_auc": auc, "n_present_classes": int(present.sum())})
    return out


# ---------------------------------------------------------------------------
# 8. EXPERIMENT DRIVER
# ---------------------------------------------------------------------------
def run_all(quick=False):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True     # reproducible convolutions
    global RESULTS, FIG_DIR
    cfg = dict(CFG)
    if quick:                                      # smoke-test settings, separate output folder
        CFG.update({"budget_epochs": 2, "local_epochs": [1, 2], "partitions": ["iid", "dir0.1"],
                    "seeds": [0]})
        RESULTS, FIG_DIR = HERE / "results_quick", HERE / "results_quick" / "figures"
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = load_fashion_mnist()
    ytr = data[1]
    meta = {"config": CFG, "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
            "torch": torch.__version__, "n_params": n_params(SmallCNN()),
            "train_size": int(len(ytr)), "test_size": int(len(data[3])),
            "bytes_per_model": n_params(SmallCNN()) * 4}
    (RESULTS / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    # Resume support: runs already saved in training_logs.csv are skipped, so a
    # crash (we saw rare segfaults when the GPU is shared) loses at most one run.
    log_path, priv_path = RESULTS / "training_logs.csv", RESULTS / "privacy_label_inference.csv"
    logs = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    priv_scores = pd.read_csv(priv_path).to_dict("records") if priv_path.exists() else []
    done = {(r["method"], str(r["partition"]), str(r["local_epochs"]), int(r["seed"])) for r in logs}
    hists = {}
    # --- centralized baseline
    for seed in CFG["seeds"]:
        if ("centralized", "-", "-", seed) in done:
            continue
        t = time.time()
        logs += run_centralized(data, seed)
        print(f"central seed={seed}: acc={logs[-1]['test_acc']:.4f} ({time.time() - t:.0f}s)", flush=True)
    # --- federated runs
    for part in CFG["partitions"]:
        for seed in CFG["seeds"]:
            parts = partition(ytr, part, CFG["num_clients"], seed)
            hist = label_histogram(ytr, parts)
            hists[f"{part}|{seed}"] = hist.tolist()
            for E in CFG["local_epochs"]:
                if ("fedavg", part, str(E), seed) in done:
                    continue
                t = time.time()
                log, _ = run_fedavg(data, parts, E, seed, part)
                logs += log
                print(f"fedavg {part} E={E} seed={seed}: acc={log[-1]['test_acc']:.4f} "
                      f"drift={log[-1]['client_drift']:.3f} ({time.time() - t:.0f}s)", flush=True)
                pd.DataFrame(logs).to_csv(log_path, index=False)              # checkpoint
    # --- privacy probe: the server observes the round-1 updates of a fresh
    # FedAvg run (E=1) and tries to read each client's label distribution.
    probed = {(r["partition"], int(r["seed"])) for r in priv_scores}
    for part in CFG["partitions"]:
        for seed in CFG["seeds"]:
            if (part, seed) in probed:
                continue
            parts = partition(ytr, part, CFG["num_clients"], seed)
            priv = []
            CFG["budget_epochs"], saved = 1, CFG["budget_epochs"]     # one round is enough
            try:
                run_fedavg(data, parts, 1, seed, part, privacy_log=priv)
            finally:
                CFG["budget_epochs"] = saved
            for sc in score_label_inference(priv, label_histogram(ytr, parts)):
                priv_scores.append({"partition": part, "seed": seed, **sc})
            pd.DataFrame(priv_scores).to_csv(priv_path, index=False)
    pd.DataFrame(logs).to_csv(log_path, index=False)
    (RESULTS / "client_label_histograms.json").write_text(json.dumps(hists))
    summarize()
    if quick:
        CFG.clear()
        CFG.update(cfg)


# ---------------------------------------------------------------------------
# 9. SUMMARY TABLES
# ---------------------------------------------------------------------------
def summarize():
    logs = pd.read_csv(RESULTS / "training_logs.csv")
    final = logs.loc[logs.groupby(["method", "partition", "local_epochs", "seed"]).round.idxmax()]
    central_acc = final[final.method == "centralized"].test_acc.mean()

    # Convergence speed: data passes needed to first reach 85% test accuracy.
    # Runs that never reach it are NOT silently dropped: we report how many seeds
    # reached it, and the mean over those seeds only (clearly labelled).
    def first_reach(g, thr=0.85):
        hit = g[g.test_acc >= thr]
        return hit.epochs_equiv.min() if len(hit) else np.nan
    reach = (logs.groupby(["method", "partition", "local_epochs", "seed"])
             .apply(first_reach, include_groups=False).rename("epochs_to_85pct").reset_index())
    final = final.merge(reach, on=["method", "partition", "local_epochs", "seed"])
    # Secondary, smoother number: mean test accuracy over evaluations in the final
    # 5 data passes. NOTE: the number of evaluations in that window differs with E
    # (5 for E=1 or centralized, 3 for E=2, 1 for E=5), so the PRIMARY comparison
    # uses the final accuracy at exactly 30 passes, which is defined identically
    # for every method.
    late = logs[logs.epochs_equiv > CFG["budget_epochs"] - 5]
    late = (late.groupby(["method", "partition", "local_epochs", "seed"]).test_acc.mean()
            .rename("acc_last5").reset_index())
    final = final.merge(late, on=["method", "partition", "local_epochs", "seed"])
    final["drift_per_local_epoch"] = final.client_drift / pd.to_numeric(final.local_epochs, errors="coerce")
    summ = (final.groupby(["method", "partition", "local_epochs"])
            .agg(test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
                 acc_last5_mean=("acc_last5", "mean"), acc_last5_std=("acc_last5", "std"),
                 test_loss_mean=("test_loss", "mean"), final_drift=("client_drift", "mean"),
                 drift_per_local_epoch=("drift_per_local_epoch", "mean"),
                 seeds_reaching_85pct=("epochs_to_85pct", lambda v: int(v.notna().sum())),
                 epochs_to_85pct_reached_only=("epochs_to_85pct", "mean"), n_seeds=("seed", "nunique"),
                 wall_s=("wall_s", "mean"))
            .reset_index())
    summ["gap_to_central_pp"] = (100 * (central_acc - summ.test_acc_mean)).round(6)
    central_last5 = final[final.method == "centralized"].acc_last5.mean()
    summ["gap_last5_pp"] = (100 * (central_last5 - summ.acc_last5_mean)).round(6)
    # communication: each round, K clients download + upload one model
    meta = json.loads((RESULTS / "run_metadata.json").read_text())
    rounds = np.where(summ.method == "fedavg",
                      CFG["budget_epochs"] / pd.to_numeric(summ.local_epochs, errors="coerce"), 0)
    summ["rounds"] = rounds
    summ["comm_MB"] = rounds * 2 * CFG["num_clients"] * meta["bytes_per_model"] / 1e6
    # sequential optimisation steps (see the equal-budget caveat in the module docstring)
    steps_epoch_central = math.ceil(meta["train_size"] / CFG["batch_size"])
    steps_epoch_client = math.ceil(meta["train_size"] / CFG["num_clients"] / CFG["batch_size"])
    summ["sequential_sgd_steps"] = np.where(summ.method == "fedavg",
                                            CFG["budget_epochs"] * steps_epoch_client,
                                            CFG["budget_epochs"] * steps_epoch_central)
    summ.to_csv(RESULTS / "summary.csv", index=False)
    print(summ.round(4).to_string(index=False))
    if (RESULTS / "privacy_label_inference.csv").exists():
        pr = pd.read_csv(RESULTS / "privacy_label_inference.csv")
        if len(pr):
            ps = pr.groupby("partition")[["spearman", "topk_precision", "presence_auc"]].agg(["mean", "std"])
            ps.to_csv(RESULTS / "privacy_summary.csv")
            print("\nLabel-inference attack (server reads client label mix from one update):")
            print(ps.round(3).to_string())


# ---------------------------------------------------------------------------
# 10. FIGURES
# ---------------------------------------------------------------------------
# Colour-blind-checked categorical slots, assigned in a fixed order so a given
# partition always has the same colour; centralized baseline is neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PART_LABEL = {"iid": "IID", "dir1.0": "Dirichlet α=1.0", "dir0.3": "Dirichlet α=0.3",
              "dir0.1": "Dirichlet α=0.1", "patho2": "Pathological (2 classes/client)"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # larger default text so figures stay readable when scaled to A4 width
    plt.rcParams.update({"font.size": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.labelsize": 12, "legend.fontsize": 11})
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    logs = pd.read_csv(RESULTS / "training_logs.csv")
    summ = pd.read_csv(RESULTS / "summary.csv")
    parts = [p for p in CFG["partitions"] if p in set(logs.partition)]
    Es = sorted(int(e) for e in logs[logs.method == "fedavg"].local_epochs.unique())
    cen = logs[logs.method == "centralized"].groupby("epochs_equiv").test_acc.agg(["mean", "std"])

    # Fig 1: test accuracy vs training budget, one panel per local-epoch setting
    fig, axes = plt.subplots(1, len(Es), figsize=(4.4 * len(Es), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, E in zip(axes, Es):
        ax.plot(cen.index, cen["mean"], color=INK, lw=2, label="Centralized")
        ax.fill_between(cen.index, cen["mean"] - cen["std"], cen["mean"] + cen["std"], color=INK, alpha=0.12)
        for j, p in enumerate(parts):
            d = logs[(logs.method == "fedavg") & (logs.partition == p) & (logs.local_epochs.astype(str) == str(E))]
            g = d.groupby("epochs_equiv").test_acc.agg(["mean", "std"])
            ax.plot(g.index, g["mean"], color=SERIES[j], lw=2, marker="o" if E > 1 else None, ms=4,
                    label=PART_LABEL[p])
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=SERIES[j], alpha=0.15)
        ax.set_title(f"E = {E} local epoch(s), {CFG['budget_epochs'] // E} rounds",
                     fontsize=14, color=INK)
        ax.set_xlabel("budget used (data passes)", fontsize=12.8, color=INK2)
        _style(ax)
    axes[0].set_ylabel("test accuracy", fontsize=12.8, color=INK2)
    axes[0].set_ylim(max(0, logs.test_acc.min() - 0.02), 0.95)
    axes[-1].legend(fontsize=11.2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_accuracy_curves.png", dpi=200)
    plt.close(fig)

    # Fig 2: final accuracy gap to centralized, by heterogeneity and E
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(Es)
    fs = summ[summ.method == "fedavg"].copy()
    fs["local_epochs"] = fs.local_epochs.astype(int)
    for k, E in enumerate(Es):
        d = fs[fs.local_epochs == E].set_index("partition").reindex(parts)
        x = np.arange(len(parts)) + (k - (len(Es) - 1) / 2) * width
        ax.bar(x, d.gap_to_central_pp, width=width * 0.92, color=SERIES[k], label=f"E = {E}",
               yerr=100 * d.test_acc_std, error_kw={"elinewidth": 1, "ecolor": INK2})
    ax.set_xticks(range(len(parts)), [PART_LABEL[p].replace(" (", "\n(") for p in parts], fontsize=12.8)
    ax.set_ylabel("gap to centralized (pp)", fontsize=12.8, color=INK2)
    ax.set_title("Test-accuracy gap to centralized after the full 30-pass budget\n(final round, ± sd over 3 seeds; lower is better)",
                 fontsize=14, color=INK)
    ax.legend(fontsize=12.8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_gap_to_centralized.png", dpi=200)
    plt.close(fig)

    # Fig 3: client drift per round (E=1 and largest E)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, E in zip(axes, [Es[0], Es[-1]]):
        for j, p in enumerate(parts):
            d = logs[(logs.method == "fedavg") & (logs.partition == p) & (logs.local_epochs.astype(str) == str(E))]
            g = d.groupby("round").client_drift.mean()
            ax.plot(g.index, g.values, color=SERIES[j], lw=2, label=PART_LABEL[p])
        ax.set_title(f"Client drift, E = {E}", fontsize=14, color=INK)
        ax.set_xlabel("communication round", fontsize=12.8, color=INK2)
        _style(ax)
    axes[0].set_ylabel("mean ||w_k − w_avg||₂", fontsize=12.8, color=INK2)
    axes[1].legend(fontsize=11.2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_client_drift.png", dpi=200)
    plt.close(fig)

    # Fig 4: client label distributions (seed 0) for each partition
    hists = json.loads((RESULTS / "client_label_histograms.json").read_text())
    fig, axes = plt.subplots(1, len(parts), figsize=(2.9 * len(parts), 3.6))
    for ax, p in zip(np.atleast_1d(axes), parts):
        h = np.array(hists[f"{p}|{CFG['seeds'][0]}"], float)
        ax.imshow(h / h.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(PART_LABEL[p].replace(" (", "\n("), fontsize=12.8, color=INK)
        ax.set_xlabel("class", fontsize=11.2, color=INK2)
        ax.set_xticks(range(10))
        ax.set_yticks(range(h.shape[0]))
        ax.tick_params(labelsize=6, colors=INK2)
    np.atleast_1d(axes)[0].set_ylabel("client", fontsize=11.2, color=INK2)
    fig.suptitle("Share of each class in each client's data (seed 0)", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_client_label_distributions.png", dpi=200)
    plt.close(fig)

    # Fig 5: privacy probe — inferred vs true label mix
    pr_path = RESULTS / "privacy_label_inference.csv"
    if pr_path.exists():
        pr = pd.read_csv(pr_path)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        mets = [("spearman", "rank corr. (Spearman)"), ("topk_precision", "top-k class precision"),
                ("presence_auc", "class-presence AUC")]
        width = 0.8 / len(mets)
        for k, (m, lab) in enumerate(mets):
            g = pr.groupby("partition")[m].agg(["mean", "std"]).reindex(parts)
            x = np.arange(len(parts)) + (k - (len(mets) - 1) / 2) * width
            ax.bar(x, g["mean"], width=width * 0.92, color=SERIES[k], label=lab,
                   yerr=g["std"], error_kw={"elinewidth": 1, "ecolor": INK2})
        ax.set_xticks(range(len(parts)), [PART_LABEL[p].replace(" (", "\n(") for p in parts], fontsize=12.8)
        ax.set_ylim(-0.25, 1.3)
        ax.set_ylabel("attack success (1 = perfect)", fontsize=12.8, color=INK2)
        ax.set_title("Server infers each client's label mix from ONE model update", fontsize=14, color=INK)
        ax.legend(fontsize=11.2, frameon=False, loc="upper center", ncol=3)
        ax.text(0, 0.45, "uniform mix:\nnothing to infer", ha="center", fontsize=11.2, color=INK2)
        _style(ax)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig5_privacy_label_inference.png", dpi=200)
        plt.close(fig)
    print(f"figures -> {FIG_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "summarize", "plot"])
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    args = ap.parse_args()
    if args.stage == "run":
        run_all(quick=args.quick)
        plot()
    elif args.stage == "summarize":
        summarize()
    else:
        plot()


if __name__ == "__main__":
    main()
