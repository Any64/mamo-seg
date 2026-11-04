"""Figuras del proyecto (todas guardan en disco si se entrega ``path``)."""
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _finish(fig, path):
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def show_samples(X, Y, P=None, idx=None, n=12, ncols=4, seed=0, path=None, titles=None):
    """Recortes con el contorno real (verde) y, si se entrega, el predicho (rojo)."""
    rng = np.random.default_rng(seed)
    if idx is None:
        idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), idx):
        ax.imshow(X[i], cmap="gray")
        if Y[i].any():
            ax.contour(Y[i], levels=[0.5], colors="lime", linewidths=1.2)
        if P is not None and P[i].any():
            ax.contour(P[i].astype(float), levels=[0.5], colors="red", linewidths=1.2)
        ax.set_title(titles[i] if titles is not None else f"#{i}", fontsize=9)
    return _finish(fig, path)


def plot_history(hist, path=None):
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(hist["epoch"], hist["train_loss"], color="tab:blue", label="pérdida (train)")
    ax1.set_xlabel("época")
    ax1.set_ylabel("pérdida BCE+Dice", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(hist["epoch"], hist["val_dice"], color="tab:orange", label="Dice (val)")
    ax2.set_ylabel("Dice validación", color="tab:orange")
    ax1.set_title("Curvas de entrenamiento de la U-Net")
    return _finish(fig, path)


def plot_confusion(cm, row_labels, col_labels, title="", path=None):
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("forma predicha")
    ax.set_ylabel("forma anotada (radiólogo)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fracción por fila")
    # título de figura (no del eje) para que la barra de color no lo tape
    fig.suptitle("\n".join(textwrap.wrap(title, 48)), fontsize=10)
    return _finish(fig, path)


def plot_feature_boxplots(df, label_col, features, order, path=None):
    fig, axes = plt.subplots(2, int(np.ceil(len(features) / 2)),
                             figsize=(3.2 * int(np.ceil(len(features) / 2)), 6))
    for ax, f in zip(np.atleast_1d(axes).ravel(), features):
        data = [df.loc[df[label_col] == c, f].dropna().values for c in order]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(order) + 1), [c[:4] for c in order])
        ax.set_title(f, fontsize=10)
    return _finish(fig, path)


def plot_hist(values, xlabel, title="", path=None, bins=30):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(values, bins=bins, color="tab:blue", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frecuencia")
    ax.set_title(title)
    return _finish(fig, path)


def plot_grouped_bars(df, x_col, group_col, value_col, panel_col, x_order, group_order,
                      ref_lines=None, ylabel="", title="", path=None, ylim=(0, 0.6)):
    """Barras agrupadas (un panel por valor de ``panel_col``) con líneas de referencia
    opcionales ``ref_lines`` = {etiqueta: {panel: valor}}."""
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # paleta categórica validada (máx. 3 series)
    panels = list(dict.fromkeys(df[panel_col]))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), sharey=True, squeeze=False)
    w = 0.8 / len(group_order)
    for ax, panel in zip(axes[0], panels):
        d = df[df[panel_col] == panel]
        for j, g in enumerate(group_order):
            v = [d[(d[x_col] == x) & (d[group_col] == g)][value_col].iloc[0] for x in x_order]
            xs = np.arange(len(x_order)) + (j - (len(group_order) - 1) / 2) * w
            ax.bar(xs, v, width=w - 0.03, color=colors[j], label=g)
            for xi, vi in zip(xs, v):
                ax.text(xi, vi + 0.006, f"{vi:.2f}", ha="center", va="bottom", fontsize=7, color="#444")
        for (label, vals), ls in zip((ref_lines or {}).items(), ("--", ":")):
            ax.axhline(vals[panel], color="#666", lw=1, ls=ls, label=label)
        ax.set_xticks(range(len(x_order)), [str(x) for x in x_order])
        ax.set_title(str(panel), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel(ylabel)
    axes[0][0].set_ylim(*ylim)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _finish_saved(fig, path)


def plot_roc_panels(panels, title="", path=None):
    """Curvas ROC. ``panels`` = [(título, [(etiqueta, fpr, tpr, auc), ...]), ...] (máx. 3 curvas)."""
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.6), squeeze=False)
    for ax, (ptitle, curves) in zip(axes[0], panels):
        for color, (label, fpr, tpr, auc) in zip(colors, curves):
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC {auc:.2f})")
        ax.plot([0, 1], [0, 1], color="#888", lw=1, ls="--", label="azar (AUC 0.50)")
        ax.set_xlabel("1 − especificidad")
        ax.set_ylabel("sensibilidad")
        ax.set_title(ptitle, fontsize=10)
        ax.set_aspect("equal")
        ax.grid(color="#e5e5e5", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), fontsize=7.5, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _finish_saved(fig, path)


def _finish_saved(fig, path):
    """Como ``_finish`` pero sin volver a llamar a tight_layout (ya ajustado)."""
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
