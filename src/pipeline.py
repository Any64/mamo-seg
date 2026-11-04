"""Pipeline completo del experimento principal.

Lo usan tanto ``scripts/run_experiment.py`` como el notebook de demostración, de
modo que la lógica existe en un solo lugar:

1. ``prepare_data``     recortes ROI (con caché) y división train/val por paciente
2. ``run_baselines``    prior centrado y método clásico (Otsu + morfología)
3. ``train_unet``       (src/train.py) entrenamiento con early stopping
4. ``evaluate_unet``    Dice/IoU/precisión/sensibilidad en val y test
5. ``shape_analysis``   forma BI-RADS desde la máscara (árbol de profundidad 3)
6. ``run_experiment``   todo lo anterior + guardado en ``outputs/runs/<name>/``
"""
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from . import shape as shp
from .classical import center_prior, classical_segment_stack, fit_center_prior
from .config import Config
from .data_index import load_case_csv
from .data_utils import patient_split
from .metrics import per_sample_metrics
from .preprocess import build_split, load_split, save_split

PRIOR, CLASSICAL, UNET = "Prior centrado", "Clásico (Otsu+morfología)", "U-Net"
EVAL_SPLITS = ("val", "test")


class Split(NamedTuple):
    X: np.ndarray      # recortes uint8 (N,H,W)
    Y: np.ndarray      # máscaras anotadas uint8 {0,1} (N,H,W)
    meta: pd.DataFrame


# --------------------------------------------------------------------------- datos
def ensure_split(cfg: Config, split: str, progress: bool = True):
    """Genera los recortes de ``split`` ('train' o 'test' oficial) si no están en caché.

    Devuelve el conteo de filas procesadas/descartadas, o None si ya existían.
    """
    if (cfg.processed_dir / f"{split}_{cfg.cache_tag}.npz").exists():
        return None
    if not cfg.csv_dir.is_dir():
        raise FileNotFoundError(
            f"No se encontró el dataset en {cfg.archive_dir} (se espera csv/ y jpeg/). "
            "Indica la ruta con la variable de entorno MAMO_DATA_DIR, con archive_dir en la "
            "config o con --data-dir. Ver docs/DATA.md.")
    df = load_case_csv(cfg.csv_dir, split)
    X, Y, meta, st = build_split(df, cfg.jpeg_dir, cfg.img_size, cfg.context, cfg.min_side,
                                 jitter=cfg.jitter, seed=cfg.seed, progress=progress,
                                 workers=cfg.workers)
    save_split(cfg.processed_dir, f"{split}_{cfg.cache_tag}", X, Y, meta)
    return dict(st)


def prepare_data(cfg: Config, progress: bool = True, include_test: bool = True):
    """Genera (o carga de la caché) los recortes y divide train/val por paciente.

    Devuelve ``(data, status)``: ``data`` es un dict {'train','val'[, 'test']} -> Split
    (test es el conjunto oficial de CBIS-DDSM; con ``include_test=False`` ni se
    genera ni se carga) y ``status`` el conteo de filas de los splits regenerados.
    """
    splits = ("train", "test") if include_test else ("train",)
    status = {s: st for s in splits if (st := ensure_split(cfg, s, progress)) is not None}

    Xall, Yall, mall = load_split(cfg.processed_dir, f"train_{cfg.cache_tag}")
    tr, va = patient_split(mall, cfg.val_frac, cfg.seed)
    data = {
        "train": Split(Xall[tr], Yall[tr], mall.iloc[tr].reset_index(drop=True)),
        "val": Split(Xall[va], Yall[va], mall.iloc[va].reset_index(drop=True)),
    }
    if include_test:
        Xte, Yte, mte = load_split(cfg.processed_dir, f"test_{cfg.cache_tag}")
        data["test"] = Split(Xte, Yte, mte.reset_index(drop=True))
    assert set(data["train"].meta["patient_id"]).isdisjoint(data["val"].meta["patient_id"])
    return data, status


# --------------------------------------------------------------------------- referencias
def run_baselines(data: dict, splits=EVAL_SPLITS) -> dict:
    """Métricas por muestra de los métodos de referencia: {(método, split): DataFrame}."""
    radius = fit_center_prior(data["train"].Y)
    results = {}
    for s in splits:
        X, Y = data[s].X, data[s].Y
        prior = np.broadcast_to(center_prior(X.shape[1:], radius), X.shape)
        results[(PRIOR, s)] = per_sample_metrics(prior, Y)
        results[(CLASSICAL, s)] = per_sample_metrics(classical_segment_stack(X), Y)
    return results


# --------------------------------------------------------------------------- U-Net
def predict_masks(model, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Máscaras binarias (N,H,W) predichas por la U-Net para recortes uint8."""
    from .dataset import to_input_tensor
    from .train import get_device, predict_probs
    return predict_probs(model, to_input_tensor(X, get_device())) > threshold


def load_unet(cfg: Config, ckpt_path):
    """U-Net con los pesos guardados en ``ckpt_path`` (en el dispositivo disponible)."""
    import torch
    from .model import UNet
    from .train import get_device
    dev = get_device()
    model = UNet(base=cfg.base_channels).to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    return model


def evaluate_unet(model, data: dict, cfg: Config, splits=EVAL_SPLITS):
    """Devuelve (predicciones {split: máscaras}, métricas {(U-Net, split): DataFrame})."""
    preds = {s: predict_masks(model, data[s].X, cfg.threshold) for s in splits}
    return preds, {(UNET, s): per_sample_metrics(preds[s], data[s].Y) for s in splits}


# --------------------------------------------------------------------------- tablas
def summary_table(results: dict) -> pd.DataFrame:
    """Tabla legible (media ± desviación) por método y conjunto."""
    rows = []
    for (m, s), d in results.items():
        rows.append({"método": m, "conjunto": s, "n": len(d),
                     "Dice": f"{d.dice.mean():.3f} ± {d.dice.std():.3f}",
                     "IoU": f"{d.iou.mean():.3f} ± {d.iou.std():.3f}",
                     "Precisión": f"{d.precision.mean():.3f}", "Sensibilidad": f"{d.recall.mean():.3f}"})
    return pd.DataFrame(rows)


def per_sample_table(results: dict, data: dict) -> pd.DataFrame:
    """Métricas por recorte de todos los métodos, con identificadores y desplazamiento."""
    cols = ["row_idx", "patient_id", "view", "pathology", "shape_label", "subtlety",
            "lesion_area_frac", "shift_x", "shift_y"]
    frames = []
    for (m, s), d in results.items():
        meta = data[s].meta[cols].reset_index(drop=True)
        frames.append(pd.concat([pd.DataFrame({"método": m, "conjunto": s}, index=d.index),
                                 meta, d], axis=1))
    out = pd.concat(frames, ignore_index=True)
    out["abs_shift"] = np.hypot(out["shift_x"], out["shift_y"])
    return out


def dice_by_shift(per_sample: pd.DataFrame, method: str = UNET, split: str = "test") -> pd.DataFrame:
    """Dice según el tercil de |desplazamiento| de la lesión y fracción de fallos (Dice < 0.1)."""
    d = per_sample[(per_sample["método"] == method) & (per_sample["conjunto"] == split)].copy()
    if d["abs_shift"].nunique() >= 3:
        d["tercil_shift"] = pd.qcut(d["abs_shift"], 3, labels=["bajo", "medio", "alto"])
    else:  # sin jitter todos los desplazamientos son 0: no hay terciles
        d["tercil_shift"] = "sin desplazamiento"
    return d.groupby("tercil_shift", observed=True).agg(
        n=("dice", "size"), shift_min=("abs_shift", "min"), shift_max=("abs_shift", "max"),
        dice_media=("dice", "mean"), dice_mediana=("dice", "median"),
        frac_dice_lt_0_1=("dice", lambda v: float((v < 0.1).mean())))


# --------------------------------------------------------------------------- forma
def shape_analysis(data: dict, preds: dict, cfg: Config, splits=EVAL_SPLITS):
    """Árbol de profundidad 3 ajustado con máscaras anotadas de train; evaluado con
    máscaras anotadas (cota superior) y predichas por la U-Net.

    Devuelve (resultados {(split, máscara): dict}, árbol, descriptores de train con la forma).
    """
    mtr = data["train"].meta
    ktr = (mtr["shape_label"] != "").values
    F_tr = shp.features_dataframe(data["train"].Y[ktr])
    clf = shp.fit_shape_tree(F_tr, mtr.loc[ktr, "shape_label"].values, seed=cfg.seed)
    res = {}
    for s in splits:
        k = (data[s].meta["shape_label"] != "").values
        y = data[s].meta.loc[k, "shape_label"].values
        for name, M in (("máscara anotada", data[s].Y), ("máscara U-Net", preds[s])):
            res[(s, name)] = shp.evaluate_shape(y, shp.predict_shape(clf, shp.features_dataframe(M[k])))
    return res, clf, F_tr.assign(forma=mtr.loc[ktr, "shape_label"].values)


def shape_table(shape_res: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"conjunto": s, "máscara usada": m, "n": r["n"], "sin descriptores": r["n_invalid"],
         "exactitud": round(r["accuracy"], 4), "exactitud balanceada": round(r["balanced_accuracy"], 4),
         "F1 macro": round(r["macro_f1"], 4)} for (s, m), r in shape_res.items()])


# --------------------------------------------------------------------------- experimento completo
def run_experiment(cfg: Config, verbose: bool = True, figures: bool = True,
                   evaluate_test: bool = True) -> dict:
    """Preprocesa (con caché), entrena, evalúa y guarda todo en ``cfg.run_dir``:

    config.yaml, metrics.json, per_sample.csv, history.csv, train.log, unet_best.pt,
    tablas CSV y figuras (``figures/``). Devuelve el diccionario de métricas.

    Con ``evaluate_test=False`` (ablaciones y selección de modelo) el test oficial
    no se genera, no se carga y no se evalúa: todo se reporta en validación.
    """
    import yaml
    from .train import train_unet

    run_dir = Path(cfg.run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)

    say = print if verbose else (lambda *a, **k: None)
    data, status = prepare_data(cfg, progress=verbose, include_test=evaluate_test)
    splits = EVAL_SPLITS if evaluate_test else ("val",)
    last = splits[-1]   # conjunto de las figuras y del análisis por desplazamiento
    say(" | ".join(f"{s}={len(data[s].X)}" for s in data))

    results = run_baselines(data, splits)
    model, hist = train_unet(cfg, data["train"].X, data["train"].Y, data["val"].X, data["val"].Y,
                             verbose=verbose, ckpt_path=run_dir / "unet_best.pt",
                             log_path=run_dir / "train.log")
    preds, unet_res = evaluate_unet(model, data, cfg, splits)
    results.update(unet_res)
    shape_res, clf, F_plot = shape_analysis(data, preds, cfg, splits)

    per_sample = per_sample_table(results, data)
    by_shift = dice_by_shift(per_sample, split=last)
    per_sample.to_csv(run_dir / "per_sample.csv", index=False)
    hist.to_csv(run_dir / "history.csv", index=False)
    summary_table(results).to_csv(run_dir / "metricas_segmentacion.csv", index=False)
    shape_table(shape_res).to_csv(run_dir / "metricas_forma.csv", index=False)
    by_shift.to_csv(run_dir / "dice_por_desplazamiento.csv")
    (run_dir / "reglas_arbol_forma.txt").write_text(shp.tree_rules(clf), encoding="utf-8")

    metrics = {
        "name": cfg.name,
        "test_evaluado": evaluate_test,
        "n": {s: int(len(data[s].X)) for s in data},
        "preprocesamiento_regenerado": status,
        "segmentacion": {f"{m} | {s}": {
            "dice": round(float(d.dice.mean()), 4), "dice_std": round(float(d.dice.std()), 4),
            "iou": round(float(d.iou.mean()), 4), "precision": round(float(d.precision.mean()), 4),
            "recall": round(float(d.recall.mean()), 4)} for (m, s), d in results.items()},
        "forma": {f"{s} | {m}": {k: round(float(r[k]), 4)
                                 for k in ("accuracy", "balanced_accuracy", "macro_f1")}
                  for (s, m), r in shape_res.items()},
        f"unet_{last}_frac_dice_lt_0_1": round(float((unet_res[(UNET, last)].dice < 0.1).mean()), 4),
        "epocas_entrenadas": int(len(hist)),
        "mejor_epoca": int(hist.loc[hist.val_dice.idxmax(), "epoch"]),
        "mejor_dice_val": round(float(hist.val_dice.max()), 4),
        "segundos_entrenamiento": round(float(hist.seconds.sum()), 1),
        "segundos_por_epoca": round(float(hist.seconds.iloc[1:].mean() if len(hist) > 1
                                          else hist.seconds.mean()), 1),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if figures:
        save_figures(data, preds, results, shape_res, hist, F_plot, fig_dir, split=last)
    say(summary_table(results).to_string(index=False))
    say(f"Resultados guardados en {run_dir.resolve()}")
    return metrics


def save_figures(data, preds, results, shape_res, hist, F_plot, fig_dir: Path, split: str = "test") -> None:
    import matplotlib.pyplot as plt
    from .viz import plot_confusion, plot_feature_boxplots, plot_history, show_samples
    X, Y, P = data[split].X, data[split].Y, preds[split]
    plot_history(hist, fig_dir / "curvas_entrenamiento.png")
    show_samples(data["train"].X, data["train"].Y, n=12, path=fig_dir / "muestras_recortes.png")
    show_samples(X, Y, P, n=12, path=fig_dir / f"predicciones_{split}.png")
    d = results[(UNET, split)]
    worst = d["dice"].sort_values().index[:8].values
    show_samples(X, Y, P, idx=worst, path=fig_dir / f"peores_casos_{split}.png",
                 titles={i: f"#{i}  Dice={d.dice[i]:.2f}" for i in worst})
    plot_feature_boxplots(F_plot, "forma", shp.FEATURE_NAMES, shp.SHAPE_CLASSES,
                          fig_dir / "descriptores_por_forma.png")
    for name, suf in (("máscara anotada", "anotada"), ("máscara U-Net", "unet")):
        plot_confusion(shape_res[(split, name)]["confusion"], shp.SHAPE_CLASSES,
                       shp.SHAPE_CLASSES + ["NONE"], f"Forma BI-RADS en {split}, a partir de la {name}",
                       fig_dir / f"confusion_forma_{suf}.png")
    plt.close("all")


# --------------------------------------------------------------------------- validación cruzada
def cross_validate(cfg: Config, n_splits: int = 5, out_dir=None, verbose: bool = True) -> pd.DataFrame:
    """Validación cruzada de ``n_splits`` particiones AGRUPADA POR PACIENTE sobre el train oficial.

    En cada partición se entrena la U-Net con las otras (early stopping sobre la
    partición retenida, igual que en el experimento principal) y se mide Dice/IoU
    de la U-Net y del prior centrado en la partición retenida. El test oficial no
    se usa. Guarda ``cv_folds.csv`` y ``cv_summary.json`` en ``out_dir``.
    """
    from sklearn.model_selection import GroupKFold
    from .train import train_unet

    out_dir = Path(out_dir or cfg.out_dir / "runs" / f"{cfg.name}_cv{n_splits}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_split(cfg, "train", progress=verbose)
    X, Y, meta = load_split(cfg.processed_dir, f"train_{cfg.cache_tag}")
    rows = []
    folds = GroupKFold(n_splits=n_splits).split(X, groups=meta["patient_id"])
    for k, (tr, va) in enumerate(folds):
        assert set(meta["patient_id"].iloc[tr]).isdisjoint(meta["patient_id"].iloc[va])
        fold_csv = out_dir / f"fold{k}.json"
        if fold_csv.exists():   # reanudable: una partición terminada no se repite
            rows.append(json.loads(fold_csv.read_text(encoding="utf-8")))
            continue
        model, hist = train_unet(cfg, X[tr], Y[tr], X[va], Y[va], verbose=verbose,
                                 ckpt_path=out_dir / f"fold{k}_unet.pt",
                                 log_path=out_dir / "train.log")
        unet = per_sample_metrics(predict_masks(model, X[va], cfg.threshold), Y[va])
        prior = per_sample_metrics(np.broadcast_to(center_prior(X.shape[1:], fit_center_prior(Y[tr])),
                                                   X[va].shape), Y[va])
        row = {"fold": k, "n_train": int(len(tr)), "n_val": int(len(va)),
               "pacientes_val": int(meta["patient_id"].iloc[va].nunique()),
               "dice_unet": float(unet.dice.mean()), "iou_unet": float(unet.iou.mean()),
               "dice_prior": float(prior.dice.mean()), "iou_prior": float(prior.iou.mean()),
               "epocas": int(len(hist)), "mejor_epoca": int(hist.loc[hist.val_dice.idxmax(), "epoch"])}
        fold_csv.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        if verbose:
            print(f"partición {k}: Dice U-Net {row['dice_unet']:.4f} | prior {row['dice_prior']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cv_folds.csv", index=False)
    summary = {c: {"media": round(float(df[c].mean()), 4), "de": round(float(df[c].std(ddof=1)), 4)}
               for c in ("dice_unet", "iou_unet", "dice_prior", "iou_prior")}
    with open(out_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({"n_splits": n_splits, "config": cfg.to_dict(), **summary}, f, indent=2, ensure_ascii=False)
    return df
