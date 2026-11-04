"""Estadística para comparar métodos sobre el Dice por recorte (bootstrap y Wilcoxon)."""
import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_mean_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42, groups=None) -> dict:
    """IC (percentil) del promedio por bootstrap.

    Sin ``groups`` se remuestrean recortes. Con ``groups`` (p. ej. patient_id) se
    remuestrean grupos completos (bootstrap por conglomerados), lo que respeta que
    varios recortes de un mismo paciente no son independientes.
    """
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        means = v[idx].mean(axis=1)
    else:
        g = pd.Series(v).groupby(np.asarray(groups))
        sums, counts = g.sum().values, g.size().values
        pick = rng.integers(0, len(sums), size=(n_boot, len(sums)))
        means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"media": float(v.mean()), "ic_inf": float(lo), "ic_sup": float(hi), "n_boot": n_boot}


def paired_wilcoxon(a, b) -> dict:
    """Prueba de rangos con signo de Wilcoxon pareada (bilateral) de a vs b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    res = stats.wilcoxon(a, b, alternative="two-sided")
    d = a - b
    return {"n": int(len(a)), "estadistico_W": float(res.statistic), "p_valor": float(res.pvalue),
            "mediana_diferencia": float(np.median(d)), "media_diferencia": float(d.mean()),
            "frac_a_mejor": float((d > 0).mean())}
