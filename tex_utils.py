import os
import numpy as np
import pandas as pd

SUFFIX_TEX = {
    "_NULL_RS": "noRS",
    "_NULL_DSM": "noDSM",
    "_NULL_GIU": "noNLRS",
    "_noPOI": "noPOI",
    "_NULL": "noAll",
}
MODEL_TEX = {
    "BASE": "base",
    "AE": "aef",
    "AS": "as",
    "TM": "tm",
    "AE+AS": "aef+as",
    "AE+TM": "aef+tm",
    "AS+TM": "as+tm",
    "AE+AS+TM": "aef+as+tm",
}
# Stratum labels are emitted as plain text (not \textsc) so they can contain
# special characters such as $\geq$.
STRATUM_TEX = {
    "none":   r"none",
    "any":    r"any ($\geq$1)",
    "sparse": r"sparse (1)",
    "low":    r"low (2--3)",
    "medium": r"medium (4--8)",
    "dense":  r"dense ($>$8)",
}

def _tlabel(val, name):
    if name == "suffix":
        return SUFFIX_TEX.get(val, val.lstrip("_"))
    if name == "model":
        return MODEL_TEX.get(val, val.lower())
    if name == "stratum":
        return STRATUM_TEX.get(val, str(val))
    return str(val).lower()


def save_tex(df, path, caption, label):
    """Save a DataFrame as a LaTeX table.

    Row index levels become left-hand columns; the outermost levels use
    \\multirow.  Groups are separated by \\midrule.  Within each group
    (= every combination of levels except the last "model" level) the
    minimum per metric column is bolded.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    metric_cols = list(df.columns)
    n_m = len(metric_cols)

    if not isinstance(df.index, pd.MultiIndex):
        df = df.copy()
        df.index = pd.MultiIndex.from_arrays(
            [df.index], names=[df.index.name or "model"]
        )
    idx_names = list(df.index.names)
    n_lev = len(idx_names)

    rows_data = [
        (t if isinstance(t, tuple) else (t,), list(v))
        for t, v in zip(df.index, df.values)
    ]

    # Partition into groups by all levels except the last ("model")
    groups, cur_key, cur_rows = [], None, []
    for r in rows_data:
        k = r[0][:-1]
        if k != cur_key:
            if cur_rows:
                groups.append((cur_key, cur_rows))
            cur_key, cur_rows = k, [r]
        else:
            cur_rows.append(r)
    if cur_rows:
        groups.append((cur_key, cur_rows))

    needs_resize = n_lev > 1
    col_spec = "l" * n_lev + " " + "c" * n_m

    out = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if needs_resize:
        out.append(r"\resizebox{0.9\linewidth}{!}{")
    out.append(rf"\begin{{tabular}}{{{col_spec}}}")
    out.append(r"\toprule")

    # Header
    hdr = []
    for n in idx_names:
        if n == "model":
            hdr.append(r"\textbf{Model}")
        elif n == "suffix":
            hdr.append(r"\textbf{Ablation}")
        elif n == "city":
            hdr.append(r"\textbf{City}")
        elif n == "stratum":
            hdr.append(r"\textbf{POI stratum}")
        else:
            hdr.append(rf"\textbf{{{n.capitalize()}}}")
    hdr += [rf"\textbf{{{m}}}" for m in metric_cols]
    out.append(" & ".join(hdr) + r" \\")
    out.append(r"\midrule")

    first_group = True
    for g_key, g_rows in groups:
        if not first_group:
            out.append(r"\midrule")
        first_group = False

        vals_arr = np.array([r[1] for r in g_rows], dtype=float)
        col_min = np.nanmin(np.round(vals_arr, 3), axis=0)

        for ri, (idx, vals) in enumerate(g_rows):
            parts = []

            # Non-model index levels → \multirow or blank
            for li in range(n_lev - 1):
                same_as_prev = ri > 0 and all(
                    g_rows[ri - 1][0][j] == idx[j] for j in range(li + 1)
                )
                if same_as_prev:
                    parts.append("")
                else:
                    cnt = sum(
                        1 for r in g_rows[ri:]
                        if all(r[0][j] == idx[j] for j in range(li + 1))
                    )
                    tex_val = _tlabel(idx[li], idx_names[li])
                    use_sc = idx_names[li] != "stratum"
                    cell = rf"\textsc{{{tex_val}}}" if use_sc else tex_val
                    if cnt > 1:
                        parts.append(rf"\multirow{{{cnt}}}{{*}}{{{cell}}}")
                    else:
                        parts.append(cell)

            # Model (last level)
            parts.append(rf"\textsc{{{_tlabel(idx[-1], 'model')}}}")

            # Metric values
            for ci, v in enumerate(vals):
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    parts.append("")
                    continue
                if np.isnan(fv):
                    parts.append("")
                else:
                    s = f"{fv:.3f}"
                    if abs(round(fv, 3) - col_min[ci]) < 1e-9:
                        s = rf"\textbf{{{s}}}"
                    parts.append(s)

            out.append(" & ".join(parts) + r" \\")

    out += [r"\bottomrule", rf"\end{{tabular}}"]
    if needs_resize:
        out.append("}")
    out += [r"\vspace{-1.5em}", r"\end{table}"]

    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"Saved: {path}")
