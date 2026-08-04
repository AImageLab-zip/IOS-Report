#!/usr/bin/env bash
python - <<'PY'
import numpy as np
import pandas as pd

csv_path = 'output/validation_results_with_stats.csv'
df = pd.read_csv(csv_path)

metrics = [
    ('accuracy', 'Accuracy'),
    ('coverage', 'Coverage'),
    ('bleu-1', 'BLEU-1'),
    ('rouge-l-p', 'ROUGE-L P'),
    ('rouge-l-r', 'ROUGE-L R'),
    ('rouge-l-f', 'ROUGE-L F1'),
    ('meteor', 'METEOR'),
    ('sbert-sim', 'SBERT-sim'),
]

def escape_latex(s: str) -> str:
    return s.replace('_', r'\_')

def rank_masks(values: np.ndarray):
    valid = ~np.isnan(values)
    bold = np.zeros(values.shape[0], dtype=bool)
    under = np.zeros(values.shape[0], dtype=bool)
    if valid.sum() == 0:
        return bold, under

    uniq = np.unique(values[valid])
    uniq_sorted = np.sort(uniq)[::-1]
    best = uniq_sorted[0]
    bold = np.isclose(values, best, equal_nan=False)

    if len(uniq_sorted) > 1:
        second = uniq_sorted[1]
        under = np.isclose(values, second, equal_nan=False)

    return bold, under

highlight = {}
for m, _ in metrics:
    vals = df[m].astype(float).to_numpy() if m in df.columns else np.full(len(df), np.nan)
    highlight[m] = rank_masks(vals)

def fmt_cell(val, std, is_bold, is_under):
    if pd.isna(val):
        base = '--'
    elif pd.isna(std):
        base = f'{val:.3f}'
    else:
        base = f'{val:.3f} ' + r'{\scriptsize$\pm$' + f'{std:.3f}' + '}'

    if base == '--':
        return base
    if is_bold:
        return r'\textbf{' + base + '}'
    if is_under:
        return r'\underline{' + base + '}'
    return base

lines = []
lines.append(r'\begin{tabular}{l' + 'c' * len(metrics) + '}')
lines.append(r'\toprule')
lines.append('Model & ' + ' & '.join(lbl for _, lbl in metrics) + r' \\')
lines.append(r'\midrule')

for i, row in df.iterrows():
    model = escape_latex(str(row['model']))
    cells = []
    for m, _ in metrics:
        std_col = m + '_std'
        v = row[m] if m in row else np.nan
        s = row[std_col] if std_col in row else np.nan
        bmask, umask = highlight[m]
        cells.append(fmt_cell(v, s, bool(bmask[i]), bool(umask[i])))
    lines.append(model + ' & ' + ' & '.join(cells) + r' \\')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

print('\n'.join(lines))
PY
