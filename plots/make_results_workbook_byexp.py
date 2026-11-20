#!/usr/bin/env python3
"""Per-experiment results workbook: docs/exp3_all_results_byexp.xlsx.
One sheet per experiment (A1-ML..A4-ML, B1-W, B2-W; B3/B4/B5 not in the paper), each stacking that
experiment's canonical result table(s) as labeled blocks. Per-method CSVs are concatenated
into a single table with a leading 'method' column. Contents sheet lists every sheet."""
import os as _os, sys as _sys
_R = _os.path.abspath(_os.path.dirname(__file__))
while _R != _os.path.dirname(_R) and not _os.path.isdir(_os.path.join(_R, ".git")):
    _R = _os.path.dirname(_R)
_sys.path.insert(0, _os.path.join(_R, "lib"))
_os.chdir(_R)
import os
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, PatternFill
from openpyxl.drawing.image import Image as XLImage

# sheet -> (figure path, caption) embedded below that sheet's data blocks
FIGS = {
    'A2-ML': ('docs/transfer_fooling_vs_eps.png',
              'A2 digital cross-model transferability: one CVNN-crafted perturbation fools '
              'CNN1D and the physics-feature SVM/RF.'),
}

METHODS = ['fgsm', 'bim', 'mim', 'pgd_a0.01', 'pgd_a0.05', 'pgd_a0.10']
OUT = 'docs/exp3_all_results_byexp.xlsx'

# B1 0920 gain -> received PSR (dB, adversary/victim), the same calibration the B1 figures use:
# measured points, linear 1 dB/dB extrapolation elsewhere (anchored gain 20 -> PSR -4.5).
PSR_MEAS = {20: -4.5, 25: 1.7, 30: 6.3, 35: 9.9, 40: 10.0}


def g2psr(g):
    g = int(round(float(g)))
    return PSR_MEAS[g] if g in PSR_MEAS else -4.5 + (g - 20)


def with_psr(df):
    """Insert a psr_db column right after adv_gain_db (B1 0920 calibration)."""
    if df is None or 'adv_gain_db' not in df.columns:
        return df
    df = df.copy()
    gi = df.columns.get_loc('adv_gain_db')
    df.insert(gi + 1, 'psr_db', [g2psr(v) for v in df['adv_gain_db']])
    return df


def load(path):
    return pd.read_csv(path, comment='#', skip_blank_lines=True)


def concat_methods(prefix, suffix, root='docs', methods=METHODS):
    frames = []
    for m in methods:
        p = os.path.join(root, f'{prefix}{m}{suffix}')
        if os.path.exists(p):
            df = load(p); df.insert(0, 'method', m); frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def block(title, df):
    return (title, df) if df is not None else None


# sheet_name -> (experiment title, [ (block title, dataframe), ... ])
SHEETS = []


def add(sheet, title, blocks):
    SHEETS.append((sheet, title, [b for b in blocks if b is not None]))


A1_root = '.'
add('A1-ML', 'A1 — white-box CNN1D attack over the air', [
    block('CNN1D reachability matrix', load('docs/reachability_matrix_cnn1d.csv')),
    block('CVNN reachability matrix', load('docs/reachability_matrix.csv')),
    block('Attack sweep — CVNN (fool / landed / BER vs eps)', load('docs/reachability_cvnn.csv')),
    block('OTA BER per method — CNN1D', concat_methods('a1_', '_ota_ber.csv')),
    block('OTA BER per method — CVNN', concat_methods('a1cvnn_', '_ota_ber.csv')),
])
add('A2-ML', 'A2 — cross-model transferability', [
    block('CNN1D <-> CVNN transferability', load('docs/exp3_transfer_cnn1d_cvnn.csv')),
    block('OTA: one delta scored across all 5 models', load('docs/exp3_ota_multimodel.csv')),
])


def _npz_curves(path):
    z = np.load(path)
    return pd.DataFrame({
        'eps': z['grid'],
        'craft_cvnn_eval_cvnn': z['cvnn_on_cvnn'],
        'craft_cvnn_eval_cnn1d': z['cvnn_on_cnn1d'],
        'craft_cnn1d_eval_cvnn': z['cnn1d_on_cvnn'],
        'craft_cnn1d_eval_cnn1d': z['cnn1d_on_cnn1d'],
    })


add('A2-transfer-raw', 'A2 digital transfer graph — raw input files', [
    block('exp3_transfer_cnn1d_cvnn.npz — neural-to-neural transfer curves (left panel)',
          _npz_curves('docs/exp3_transfer_cnn1d_cvnn.npz')),
    block('exp3_a3_feature_model.csv — transfer onto physics-feature SVM / RF (right panel)',
          load('docs/exp3_a3_feature_model.csv')),
])
add('A3-ML', 'A3 — physics-feature SVM / Random Forest', [
    block('Feature-model robustness', load('docs/exp3_a3_feature_model.csv')),
])
add('A4-ML', 'A4 — frequency-domain CVNN', [
    block('Freq-domain CVNN reachability matrix', load('docs/reachability_matrix_freqcvnn.csv')),
])
add('B1-W', 'B1 — adversary power sweep (fooling vs BER), eps=0.01 re-baseline 0920', [
    block('Untargeted — fooling / BER vs adversary gain & received PSR (per method, 0920)',
          with_psr(concat_methods('b1_cvnn_foolber_', '.csv'))),
    block('Targeted device_2 — target rate / BER vs adversary gain & received PSR (per method, 0920)',
          with_psr(concat_methods('b1_cvnn_tgt_d2_foolber_', '.csv'))),
])
add('B2-W', 'B2 — matched vs unmatched (universal) frames', [
    block('Matched vs universal, all 6 methods (eps sweep, untargeted + targeted)',
          load('docs/b2_matched_vs_unmatched.csv')),
])
TITLE_FONT = Font(bold=True, size=12, color='1F3864')
HDR_FONT = Font(bold=True)
HDR_FILL = PatternFill('solid', fgColor='DDE6EF')

wb = Workbook()
wb.remove(wb.active)

# Contents
cs = wb.create_sheet('Contents')
cs.cell(1, 1, 'Sheet').font = HDR_FONT
cs.cell(1, 2, 'Experiment').font = HDR_FONT
cs.cell(1, 3, 'Tables (rows)').font = HDR_FONT
for c in cs[1]:
    c.fill = HDR_FILL
cr = 2
for sheet, title, blocks in SHEETS:
    cs.cell(cr, 1, sheet)
    cs.cell(cr, 2, title)
    cs.cell(cr, 3, '; '.join(f'{bt} ({len(df)})' for bt, df in blocks))
    cr += 1
for i, w in enumerate([10, 62, 90], 1):
    cs.column_dimensions[get_column_letter(i)].width = w
cs.freeze_panes = 'A2'


def cellval(v):
    if isinstance(v, float) and pd.isna(v):
        return ''
    if isinstance(v, np.generic):
        return v.item()
    return v


for sheet, title, blocks in SHEETS:
    ws = wb.create_sheet(sheet)
    ws.cell(1, 1, title).font = Font(bold=True, size=13, color='1F3864')
    r = 3
    maxcols = 1
    for bt, df in blocks:
        ws.cell(r, 1, bt).font = TITLE_FONT
        r += 1
        for j, col in enumerate(df.columns, 1):
            c = ws.cell(r, j, str(col)); c.font = HDR_FONT; c.fill = HDR_FILL
        maxcols = max(maxcols, len(df.columns))
        r += 1
        for _, row in df.iterrows():
            for j, v in enumerate(row, 1):
                ws.cell(r, j, cellval(v))
            r += 1
        r += 2
    # embedded figure (if any) below the data blocks
    if sheet in FIGS:
        path, caption = FIGS[sheet]
        if os.path.exists(path):
            ws.cell(r, 1, 'Figure').font = TITLE_FONT
            ws.cell(r + 1, 1, caption).font = Font(italic=True, size=10)
            img = XLImage(path)
            scale = 1000 / img.width
            img.width = 1000
            img.height = int(img.height * scale)
            ws.add_image(img, f'A{r + 3}')
    # column widths
    for j in range(1, maxcols + 1):
        letter = get_column_letter(j)
        w = 10
        for cell in ws[letter]:
            if cell.value is not None:
                w = max(w, min(46, len(str(cell.value)) + 2))
        ws.column_dimensions[letter].width = w
    ws.freeze_panes = 'A3'

wb.save(OUT)
print('wrote', OUT, '-', len(SHEETS), 'experiment sheets + Contents')
for sheet, title, blocks in SHEETS:
    print(f'  {sheet:7s} {sum(len(df) for _, df in blocks):5d} rows across {len(blocks)} block(s)')
