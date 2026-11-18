#!/usr/bin/env python3
"""exp3_a3_feature_model.py -- physics-explicit classifier (SVM / Random Forest), variation A3.

Tests whether a model trained DIRECTLY on explicit physical impairment features resists an attack
the learned IQ-domain models (CVNN/CNN1D) do not. Features per 1024-window (reuse of
exp3_impairment_fit.frame_stats math): DC offset, I/Q gain imbalance, I/Q quadrature skew, CFO (Hz),
PAPR (dB) -- the same quantities the fingerprint is physically built on.

Train SVM (RBF, standardized) and Random Forest on the SAME build() split as the CVNN (runs 1-2
train, run 3 held out). Then the attack: craft the untargeted six-attack ensemble delta IN THE IQ
DOMAIN on the CVNN (the learned model), over the held-out run-3 windows, at a grid of eps.
Reconstruct the perturbed complex window, RECOMPUTE its physical features, and classify with SVM/RF.
Compare the feature-model fooling rate to the CVNN's own at the same eps.

Hypothesis: a small L_inf delta that saturates the CNN barely moves the robust aggregate features
(CFO estimated over the whole window, PAPR a ratio), so the physics model resists -- a DEFENSE
result, not a generalization check.  (White-box attack directly on the feature model is a separate,
harder job -- RF is non-differentiable; noted as follow-on.)
"""
import argparse, csv
import numpy as np, torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from exp3_cvnn import CVNN, build
from exp3_fp_model import NUM_CLASSES, FS
from exp3_attack_matrix import craft, to_reim_t, predict, ATTACKS
import os
CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))

ATK = ['fgsm', 'bim', 'mim', 'pgd_a0.01', 'pgd_a0.05', 'pgd_a0.10']
FEATS = ['DC', 'IQgain', 'IQquad', 'CFO_Hz', 'PAPR_dB']


def win_feats(W):
    """physical features for complex windows W [N, L] (unit-RMS normalized here). -> [N,5]."""
    W = np.asarray(W, np.complex128)
    rms = np.sqrt(np.mean(np.abs(W) ** 2, axis=1, keepdims=True)) + 1e-12
    x = W / rms
    I, Q = x.real, x.imag
    dc = np.abs(np.mean(x, axis=1))
    gimb = np.std(I, axis=1) / (np.std(Q, axis=1) + 1e-12)
    quad = np.mean(I * Q, axis=1) / (np.sqrt(np.mean(I ** 2, axis=1) * np.mean(Q ** 2, axis=1)) + 1e-12)
    r = np.sum(x[:, 1:] * np.conj(x[:, :-1]), axis=1)
    cfo = FS / (2 * np.pi) * np.angle(r)
    papr = 10 * np.log10(np.max(np.abs(x) ** 2, axis=1) / (np.mean(np.abs(x) ** 2, axis=1) + 1e-12))
    return np.stack([dc, gimb, quad, cfo, papr], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cvnn', default='fingerprint_cvnn.pt')
    ap.add_argument('--root', default=f'{CAPTURES}/train')
    ap.add_argument('--val-run', type=int, default=3)
    ap.add_argument('--cache', default='.cnn1d_cache/train.npz')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.cache), exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0)

    # same build() split as the CVNN/CNN1D (runs 1-2 train, run 3 val)
    Xtr, ytr, Xva, yva, vfid, counts = build(a.root, a.val_run, 384, 2.0, a.cache, False, None, False)
    Ftr, Fva = win_feats(Xtr), win_feats(Xva)
    print(f"features {Ftr.shape} train / {Fva.shape} val  ({', '.join(FEATS)})")

    models = {
        'svm': make_pipeline(StandardScaler(), SVC(C=10, gamma='scale')),
        'rf':  RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
    }
    for name, clf in models.items():
        clf.fit(Ftr, ytr)
        acc = (clf.predict(Fva) == yva).mean()
        print(f"  {name.upper():3} clean val window-accuracy: {acc:.4f}")

    # ---- transferred attack: craft delta on CVNN over held-out val windows, recompute features ----
    # subsample the attack set (stratified ~300/class) so the six-attack x eps sweep is tractable;
    # the classifiers are already trained on the FULL split above.
    rng = np.random.RandomState(0)
    sel = np.concatenate([rng.permutation(np.where(yva == c)[0])[:300] for c in range(NUM_CLASSES)])
    Xva, yva = Xva[sel], yva[sel]
    print(f"  attack set (stratified): {len(yva)} windows")

    cvnn = CVNN(NUM_CLASSES).to(dev)
    cvnn.load_state_dict(torch.load(a.cvnn, map_location=dev, weights_only=False)); cvnn.eval()
    re, im = to_reim_t(Xva.astype(np.complex64), dev)
    y = torch.from_numpy(yva).long().to(dev)
    # clean predictions per model; attack success is measured ONLY on windows each model gets
    # right when clean, so different baseline accuracies do not inflate the fooling rate.
    Fva_s = win_feats(Xva)
    clean_pred = {
        'cvnn': predict(cvnn, re, im).cpu().numpy(),
        'svm':  models['svm'].predict(Fva_s),
        'rf':   models['rf'].predict(Fva_s),
    }
    correct = {k: (v == yva) for k, v in clean_pred.items()}
    for k in ['cvnn', 'svm', 'rf']:
        print(f"  {k.upper():4} clean-correct on attack set: {correct[k].mean():.4f} ({correct[k].sum()} win)")

    grid = np.round(np.arange(0.005, 0.0801, 0.005), 3)
    print("\ntransferred attack (craft on CVNN, recompute physics features)")
    print("attack success = flip from a CORRECT clean prediction (among clean-correct windows):")
    print(f"{'eps':>6}{'CVNN':>8}{'SVM':>8}{'RF':>8}")
    rows = []
    for eps in grid:
        flipped = {k: np.zeros(len(yva), bool) for k in ['cvnn', 'svm', 'rf']}
        for atk in ATK:
            dre, dim = craft(cvnn, re, im, y, float(eps), ATTACKS[atk], +1.0, dev)
            flipped['cvnn'] |= (predict(cvnn, re + dre, im + dim).cpu().numpy() != yva)
            Wadv = ((re + dre).squeeze(1) + 1j * (im + dim).squeeze(1)).cpu().numpy()
            Fadv = win_feats(Wadv)
            for name in ['svm', 'rf']:
                flipped[name] |= (models[name].predict(Fadv) != yva)
        # attack success rate on clean-correct windows
        asr = {k: (flipped[k] & correct[k]).sum() / max(1, correct[k].sum()) for k in flipped}
        rows.append((eps, asr['cvnn'], asr['svm'], asr['rf']))
        print(f"{eps:>6.3f}{asr['cvnn']:>8.2f}{asr['svm']:>8.2f}{asr['rf']:>8.2f}")

    with open('docs/exp3_a3_feature_model.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['eps', 'fool_cvnn', 'fool_svm', 'fool_rf'])
        for r in rows:
            w.writerow([f'{r[0]:.3f}', f'{r[1]:.4f}', f'{r[2]:.4f}', f'{r[3]:.4f}'])
    np.savez('docs/exp3_a3_feature_model.npz', grid=grid, rows=np.array(rows),
             feats=FEATS)
    print("\nsaved exp3_a3_feature_model.csv + .npz")


if __name__ == '__main__':
    main()
