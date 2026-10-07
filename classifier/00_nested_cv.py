"""Predict aiTBS response from baseline features with nested cross-validation.

Runs three feature sets (EEG only, fMRI only and early fusion of both) for two
targets (HAM-D responder primary, BDI responder secondary). Each outer
leave-one-out fold tunes feature selection, model choice and hyperparameters
with an inner stratified CV, refits on the training subjects and predicts the
held-out subject. The pooled out-of-fold predictions give ROC-AUC, balanced
accuracy, sensitivity and specificity. Each model is also evaluated on its
own.

Inputs
------
MASTER
    master_features.csv (from classification/01_build_master_table.py) with
    baseline eyes-closed features; absolute power is excluded.

Outputs
-------
OUTDIR/{project}/{target}/
    oof_predictions.csv, model_selection_frequency.csv,
    feature_selection_frequency.csv, permutation_null_auc.csv,
    model_comparison.csv, model_comparison_oof.csv, roc_nested.png and
    model_comparison.png.
OUTDIR
    nested_summary.csv, merged_nested_{target}.png and nested_manifest.json.

Notes
-----
- This evaluates the whole pipeline, not one fixed model; the selection
  frequencies show which models and features were chosen most often.
- Inner folds shrink to min(INNER_K, n_pos, n_neg) to keep both classes.
- Balanced accuracy, sensitivity and specificity use THRESHOLD; AUC is
  threshold-free and is the primary metric.
- The permutation test reruns the full nested CV N_PERM times; set N_PERM to
  0 to skip it.

Authors: Karahan Yılmazer & Elvira Shemena
"""

MASTER = "results/master/master_features.csv"
OUTDIR = "results/nested"

PROJECTS = [
    "eeg",
    "fmri",
    "fusion",
]
TARGETS = ["HAMD", "BDI"]
MODELS_ORDER = [
    "logistic",
    "lda",
    "svc",
    "knn",
    "rf",
]

K_GRID = [2, 4, 6, 8, 10, 15]
INNER_K = 5
THRESHOLD = 0.5

N_PERM = (
    1000
)
N_JOBS = -1
SEED = 42
CONDITION = "ec"
EXCLUDE = ["pow_abs"]

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore")
CLF_COL = {"HAMD": "responder_HAMD17", "BDI": "responder_BDI"}
RESPONDER_POSITIVE = "responder"


class _AdaptiveStratifiedKFold:
    """StratifiedKFold whose fold count adapts to the minority class at split time.

    Used for SVC probability calibration, where nested CV can pass very small,
    imbalanced subsets. Having no n_splits attribute avoids the upfront class
    count check of CalibratedClassifierCV.
    """

    def __init__(self, max_splits=3, random_state=SEED):
        self.max_splits = max_splits
        self.random_state = random_state

    def _n(self, y):
        """Return the fold count min(max_splits, n_pos, n_neg), at least 2."""
        y = np.asarray(y)
        pos, neg = int(y.sum()), int(len(y) - y.sum())
        return max(2, min(self.max_splits, pos, neg))

    def split(self, X, y, groups=None):
        """Yield stratified train and test indices for y."""
        return StratifiedKFold(
            n_splits=self._n(y), shuffle=True, random_state=self.random_state
        ).split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the fold count for y."""
        return self._n(y)


BASE = {
    "logistic": LogisticRegression(class_weight="balanced", max_iter=2000),
    "lda": LinearDiscriminantAnalysis(solver="lsqr"),
    "svc": CalibratedClassifierCV(
        SVC(kernel="linear", class_weight="balanced", random_state=SEED),
        cv=_AdaptiveStratifiedKFold(),
        ensemble=False,
    ),
    "knn": KNeighborsClassifier(),
    "rf": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=SEED
    ),
}
HPGRID = {
    "logistic": ("mdl__C", [0.01, 0.1, 1.0, 10.0]),
    "lda": ("mdl__shrinkage", [0.9, 0.5, 0.2]),
    "svc": ("mdl__estimator__C", [0.1, 1.0, 10.0]),
    "knn": ("mdl__n_neighbors", [3, 5, 7]),
    "rf": ("mdl__max_depth", [2, 3, 4]),
}


def load_master():
    """Load the master feature table."""
    return pd.read_csv(MASTER)


def feats(df, project):
    """Return the feature columns of a project ('eeg', 'fmri' or 'fusion')."""
    eeg = [
        c
        for c in df.columns
        if c.startswith("eeg_")
        and c.endswith("_base")
        and f"_{CONDITION}_" in c
        and not any(x in c for x in EXCLUDE)
    ]
    fmri = [
        c
        for c in df.columns
        if c.startswith("fmri_")
        and c.endswith("_base")
        and not any(x in c for x in EXCLUDE)
    ]
    return {"eeg": eeg, "fmri": fmri, "fusion": eeg + fmri}[project]


def y_vec(df, target):
    """Return the binary responder labels (1 = responder) of a target."""
    return (
        (df[CLF_COL[target]].astype(str).str.strip() == RESPONDER_POSITIVE)
        .astype(int)
        .to_numpy()
    )


def base_pipe(n_features):
    """Return the impute, scale, select and classify pipeline template."""
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("sel", SelectKBest(f_classif)),
            ("mdl", LogisticRegression(max_iter=2000)),
        ]
    )


def param_grid(models, n_features):
    """Return one grid per model over its hyperparameter and the SelectKBest k."""
    ks = [k for k in K_GRID if k <= n_features] or [min(2, n_features)]
    grids = []
    for m in models:
        hp_name, hp_vals = HPGRID[m]
        grids.append({"mdl": [BASE[m]], hp_name: hp_vals, "sel__k": ks})
    return grids


def inner_splits(y_train):
    """Return the inner CV splitter, shrunk to keep both classes in each fold."""
    pos, neg = int(y_train.sum()), int(len(y_train) - y_train.sum())
    n = max(2, min(INNER_K, pos, neg))
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=SEED)


def _outer_fold(X, y, tr, te, grids):
    """Fit one outer fold and predict its held-out subject.

    Runs the inner grid search on the training subjects and refits the best
    pipeline. Returns the test index, predicted probability, chosen model, k,
    hyperparameter value and feature selection mask.
    """
    gs = GridSearchCV(
        base_pipe(X.shape[1]),
        grids,
        cv=inner_splits(y[tr]),
        scoring="roc_auc",
        refit=True,
        error_score=np.nan,
        n_jobs=1,
    )
    gs.fit(X[tr], y[tr])
    best = gs.best_estimator_
    bp = gs.best_params_
    mdl_name = [k for k, v in BASE.items() if type(v) is type(bp["mdl"])][0]
    hp = next(v for k, v in bp.items() if k.startswith("mdl__"))
    return (
        int(te[0]),
        float(best.predict_proba(X[te])[:, 1][0]),
        mdl_name,
        bp["sel__k"],
        hp,
        best.named_steps["sel"].get_support().astype(int),
    )


def _perm_outer_fold(i, X, yp, tr, te, grids):
    """Run _outer_fold on permuted labels and tag the result with permutation i."""
    idx, prob, *_ = _outer_fold(X, yp, tr, te, grids)
    return i, idx, prob


def nested_oof(X, y, models, n_jobs=N_JOBS, desc=None, progress=True):
    """Run nested CV and return out-of-fold probabilities and chosen settings.

    Results are placed by subject index because the parallel pool returns them
    out of order. The settings dict holds the per-subject model, k and
    hyperparameter, and the summed feature selection counts.
    """
    grids = param_grid(models, X.shape[1])
    tasks = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(_outer_fold)(X, y, tr, te, grids) for tr, te in LeaveOneOut().split(X)
    )
    res = list(
        tqdm(tasks, total=len(y), desc=desc or "outer folds", leave=False, disable=not progress)
    )
    n = len(y)
    oof = np.full(n, np.nan)
    chosen_models, chosen_k, chosen_hp = [None] * n, [None] * n, [None] * n
    feat_support = np.zeros(X.shape[1], int)
    for idx, prob, mdl_name, k, hp, support in res:
        oof[idx] = prob
        chosen_models[idx] = mdl_name
        chosen_k[idx] = k
        chosen_hp[idx] = hp
        feat_support += support
    return oof, dict(
        models=chosen_models, k=chosen_k, hp=chosen_hp, feat_support=feat_support
    )


def metrics_from_oof(y, oof):
    """Return AUC, balanced accuracy, sensitivity, specificity and confusion counts."""
    auc = roc_auc_score(y, oof) if len(np.unique(y)) == 2 else np.nan
    pred = (oof >= THRESHOLD).astype(int)
    bal = balanced_accuracy_score(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return dict(
        auc=auc,
        balanced_acc=bal,
        sensitivity=sens,
        specificity=spec,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def permutation_p(X, y, models, observed_auc):
    """Return the permutation p-value of the observed AUC and the null AUCs.

    Reruns the full nested CV on N_PERM label shuffles, flattened into one
    parallel pool of fold fits.
    """
    if not N_PERM or np.isnan(observed_auc):
        return np.nan, np.array([])

    grids = param_grid(models, X.shape[1])
    splits = list(LeaveOneOut().split(X))
    perms = {
        i: y[np.random.default_rng(SEED + i).permutation(len(y))]
        for i in range(N_PERM)
    }
    tasks = Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(
        delayed(_perm_outer_fold)(i, X, perms[i], tr, te, grids)
        for i in range(N_PERM)
        for tr, te in splits
    )
    oof_by_perm = {i: np.full(len(y), np.nan) for i in range(N_PERM)}
    for i, idx, prob in tqdm(
        tasks, total=N_PERM * len(splits), desc="permutation folds", leave=False
    ):
        oof_by_perm[i][idx] = prob

    null = np.array(
        [
            roc_auc_score(perms[i], oof_by_perm[i])
            for i in range(N_PERM)
            if len(np.unique(perms[i])) == 2
        ],
        float,
    )
    p = (
        float((1 + np.sum(null >= observed_auc)) / (null.size + 1))
        if null.size
        else np.nan
    )
    return p, null


def _plt():
    """Import pyplot with the non-interactive Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def roc_plot(y, oof, auc, title, path):
    """Save the ROC curve of the pooled out-of-fold predictions."""
    plt = _plt()
    fpr, tpr, _ = roc_curve(y, oof)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot(fpr, tpr, color="#4f6d9a", lw=2, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def model_comparison_plot(names, aucs, balaccs, title, path):
    """Save a bar chart of pooled AUC and balanced accuracy per model."""
    plt = _plt()
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 2.5, 4.2))
    ax.bar(x - w / 2, aucs, w, label="pooled AUC", color="#4f6d9a")
    ax.bar(x + w / 2, balaccs, w, label="balanced acc.", color="#b9c6dd")
    for i, v in enumerate(aucs):
        ax.text(x[i] - w / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="0.5", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def merged_nested_plot(rows, target, path):
    """Save a bar chart comparing EEG, fMRI and fusion for one target."""
    plt = _plt()
    order = [
        r for p in PROJECTS for r in rows if r["project"] == p and r["target"] == target
    ]
    labels = [r["project"].upper() for r in order]
    auc = [r["auc"] for r in order]
    bal = [r["balanced_acc"] for r in order]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 2.5, 4.4))
    ax.bar(x - w / 2, auc, w, label="pooled AUC", color="#4f6d9a")
    ax.bar(x + w / 2, bal, w, label="balanced acc.", color="#b9c6dd")
    for i, v in enumerate(auc):
        ax.text(x[i] - w / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="0.5", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("score")
    ax.set_title(f"Nested CV - {target}: EEG vs fMRI vs fusion")
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(df, project, target):
    """Run the nested CV and per-model comparison for one project and target.

    Saves all outputs of this combination and returns its summary row.
    """
    fnames = feats(df, project)
    X = df[fnames].to_numpy(float)
    y = y_vec(df, target)
    out = Path(OUTDIR) / project / target
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"\n=== {project.upper()} -> {target}  (n={len(y)}, responders={int(y.sum())}, feats={len(fnames)}) ==="
    )

    oof, sel = nested_oof(X, y, MODELS_ORDER, desc=f"{project}/{target} full")
    m = metrics_from_oof(y, oof)
    p, null_auc = permutation_p(X, y, MODELS_ORDER, m["auc"])
    roc_plot(
        y,
        oof,
        m["auc"],
        f"{project.upper()} {target} - nested CV (pooled over {len(y)})",
        out / "roc_nested.png",
    )
    pd.DataFrame(
        {
            "mri_id": df["mri_id"].to_numpy(),
            "y_true": y,
            "oof_prob": oof,
            "chosen_model": sel["models"],
            "chosen_k": sel["k"],
            "chosen_hp": sel["hp"],
        }
    ).to_csv(out / "oof_predictions.csv", index=False)
    if null_auc.size:
        pd.DataFrame(
            {"perm_idx": np.arange(null_auc.size), "perm_auc": null_auc}
        ).to_csv(out / "permutation_null_auc.csv", index=False)

    mc = Counter(sel["models"])
    freq = pd.DataFrame(
        {"model": list(mc), "times_chosen": list(mc.values())}
    ).sort_values("times_chosen", ascending=False)
    freq.to_csv(out / "model_selection_frequency.csv", index=False)
    fs = pd.DataFrame(
        {
            "feature": fnames,
            "times_selected": sel["feat_support"],
            "pct_folds": (100 * sel["feat_support"] / len(y)).round(1),
        }
    ).sort_values("times_selected", ascending=False)
    fs.to_csv(out / "feature_selection_frequency.csv", index=False)
    top_feats = "; ".join(fs.head(6)["feature"].tolist())

    comp = []
    oof_by_model = {"mri_id": df["mri_id"].to_numpy(), "y_true": y}
    for mdl in tqdm(MODELS_ORDER, desc=f"{project}/{target} per-model", leave=False):
        o2, sel2 = nested_oof(X, y, [mdl], desc=f"{project}/{target} {mdl}")
        mm = metrics_from_oof(y, o2)
        oof_by_model[f"oof_prob_{mdl}"] = o2
        comp.append(
            dict(
                model=mdl,
                auc=round(mm["auc"], 3),
                balanced_acc=round(mm["balanced_acc"], 3),
                sensitivity=round(mm["sensitivity"], 3),
                specificity=round(mm["specificity"], 3),
                tp=mm["tp"],
                fp=mm["fp"],
                fn=mm["fn"],
                tn=mm["tn"],
                median_k=int(np.median(sel2["k"])),
            )
        )
    pd.DataFrame(comp).to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame(oof_by_model).to_csv(out / "model_comparison_oof.csv", index=False)
    model_comparison_plot(
        [c["model"] for c in comp],
        [c["auc"] for c in comp],
        [c["balanced_acc"] for c in comp],
        f"{project.upper()} {target} - nested CV per model",
        out / "model_comparison.png",
    )

    med_k = int(np.median(sel["k"]))
    print(
        f"  full nested: AUC={m['auc']:.3f} | bal.acc={m['balanced_acc']:.3f} | "
        f"sens={m['sensitivity']:.2f} spec={m['specificity']:.2f} | "
        f"perm p={p if np.isnan(p) else round(p, 3)}"
    )
    print(f"  most-chosen model={mc.most_common(1)[0][0]} | median k={med_k}")
    print(f"  top features: {top_feats}")
    return dict(
        project=project,
        target=target,
        auc=round(m["auc"], 3),
        balanced_acc=round(m["balanced_acc"], 3),
        sensitivity=round(m["sensitivity"], 3),
        specificity=round(m["specificity"], 3),
        tp=m["tp"],
        fp=m["fp"],
        fn=m["fn"],
        tn=m["tn"],
        most_chosen_model=mc.most_common(1)[0][0],
        median_k=med_k,
        perm_p=p,
        n_perm_valid=int(null_auc.size),
        top_features=top_feats,
    )


def main():
    """Run every project and target and save the summary, plots and manifest."""
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    df = load_master().sort_values("mri_id").reset_index(drop=True)
    combos = [(proj, tgt) for proj in PROJECTS for tgt in TARGETS]
    summary = [run(df, proj, tgt) for proj, tgt in tqdm(combos, desc="project x target")]
    pd.DataFrame(summary).to_csv(Path(OUTDIR) / "nested_summary.csv", index=False)
    for tgt in TARGETS:
        merged_nested_plot(summary, tgt, Path(OUTDIR) / f"merged_nested_{tgt}.png")
    json.dump(
        dict(
            projects=PROJECTS,
            targets=TARGETS,
            models=MODELS_ORDER,
            k_grid=K_GRID,
            inner_k=INNER_K,
            threshold=THRESHOLD,
            n_perm=N_PERM,
            seed=SEED,
        ),
        open(Path(OUTDIR) / "nested_manifest.json", "w"),
        indent=2,
    )
    print("\n==== NESTED CV SUMMARY (pooled over all 20 out-of-fold predictions) ====")
    print(
        pd.DataFrame(summary)[
            [
                "project",
                "target",
                "auc",
                "balanced_acc",
                "sensitivity",
                "specificity",
                "most_chosen_model",
                "median_k",
                "perm_p",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
