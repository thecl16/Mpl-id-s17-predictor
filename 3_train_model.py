"""
=============================================================
BAGIAN 3: MODEL TRAINING — XGBoost Classifier
=============================================================
Input:  data/dataset_siap_ml.csv
Output: models/model_mpl_s17.pkl
        models/training_report.json
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler

# Try XGBoost, fall back to sklearn GradientBoosting
try:
    import xgboost as xgb
    USE_XGBOOST = True
    print("[INFO] XGBoost tersedia ✓")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGBOOST = False
    print("[INFO] XGBoost tidak ditemukan — menggunakan GradientBoostingClassifier (sklearn)")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = "data/dataset_siap_ml.csv"
MODEL_PATH   = "models/model_mpl_s17.pkl"
REPORT_PATH  = "models/training_report.json"
PLOT_DIR     = "models/"

FEATURE_COLS = [
    "a_rolling_win_rate", "b_rolling_win_rate", "diff_rolling_wr",
    "a_h2h_win_rate",     "b_h2h_win_rate",
    "a_avg_score_diff",   "b_avg_score_diff",   "diff_score",
    "a_win_streak",       "b_win_streak",       "diff_streak",
    "a_team_encoded",     "b_team_encoded",
]
LABEL_COL = "label"


# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
def load_dataset(path: str):
    df = pd.read_csv(path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[WARNING] Missing feature columns (will skip): {missing}")
    X = df[available]
    y = df[LABEL_COL]
    print(f"[LOAD] {len(df)} samples | {len(available)} features | "
          f"Class balance: {y.value_counts().to_dict()}")
    return X, y, df


# ─────────────────────────────────────────────
# 2. Train XGBoost
# ─────────────────────────────────────────────
def train_model(X_train, y_train):
    if USE_XGBOOST:
        params = dict(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        model = xgb.XGBClassifier(**params)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
        print(f"[TRAIN] Best iteration: {model.best_iteration}")
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_split=5, random_state=42
        )
        model.fit(X_train, y_train)
        print("[TRAIN] GradientBoostingClassifier fitted ✓")
    return model


# ─────────────────────────────────────────────
# 3. Evaluate
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, X_all, y_all):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring="accuracy")

    print("\n" + "="*50)
    print("           MODEL EVALUATION REPORT")
    print("="*50)
    print(f"  Test Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"  CV Accuracy     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("="*50)
    print(classification_report(y_test, y_pred,
                                 target_names=["Team B Wins", "Team A Wins"]))

    return acc, auc, cm, cv_scores, report


# ─────────────────────────────────────────────
# 4. Plot Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm, save_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Team B Wins", "Team A Wins"],
                yticklabels=["Team B Wins", "Team A Wins"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — MPL ID S17 Model")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Confusion matrix → {save_path}")


# ─────────────────────────────────────────────
# 5. Plot Feature Importance
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, save_path: str):
    importance = model.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    feat_df = feat_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1a9641" if v > feat_df["importance"].median() else "#fdae61"
              for v in feat_df["importance"]]
    ax.barh(feat_df["feature"], feat_df["importance"], color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance — XGBoost Model")
    ax.axvline(feat_df["importance"].median(), color="gray", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Feature importance → {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    Path("models").mkdir(exist_ok=True)

    X, y, df = load_dataset(DATASET_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

    model = train_model(X_train, y_train)

    acc, auc, cm, cv_scores, report = evaluate(model, X_test, y_test, X, y)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": list(X.columns)}, f)
    print(f"\n[SAVED] Model → {MODEL_PATH}")

    # Save report
    training_report = {
        "test_accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": list(X.columns),
        "classification_report": report,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(training_report, f, indent=2)
    print(f"[SAVED] Report → {REPORT_PATH}")

    plot_confusion_matrix(cm, f"{PLOT_DIR}confusion_matrix.png")
    plot_feature_importance(model, list(X.columns), f"{PLOT_DIR}feature_importance.png")

    print("\n✅  Training complete! Model ready for deployment.")
    print(f"   Accuracy: {acc*100:.1f}%  |  AUC: {auc:.3f}  |  CV: {cv_scores.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
