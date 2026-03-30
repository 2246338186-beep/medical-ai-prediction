import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier # type: ignore
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


PHYSIOLOGIC_RANGES = {
    "HR": (30, 180),
    "SpO2": (70, 100),
    "NIBP_SBP": (50, 250),
    "NIBP_DBP": (30, 150),
    "RR": (4, 40),
    "PR": (30, 180),
    "CO2_EtFi": (0, 80),
    "O2_EtFi": (0, 100),
    "Sev": (0, 8),
}

CONTINUOUS_COLS = ["HR", "SpO2", "RR", "PR", "CO2_EtFi", "O2_EtFi", "Sev"]
FEATURE_SIGNAL_COLS = [
    "HR",
    "SpO2",
    "NIBP_SBP_feat",
    "NIBP_DBP_feat",
    "MAP_feat",
    "RR",
    "PR",
    "CO2_EtFi",
    "O2_EtFi",
    "Sev",
    "PP_feat",
    "time_since_last_bp",
]


@dataclass
class Config:
    input_csv: str
    output_dir: str
    window_in: int = 10
    window_out: int = 5
    hypotension_map_threshold: float = 65.0
    cont_interp_limit: int = 2
    cont_ffill_limit: int = 2
    cont_bfill_limit: int = 1
    bp_ffill_limit: int = 3
    random_state: int = 42
    use_xgboost: bool = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "SPO2" in df.columns:
        df = df.rename(columns={"SPO2": "SpO2"})
    required = [
        "case_id",
        "time",
        "HR",
        "SpO2",
        "NIBP_SBP",
        "NIBP_DBP",
        "RR",
        "CO2_EtFi",
        "O2_EtFi",
        "Sev",
        "PR",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    return df


def clip_outliers_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, (low, high) in PHYSIOLOGIC_RANGES.items():
        if col in out.columns:
            out.loc[(out[col] < low) | (out[col] > high), col] = np.nan
    return out


def make_full_time_index(case_df: pd.DataFrame) -> pd.DataFrame:
    case_df = case_df.sort_values("time").drop_duplicates(subset=["time"], keep="first").copy()
    case_df["time"] = pd.to_numeric(case_df["time"], errors="coerce")
    case_df = case_df.dropna(subset=["time"])
    case_df["time"] = case_df["time"].astype(int)
    case_id_val = case_df["case_id"].iloc[0]

    full_time = pd.DataFrame({"time": np.arange(case_df["time"].min(), case_df["time"].max() + 1)})
    case_df_indexed = case_df.set_index("time")
    merged = full_time.join(case_df_indexed, on="time", how="left")
    merged["case_id"] = case_id_val
    return merged


def compute_time_since_last_valid_bp(df: pd.DataFrame) -> pd.Series:
    last_valid_time = None
    values = []
    for _, row in df.iterrows():
        current_time = int(row["time"])
        is_measured = pd.notna(row["NIBP_SBP_raw"]) and pd.notna(row["NIBP_DBP_raw"])

        if is_measured:
            last_valid_time = current_time
            values.append(0.0)
        else:
            if last_valid_time is None:
                values.append(np.nan)
            else:
                time_diff = current_time - last_valid_time
                values.append(float(time_diff))
    return pd.Series(values, index=df.index, dtype=float)


def preprocess_single_case(case_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    case_df = make_full_time_index(case_df)
    case_df = clip_outliers_to_nan(case_df)

    for col in CONTINUOUS_COLS:
        series = pd.to_numeric(case_df[col], errors="coerce")
        series = series.interpolate(method="linear", limit=cfg.cont_interp_limit, limit_area="inside")
        series = series.ffill(limit=cfg.cont_ffill_limit)
        series = series.bfill(limit=cfg.cont_bfill_limit)
        case_df[col] = series

    case_df["NIBP_SBP_raw"] = pd.to_numeric(case_df["NIBP_SBP"], errors="coerce")
    case_df["NIBP_DBP_raw"] = pd.to_numeric(case_df["NIBP_DBP"], errors="coerce")

    case_df["bp_measured_flag"] = (
        case_df["NIBP_SBP_raw"].notna() & case_df["NIBP_DBP_raw"].notna()
    ).astype(int)
    case_df["time_since_last_bp"] = compute_time_since_last_valid_bp(case_df)

    case_df["NIBP_SBP_feat"] = case_df["NIBP_SBP_raw"].ffill(limit=cfg.bp_ffill_limit)
    case_df["NIBP_DBP_feat"] = case_df["NIBP_DBP_raw"].ffill(limit=cfg.bp_ffill_limit)

    case_df["MAP_raw"] = (case_df["NIBP_SBP_raw"] + 2 * case_df["NIBP_DBP_raw"]) / 3.0
    case_df["MAP_feat"] = (case_df["NIBP_SBP_feat"] + 2 * case_df["NIBP_DBP_feat"]) / 3.0
    case_df["PP_feat"] = case_df["NIBP_SBP_feat"] - case_df["NIBP_DBP_feat"]
    case_df["HR_PR_diff"] = case_df["HR"] - case_df["PR"]

    for col in CONTINUOUS_COLS:
        raw_missing_col = f"{col}_missing_flag"
        case_df[raw_missing_col] = case_df[col].isna().astype(int)

    return case_df


def preprocess_all_cases(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    processed = []
    for case_id, case_df in df.groupby("case_id", sort=True):
        case_df = case_df.copy()
        case_df["case_id"] = case_id
        processed.append(preprocess_single_case(case_df, cfg))
    out = pd.concat(processed, ignore_index=True)
    out = out.sort_values(["case_id", "time"]).reset_index(drop=True)
    return out


def safe_last_valid(series: pd.Series) -> float:
    x = series.dropna()
    return float(x.iloc[-1]) if len(x) > 0 else np.nan


def safe_first_valid(series: pd.Series) -> float:
    x = series.dropna()
    return float(x.iloc[0]) if len(x) > 0 else np.nan


def safe_stat(series: pd.Series, func, default=np.nan) -> float:
    x = series.dropna().astype(float)
    if len(x) == 0:
        return default
    return float(func(x))


def calc_slope(series: pd.Series) -> float:
    y = pd.to_numeric(series, errors="coerce").astype(float)
    mask = y.notna().values
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(y))[mask]
    yv = y.values[mask]
    slope = np.polyfit(x, yv, 1)[0]
    return float(slope)


def extract_features(hist_df: pd.DataFrame) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    for col in FEATURE_SIGNAL_COLS:
        x = pd.to_numeric(hist_df[col], errors="coerce")

        feats[f"{col}_mean"] = safe_stat(x, np.mean)
        feats[f"{col}_median"] = safe_stat(x, np.median)
        feats[f"{col}_last"] = safe_last_valid(x)
        feats[f"{col}_min"] = safe_stat(x, np.min)
        feats[f"{col}_max"] = safe_stat(x, np.max)
        feats[f"{col}_std"] = safe_stat(x, lambda a: np.std(a, ddof=0))
        feats[f"{col}_range"] = (
            feats[f"{col}_max"] - feats[f"{col}_min"]
            if not np.isnan(feats[f"{col}_max"]) and not np.isnan(feats[f"{col}_min"])
            else np.nan
        )
        mean_val = feats[f"{col}_mean"]
        std_val = feats[f"{col}_std"]
        feats[f"{col}_cv"] = std_val / mean_val if mean_val not in [0, np.nan] and pd.notna(mean_val) else np.nan
        feats[f"{col}_slope"] = calc_slope(x)
        first_val = safe_first_valid(x)
        last_val = safe_last_valid(x)
        feats[f"{col}_delta"] = last_val - first_val if pd.notna(first_val) and pd.notna(last_val) else np.nan

        recent = x.iloc[-3:]
        early = x.iloc[:-3]
        recent_mean = safe_stat(recent, np.mean)
        early_mean = safe_stat(early, np.mean)
        feats[f"{col}_recent_mean"] = recent_mean
        feats[f"{col}_early_mean"] = early_mean
        feats[f"{col}_recent_early_diff"] = (
            recent_mean - early_mean if pd.notna(recent_mean) and pd.notna(early_mean) else np.nan
        )
        feats[f"{col}_missing_ratio"] = float(x.isna().mean())

    feats["bp_measured_ratio"] = float(hist_df["bp_measured_flag"].mean())
    feats["HR_PR_diff_mean"] = safe_stat(hist_df["HR_PR_diff"], np.mean)
    feats["HR_PR_diff_last"] = safe_last_valid(hist_df["HR_PR_diff"])

    return feats


def build_supervised_dataset(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for case_id, case_df in df.groupby("case_id", sort=True):
        case_df = case_df.sort_values("time").reset_index(drop=True)
        n = len(case_df)
        for t in range(cfg.window_in - 1, n - cfg.window_out):
            hist_df = case_df.iloc[t - cfg.window_in + 1 : t + 1].copy()
            fut_df = case_df.iloc[t + 1 : t + 1 + cfg.window_out].copy()

            future_map_raw = pd.to_numeric(fut_df["MAP_raw"], errors="coerce")
            if future_map_raw.notna().sum() == 0:
                continue

            label = int((future_map_raw < cfg.hypotension_map_threshold).any())
            feats = extract_features(hist_df)
            rows.append(
                {
                    "case_id": case_id,
                    "sample_time": int(case_df.iloc[t]["time"]),
                    "label": label,
                    **feats,
                }
            )
    dataset = pd.DataFrame(rows)
    return dataset


def safe_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    denom = tn + fp
    return float(tn / denom) if denom > 0 else np.nan


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_prob))


def make_models(random_state: int, y_train: Optional[pd.Series] = None, use_xgboost: bool = False) -> Dict[str, Pipeline]:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                slice(0, None),
            )
        ],
        remainder="drop",
    )

    models: Dict[str, Pipeline] = {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocess", clone(preprocess)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocess", clone(preprocess)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=6,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if use_xgboost and HAS_XGBOOST:
        scale_pos_weight = 1.0
        if y_train is not None:
            pos = max(int((y_train == 1).sum()), 1)
            neg = max(int((y_train == 0).sum()), 1)
            scale_pos_weight = neg / pos
        models["XGBoost"] = Pipeline(
            steps=[
                ("preprocess", clone(preprocess)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return models


def find_hypotension_events(case_df: pd.DataFrame, threshold: float) -> List[Tuple[int, int]]:
    event_mask = pd.to_numeric(case_df["MAP_raw"], errors="coerce") < threshold
    times = case_df["time"].astype(int).tolist()
    events: List[Tuple[int, int]] = []
    start: Optional[int] = None
    prev_time: Optional[int] = None

    for flag, time_val in zip(event_mask.fillna(False).tolist(), times):
        if flag:
            if start is None:
                start = time_val
            elif prev_time is not None and time_val != prev_time + 1:
                events.append((start, prev_time))
                start = time_val
        else:
            if start is not None and prev_time is not None:
                events.append((start, prev_time))
                start = None
        prev_time = time_val

    if start is not None and prev_time is not None:
        events.append((start, prev_time))
    return events


def compute_event_capture_rate(
    case_df: pd.DataFrame,
    sample_times: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    lookback: int,
) -> float:
    events = find_hypotension_events(case_df, threshold)
    if len(events) == 0:
        return np.nan
    captured = 0
    sample_times = np.asarray(sample_times)
    y_prob = np.asarray(y_prob)
    for event_start, _ in events:
        mask = (sample_times >= event_start - lookback) & (sample_times < event_start)
        if mask.any() and (y_prob[mask] >= 0.5).any():
            captured += 1
    return float(captured / len(events))


def run_loso_cv(dataset: pd.DataFrame, processed_df: pd.DataFrame, cfg: Config):
    meta_cols = ["case_id", "sample_time", "label"]
    feature_cols = [c for c in dataset.columns if c not in meta_cols]

    X = dataset[feature_cols].copy()
    y = dataset["label"].astype(int).copy()
    groups = dataset["case_id"].copy()

    fold_metrics: List[Dict[str, float]] = []
    predictions: List[pd.DataFrame] = []

    for test_case in sorted(groups.unique()):
        train_idx = groups != test_case
        test_idx = groups == test_case

        X_train = X.loc[train_idx].reset_index(drop=True)
        y_train = y.loc[train_idx].reset_index(drop=True)
        X_test = X.loc[test_idx].reset_index(drop=True)
        y_test = y.loc[test_idx].reset_index(drop=True)
        test_meta = dataset.loc[test_idx, ["case_id", "sample_time", "label"]].reset_index(drop=True)

        if y_train.nunique() < 2 or y_test.empty:
            continue

        models = make_models(cfg.random_state, y_train, cfg.use_xgboost)

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)
                y_prob = 1.0 / (1.0 + np.exp(-y_prob))
            y_pred = (y_prob >= 0.5).astype(int)

            case_df = processed_df[processed_df["case_id"] == test_case].sort_values("time").reset_index(drop=True)
            event_capture = compute_event_capture_rate(
                case_df=case_df,
                sample_times=test_meta["sample_time"].values,
                y_prob=y_prob,
                threshold=cfg.hypotension_map_threshold,
                lookback=cfg.window_out,
            )

            fold_metrics.append(
                {
                    "model": model_name,
                    "test_case": int(test_case),
                    "n_test": int(len(y_test)),
                    "positive_rate": float(y_test.mean()),
                    "AUROC": safe_roc_auc(y_test.values, y_prob),
                    "AUPRC": safe_auprc(y_test.values, y_prob),
                    "Accuracy": float(accuracy_score(y_test, y_pred)),
                    "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
                    "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
                    "F1": float(f1_score(y_test, y_pred, zero_division=0)),
                    "Specificity": safe_specificity(y_test.values, y_pred),
                    "EventCaptureRate": event_capture,
                }
            )

            pred_df = test_meta.copy()
            pred_df["model"] = model_name
            pred_df["y_prob"] = y_prob
            pred_df["y_pred"] = y_pred
            predictions.append(pred_df)

    fold_df = pd.DataFrame(fold_metrics)
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()

    if fold_df.empty:
        summary_df = pd.DataFrame()
    else:
        metric_cols = [
            "AUROC",
            "AUPRC",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Specificity",
            "EventCaptureRate",
        ]
        summary_df = fold_df.groupby("model")[metric_cols].agg(["mean", "std"]).reset_index()
        summary_df.columns = [
            "model" if col == ("model", "") else f"{col[0]}_{col[1]}".rstrip("_")
            for col in summary_df.columns.to_flat_index()
        ]
    return fold_df, summary_df, pred_df


def save_config(cfg: Config, output_dir: str) -> None:
    cfg_path = os.path.join(output_dir, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Intraoperative hypotension early warning pipeline")
    # ====================== 已修改路径 ======================
    parser.add_argument("--input_csv", type=str,
                        default="D:/IntraoperativeHypotensionPrediction/data/all_cases_data.csv",
                        help="Path to input CSV")
    # =======================================================
    parser.add_argument("--output_dir", type=str, default="outputs_intraop", help="Directory to save outputs")
    parser.add_argument("--window_in", type=int, default=10, help="History window length in minutes")
    parser.add_argument("--window_out", type=int, default=5, help="Prediction horizon in minutes")
    parser.add_argument("--hypotension_map_threshold", type=float, default=65.0, help="MAP threshold")
    parser.add_argument("--cont_interp_limit", type=int, default=2, help="Continuous-variable interpolation limit")
    parser.add_argument("--cont_ffill_limit", type=int, default=2, help="Continuous-variable forward fill limit")
    parser.add_argument("--cont_bfill_limit", type=int, default=1, help="Continuous-variable backward fill limit")
    parser.add_argument("--bp_ffill_limit", type=int, default=3, help="NIBP forward fill limit for features")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--use_xgboost", action="store_true", help="Also train XGBoost if installed")
    args = parser.parse_args()

    cfg = Config(**vars(args))
    ensure_dir(cfg.output_dir)
    save_config(cfg, cfg.output_dir)

    print("[1/5] Loading data...")
    df = load_data(cfg.input_csv)
    print(f"Raw shape: {df.shape}")

    print("[2/5] Preprocessing data...")
    processed_df = preprocess_all_cases(df, cfg)
    processed_path = os.path.join(cfg.output_dir, "processed_data.csv")
    processed_df.to_csv(processed_path, index=False, encoding="utf-8-sig")
    print(f"Processed shape: {processed_df.shape}")

    print("[3/5] Building supervised dataset...")
    dataset = build_supervised_dataset(processed_df, cfg)
    if dataset.empty:
        raise RuntimeError("No supervised samples were generated. Check the time axis or label settings.")
    dataset_path = os.path.join(cfg.output_dir, "feature_dataset.csv")
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    print(f"Feature dataset shape: {dataset.shape}")
    print(f"Positive rate: {dataset['label'].mean():.4f}")

    print("[4/5] Running LOSO training/evaluation...")
    fold_df, summary_df, pred_df = run_loso_cv(dataset, processed_df, cfg)
    fold_path = os.path.join(cfg.output_dir, "loso_fold_metrics.csv")
    summary_path = os.path.join(cfg.output_dir, "loso_summary_metrics.csv")
    pred_path = os.path.join(cfg.output_dir, "test_predictions.csv")
    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print("[5/5] Done.")
    print("\n===== Summary metrics =====")
    if summary_df.empty:
        print("No valid CV results were produced.")
    else:
        print(summary_df.to_string(index=False))

    print("\nSaved files:")
    for p in [processed_path, dataset_path, fold_path, summary_path, pred_path, os.path.join(cfg.output_dir, "run_config.json")]:
        print(f"- {p}")


if __name__ == "__main__":
    main()