#!/usr/bin/env python3
"""
Robust Multi-Calibration Bootstrap Venn-Abers (COMPLETE PIPELINE - FINAL V6)
"""

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, roc_auc_score, f1_score, brier_score_loss, confusion_matrix
from VennABERS import ScoresToMultiProbs
import ghostml
from joblib import Parallel, delayed

# --- Configuration -
# 1. churn-uplift-mlg DONE
# FOLDER_PATH = "/mnt/c/Users/bebek/Desktop/Master/Thesis/churn-uplift-mlg"
# DATA_FILE = "churn_uplift_anonymized.csv"
# TARGET_COLUMN = "y"

# 2. mobile churn DONE
# FOLDER_PATH = "/mnt/c/Users/bebek/Desktop/Master/Thesis/mobile_churn"
# DATA_FILE = "dataset_.arff"
# TARGET_COLUMN = "churn"


# 3. Iranian churn classification  DONE
FOLDER_PATH = "/mnt/c/Users/bebek/Desktop/Master/Thesis/iranian-churn"
DATA_FILE = "Customer Churn.csv"
TARGET_COLUMN = "Churn" 

# 4. Customer churn classification  DONE
# FOLDER_PATH = "/mnt/c/Users/bebek/Desktop/Master/Thesis/customer_churn_classification"
# DATA_FILE = "dataset_.arff"
# TARGET_COLUMN = "exited"

RANDOM_STATE = 14
RESULTS_DIR = "/mnt/c/Users/bebek/Desktop/Master/Thesis/OUTPUTS/va_bootstraps_xgboost"
USE_GPU = True
N_CALIBRATION_SPLITS = 5
CALIBRATION_SET_SIZE_RATIO = 0.3
N_BOOTSTRAP_REPLICAS = 500
PARALLEL_N_JOBS = -1

try:
    xgb.XGBClassifier(tree_method='hist', device='cuda')
    GPU_AVAILABLE = True
    print("XGBoost with GPU support proceeding.")
except Exception:
    GPU_AVAILABLE = False
    print("XGBoost with GPU support not available, using CPU.")

def set_seeds(seed=RANDOM_STATE):
    """Sets random seeds for reproducibility across all libraries."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_and_preprocess_data(file_path, target_col):
    """Loads and preprocesses data from CSV or ARFF files with target encoding."""
    if file_path.endswith('.arff'):
        try:
            from scipy.io import arff
            data, meta = arff.loadarff(file_path)
            df = pd.DataFrame(data)
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.decode('utf-8')
        except ImportError:
             df = pd.read_csv(file_path, sep=None, engine='python')
    else:
        df = pd.read_csv(file_path)
    
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    df = df.drop(columns=id_cols, errors='ignore')
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != target_col and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    if target_col in df.columns:
        unique_vals = pd.Series(df[target_col].unique()).dropna()
        if len(unique_vals) == 2:
            val1, val2 = unique_vals
            df[target_col] = df[target_col].map({val1: 0, val2: 1}).astype(int)
    return df

def create_preprocessor(df, target_col):
    """Creates a ColumnTransformer for numerical and categorical features."""
    features = [col for col in df.columns if col != target_col]
    num_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in df[features].select_dtypes(exclude=[np.number]).columns if df[col].nunique() <= 50]
    return ColumnTransformer([('num', StandardScaler(), num_cols), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)], remainder='drop')

def remove_outliers(df, target_col, contamination=0.1):
    """Removes outliers using EllipticEnvelope for numerical features."""
    from sklearn.covariance import EllipticEnvelope
    num_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]
    if len(num_cols) > 1:
        outlier_mask = EllipticEnvelope(contamination=contamination, random_state=RANDOM_STATE).fit_predict(df[num_cols]) == -1
        print(f"   - Removing {outlier_mask.sum()} outliers ({outlier_mask.sum()/len(df)*100:.1f}%)")
        return df[~outlier_mask].reset_index(drop=True)
    return df

def train_xgboost(X_train, y_train, params):
    """Trains XGBoost model with given parameters and GPU support if available."""
    model_params = {'random_state': RANDOM_STATE, 'objective': 'binary:logistic', **params}
    if USE_GPU and GPU_AVAILABLE:
        model_params.update({'tree_method': 'hist', 'device': 'cuda'})
    return xgb.XGBClassifier(**model_params).fit(X_train, y_train, verbose=False)

def tune_xgboost(X_train, y_train):
    """Tunes XGBoost hyperparameters using RandomizedSearchCV."""
    print("Tuning XGBoost hyperparameters.")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
    param_grid = {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1], 'n_estimators': [200, 300], 'scale_pos_weight': [scale_pos_weight]}
    model_params = {'random_state': RANDOM_STATE, 'eval_metric': 'logloss', 'verbosity': 0, 'objective': 'binary:logistic'}
    if USE_GPU and GPU_AVAILABLE: model_params.update({'tree_method': 'hist', 'device': 'cuda'})
    rs = RandomizedSearchCV(xgb.XGBClassifier(**model_params), param_grid, n_iter=4, cv=3, scoring='roc_auc', random_state=RANDOM_STATE, n_jobs=4)
    rs.fit(X_train, y_train)
    print("XGBoost tuning params complete.")
    return rs.best_params_

def evaluate_model(y_true, y_pred_proba, model_name="Model", threshold=0.5):
    """Evaluates model performance with multiple metrics including AUC, F1, and Recall."""
    y_pred_class = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    metrics = {'Log Loss': log_loss(y_true, np.clip(y_pred_proba, 1e-15, 1-1e-15)), 'AUC': roc_auc_score(y_true, y_pred_proba), 'F1': f1_score(y_true, y_pred_class), 'Brier': brier_score_loss(y_true, y_pred_proba), 'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0, 'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0}
    print(f"   - {model_name}: AUC={metrics['AUC']:.4f}, F1={metrics['F1']:.4f}, Recall={metrics['Recall']:.4f}")
    return metrics

def compute_va_calibration(model, X_cal, y_cal, X_pred):
    """Computes Venn-Abers calibration using ScoresToMultiProbs."""
    cal_scores = model.predict_proba(X_cal)[:, 1]
    pred_scores = model.predict_proba(X_pred)[:, 1]
    cal_pts = list(zip(cal_scores.astype(float), y_cal.astype(int)))
    p0, p1 = ScoresToMultiProbs(cal_pts, pred_scores.tolist())
    return np.array(p0), np.array(p1)

def _train_single_bootstrap_iteration(args):
    """Trains a single bootstrap iteration and returns VA calibration results."""
    X_tr, y_tr, X_cal, y_cal, X_val, X_test, model_params, seed = args
    np.random.seed(seed)
    boot_indices = np.random.choice(len(X_tr), size=len(X_tr), replace=True)
    model = train_xgboost(X_tr[boot_indices], y_tr[boot_indices], model_params)
    p0_val, p1_val = compute_va_calibration(model, X_cal, y_cal, X_val)
    p0_test, p1_test = compute_va_calibration(model, X_cal, y_cal, X_test)
    return (p0_val, p1_val, p0_test, p1_test)

def compute_robust_bootstrap_va(X_train, y_train, X_val, X_test, params):
    """Computes robust bootstrap Venn-Abers with multiple calibration splits and bootstrap replicas."""
    total_runs = N_CALIBRATION_SPLITS * N_BOOTSTRAP_REPLICAS
    print(f"Starting BA-IVAP s ({total_runs} total runs)...")
    job_args = []
    for i in range(N_CALIBRATION_SPLITS):
        X_tr, X_cal, y_tr, y_cal = train_test_split(X_train, y_train, test_size=CALIBRATION_SET_SIZE_RATIO, random_state=RANDOM_STATE + i, stratify=y_train)
        for b in range(N_BOOTSTRAP_REPLICAS):
            job_args.append((X_tr, y_tr, X_cal, y_cal, X_val, X_test, params, RANDOM_STATE + i * N_BOOTSTRAP_REPLICAS + b))
    
    results = Parallel(n_jobs=PARALLEL_N_JOBS, verbose=10)(delayed(_train_single_bootstrap_iteration)(args) for args in job_args)
    all_p0_val, all_p1_val, all_p0_test, all_p1_test = zip(*results)

    def process_results(all_p0, all_p1):
        """Processes bootstrap results to compute envelope bounds and width metrics."""
        pool_p0, pool_p1 = np.vstack(all_p0), np.vstack(all_p1)
        env_p0, env_p1 = np.min(pool_p0, axis=0), np.max(pool_p1, axis=0)
        total_width = env_p1 - env_p0
        mean_va_width = np.mean(pool_p1 - pool_p0, axis=0)
        epsilon = 1e-9
        individual_merges = [np.clip(p1 / (1.0 - p0 + p1 + epsilon), epsilon, 1 - epsilon) for p0, p1 in zip(all_p0, all_p1)]
        p_merged = np.mean(np.vstack(individual_merges), axis=0)
        return {'p_merged': p_merged, 'envelope_p0': env_p0, 'envelope_p1': env_p1, 'total_width': total_width, 'mean_va_width': mean_va_width, 'bootstrap_width': total_width - mean_va_width}

    print("\n✅ Aggregating final results...")
    return process_results(all_p0_val, all_p1_val), process_results(all_p0_test, all_p1_test)

def create_results_df(y_true, indices, pred_results, prefix, va_standard_width=None):
    """Creates DataFrame with prediction results and width metrics."""
    true_col = 'true_class' if prefix == 'val' else f'true_{TARGET_COLUMN.lower()}'
    df = pd.DataFrame({
        'original_index': indices, true_col: y_true, f'mean_pred_proba_{prefix}': pred_results['p_merged'],
        'pi_lower_bound_p_target': pred_results['envelope_p0'], 'pi_upper_bound_p_target': pred_results['envelope_p1'],
        'total_width': pred_results['total_width'], 'mean_va_width': pred_results['mean_va_width'],
        'bootstrap_width': pred_results['bootstrap_width']
    })
    if va_standard_width is not None: df['va_standard_width'] = va_standard_width
    return df

def main():
    """Main pipeline that runs all model variants and saves results."""
    start_time = time.time()
    set_seeds()
    results_path = Path(RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)
    results_summary = []

    print("\n--- 1. DATA SETUP (Correct 60/20/20 Split) ---")
    df = load_and_preprocess_data(os.path.join(FOLDER_PATH, DATA_FILE), TARGET_COLUMN)
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN])
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=RANDOM_STATE, stratify=train_val_df[TARGET_COLUMN])
    y_train, y_val, y_test = train_df[TARGET_COLUMN].values, val_df[TARGET_COLUMN].values, test_df[TARGET_COLUMN].values
    val_indices, test_indices = val_df.index, test_df.index
    preprocessor = create_preprocessor(train_df, TARGET_COLUMN).fit(train_df.drop(columns=[TARGET_COLUMN]))
    X_train_t, X_val_t, X_test_t = preprocessor.transform(train_df.drop(columns=[TARGET_COLUMN])), preprocessor.transform(val_df.drop(columns=[TARGET_COLUMN])), preprocessor.transform(test_df.drop(columns=[TARGET_COLUMN]))
    print(f"📊 Final Split Sizes: Train={len(X_train_t)}, Validation={len(X_val_t)}, Test={len(X_test_t)}")
    
    print("\n--- 2. MODEL TUNING ---")
    params = tune_xgboost(X_train_t, y_train)

    print("\n--- 3. MODELING & EVALUATION ---")
    
    print("\n▶️  1/7: Baseline XGBoost")
    base_model = train_xgboost(X_train_t, y_train, params)
    base_pred_val, base_pred_test = base_model.predict_proba(X_val_t)[:, 1], base_model.predict_proba(X_test_t)[:, 1]
    results_summary.append({'Method': 'Baseline XGB', **evaluate_model(y_test, base_pred_test, "Baseline")})

    print("\n▶️  2/7: Outlier-Robust XGBoost")
    train_clean_df = remove_outliers(train_df.copy(), TARGET_COLUMN)
    X_train_clean_t, y_train_clean = preprocessor.transform(train_clean_df.drop(columns=[TARGET_COLUMN])), train_clean_df[TARGET_COLUMN].values
    robust_model = train_xgboost(X_train_clean_t, y_train_clean, params)
    results_summary.append({'Method': 'Outlier-Robust XGB', **evaluate_model(y_test, robust_model.predict_proba(X_test_t)[:, 1], "Outlier-Robust")})

    print("\n▶️  3/7: Standard VA (Single IVAP)")
    X_tr_std, X_cal_std, y_tr_std, y_cal_std = train_test_split(X_train_t, y_train, test_size=0.3, random_state=RANDOM_STATE, stratify=y_train)
    std_va_model = train_xgboost(X_tr_std, y_tr_std, params)
    std_p0_val, std_p1_val = compute_va_calibration(std_va_model, X_cal_std, y_cal_std, X_val_t)
    std_p0_test, std_p1_test = compute_va_calibration(std_va_model, X_cal_std, y_cal_std, X_test_t)
    std_p_merged_test = np.clip(std_p1_test / (1.0 - std_p0_test + std_p1_test + 1e-9), 1e-9, 1-1e-9)
    metrics = evaluate_model(y_test, std_p_merged_test, "Standard VA")
    results_summary.append({'Method': 'Standard VA XGB', **metrics, 'Avg Total Width': (std_p1_test - std_p0_test).mean()})
    
    print("\n▶️  4/7: BA-IVAP")
    val_results, test_results = compute_robust_bootstrap_va(X_train_t, y_train, X_val_t, X_test_t, params)
    metrics = evaluate_model(y_test, test_results['p_merged'], "BA-IVAP")
    results_summary.append({'Method': 'BA-IVAP XGB', **metrics, 'Avg Total Width': test_results['total_width'].mean(), 'Avg BA-IVAP Width': test_results['mean_va_width'].mean(), 'Avg Bootstrap Width': test_results['bootstrap_width'].mean()})
    
    print("\n▶️  5/7: Outlier-Robust BA-IVAP")
    robust_val_results, robust_test_results = compute_robust_bootstrap_va(X_train_clean_t, y_train_clean, X_val_t, X_test_t, params)
    metrics = evaluate_model(y_test, robust_test_results['p_merged'], "Outlier-Robust BA-IVAP")
    results_summary.append({'Method': 'Outlier-Robust BA-IVAP XGB', **metrics, 'Avg Total Width': robust_test_results['total_width'].mean(), 'Avg BA-IVAP Width': robust_test_results['mean_va_width'].mean(), 'Avg Bootstrap Width': robust_test_results['bootstrap_width'].mean()})
    
    print("\n--- GHOST Optimization & Evaluation ---")
    X_tr_g, X_cal_g, y_tr_g, y_cal_g = train_test_split(X_train_t, y_train, test_size=0.3, random_state=RANDOM_STATE+1, stratify=y_train)
    
    print("\n▶️  6/7: Baseline + GHOST")
    ghost_base_model = train_xgboost(X_tr_g, y_tr_g, params)
    baseline_ghost_thr = 0.5
    try: baseline_ghost_thr = ghostml.optimize_threshold_from_predictions(y_cal_g, ghost_base_model.predict_proba(X_cal_g)[:, 1], ThOpt_metrics='Kappa', thresholds=np.linspace(0.01, 0.99, 99))
    except Exception as e: print(f"   - ⚠️ GHOST optimization for Baseline failed: {e}. Defaulting to 0.5.")
    results_summary.append({'Method': 'Baseline + GHOST', **evaluate_model(y_test, base_pred_test, "Baseline + GHOST", threshold=baseline_ghost_thr), 'GHOST Threshold': baseline_ghost_thr})

    print("\n▶️  7/7: BA-IVAP + GHOST (Practical Shortcut)")
    std_va_ghost_model = train_xgboost(X_tr_g, y_tr_g, params)
    p0_cal_g, p1_cal_g = compute_va_calibration(std_va_ghost_model, X_cal_g, y_cal_g, X_cal_g)
    p_merged_ghost_cal = np.clip(p1_cal_g / (1.0 - p0_cal_g + p1_cal_g + 1e-9), 1e-9, 1-1e-9)
    va_ghost_thr = 0.5
    try: va_ghost_thr = ghostml.optimize_threshold_from_predictions(y_cal_g, p_merged_ghost_cal, ThOpt_metrics='Kappa', thresholds=np.linspace(0.01, 0.99, 99))
    except Exception as e: print(f"   - ⚠️ GHOST optimization for BA-IVAP failed: {e}. Defaulting to 0.5.")
    metrics = evaluate_model(y_test, test_results['p_merged'], "BA-IVAP + GHOST", threshold=va_ghost_thr)
    results_summary.append({'Method': 'BA-IVAP + GHOST', **metrics, 'GHOST Threshold': va_ghost_thr})

    print("\n--- 4. SAVING ALL OUTPUT FILES ---")
    df_val_final = create_results_df(y_val, val_indices, val_results, prefix='val', va_standard_width=std_p1_val - std_p0_val)
    df_test_final = create_results_df(y_test, test_indices, test_results, prefix='test', va_standard_width=std_p1_test - std_p0_test)
    df_val_final.to_csv(results_path / "xgboost_standard_bootstrap_va_predictions.csv", index=False)
    df_test_final.to_csv(results_path / "xgboost_standard_bootstrap_va_test_predictions.csv", index=False)
    pd.DataFrame({'original_index': val_indices, 'true_class': y_val, 'pred_proba': base_pred_val, 'pred_class': (base_pred_val >= 0.5).astype(int)}).to_csv(results_path / "xgboost_baseline_predictions.csv", index=False)
    pd.DataFrame({'original_index': test_indices, f'true_{TARGET_COLUMN.lower()}': y_test, 'pred_proba': base_pred_test, 'pred_class': (base_pred_test >= 0.5).astype(int)}).to_csv(results_path / "xgboost_baseline_test_predictions.csv", index=False)
    pd.DataFrame({'original_index': val_indices, 'true_class': y_val, 'pred_class': (base_pred_val >= baseline_ghost_thr).astype(int)}).to_csv(results_path / "xgboost_baseline_ghost_predictions.csv", index=False)
    pd.DataFrame({'original_index': test_indices, f'true_{TARGET_COLUMN.lower()}': y_test, 'pred_class': (base_pred_test >= baseline_ghost_thr).astype(int)}).to_csv(results_path / "xgboost_baseline_ghost_test_predictions.csv", index=False)
    pd.DataFrame({'original_index': val_indices, 'true_class': y_val, 'pred_class_ghost': (df_val_final['mean_pred_proba_val'] >= va_ghost_thr).astype(int)}).to_csv(results_path / "xgboost_standard_bootstrap_va_ghost_predictions.csv", index=False)
    pd.DataFrame({'original_index': test_indices, f'true_{TARGET_COLUMN.lower()}': y_test, 'pred_class_ghost': (df_test_final['mean_pred_proba_test'] >= va_ghost_thr).astype(int)}).to_csv(results_path / "xgboost_standard_bootstrap_va_ghost_test_predictions.csv", index=False)
    print("✅ All prediction files generated.")

    print("\n" + "="*80); print("📊 FINAL MODEL COMPARISON SUMMARY"); print("="*80)
    summary_df = pd.DataFrame(results_summary)
    
    METRIC_DISPLAY_NAMES = {
        'Method': 'Method', 'AUC': 'AUC', 'F1': 'F1', 'Recall': 'Recall', 'Precision': 'Precision',
        'Log Loss': 'Log-loss', 'Brier': 'Brier', 'GHOST Threshold': 'GHOST Threshold',
        'Avg Total Width': 'Avg Total Width', 'Avg BA-IVAP Width': 'Avg BA-IVAP Width', 'Avg Bootstrap Width': 'Avg Bootstrap Width'
    }
    summary_df.rename(columns=METRIC_DISPLAY_NAMES, inplace=True)
    
    cols_order = list(METRIC_DISPLAY_NAMES.values())
    summary_df = summary_df.reindex(columns=[col for col in cols_order if col in summary_df.columns])
    
    print(summary_df.to_string(index=False))
    summary_df.to_csv(results_path / "model_comparison_summary.csv", index=False)
    print(f"\n⏱️  Total time: {time.time() - start_time:.1f} seconds")
    print(f"📁 Results saved to: {results_path.resolve()}")

if __name__ == "__main__":
    main()