import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from scipy.stats import t
from gekko import GEKKO
import time
import sys
import platform
import subprocess

def train_gekko_linear(X_boot, y_meas_boot, bound_boot, n_features):
    m = GEKKO(remote=False)
    n_samples = X_boot.shape[0]

    beta = [m.FV(value=0.0, name=f'b{j}') for j in range(n_features)]
    for b in beta:
        b.STATUS = 1

    eU = m.Array(m.Var, n_samples, lb=0)
    eL = m.Array(m.Var, n_samples, lb=0)

    std_y = float(np.std(y_meas_boot)) if np.std(y_meas_boot) > 0 else 1.0
    w_err = 1.0 / std_y

    for i in range(n_samples):
        y_pred_i = m.Intermediate(sum(beta[j] * X_boot[i, j] for j in range(n_features)))
        if bound_boot[i] == 'upper_only':
            m.Equation(eU[i] >= y_pred_i - y_meas_boot[i])
            m.Equation(eL[i] == 0)
        elif bound_boot[i] == 'lower_only':
            m.Equation(eL[i] >= y_meas_boot[i] - y_pred_i)
            m.Equation(eU[i] == 0)
        else:
            m.Equation(eU[i] >= y_pred_i - y_meas_boot[i])
            m.Equation(eL[i] >= y_meas_boot[i] - y_pred_i)

    m.Minimize(w_err * (m.sum(eU) + m.sum(eL)))
    m.options.IMODE = 3
    try:
        m.solve(disp=False)
        beta_vals = np.array([b.value[0] for b in beta])
    except Exception as e:
        print("GEKKO Error")
        beta_vals = np.zeros(n_features)
    return beta_vals

def compute_MACV_A(pred, y_meas, bound_type, true_y):
    pred = np.asarray(pred).ravel()
    y_meas = np.asarray(y_meas).ravel()
    true_y = np.asarray(true_y).ravel()
    bound = np.asarray(bound_type).astype(str)
    
    # Flags
    upper_flag = ((bound == 'upper_only') & (pred < y_meas)).astype(int)
    lower_flag = ((bound == 'lower_only') & (pred > y_meas)).astype(int)
    both_flag = (bound == 'both').astype(int)
    
    delta_upper = np.where(bound == 'upper_only', np.maximum(pred - y_meas, 0), 0)
    delta_lower = np.where(bound == 'lower_only', np.maximum(y_meas - pred, 0), 0)
    delta_both = np.where(bound == 'both', np.abs(true_y - y_meas), 0)
    
    delta = delta_upper + delta_lower + delta_both
    
    bb = both_flag.sum()
    tc = len(bound)
    ab = tc - bb
    kb = upper_flag.sum() + lower_flag.sum()
    
    A = kb / ab if ab > 0 else np.nan
    MACV = delta.sum() / ab if ab > 0 else np.nan
    return A, MACV
def compute_crps(preds_arr, y_true):
    k = preds_arr.shape[0]
    term1 = np.mean(np.abs(preds_arr - y_true), axis=0)
    preds_sorted = np.sort(preds_arr, axis=0)
    weights = (2 * np.arange(k) - k + 1)[:, None]
    term2 = np.sum(weights * preds_sorted, axis=0) / (k**2)
    
    crps_i = term1 - term2
    return np.mean(crps_i)


def main():
    start_time = time.time()
    sns.set_theme(context="paper", style="darkgrid")
    
    np.random.seed(42)
    n_samples = 20

    x0 = np.random.randn(n_samples)
    x1 = np.random.randn(n_samples)
    x2 = x0 + 0.5 * x1
    x3 = np.random.randn(n_samples)
    x4 = x3 + 2 * x0 - x2

    X_all = np.column_stack((x0, x1, x2, x3, x4))
    true_beta = np.array([0.333, 4.2, 2.4, 3, 0.5])
    y_true = X_all.dot(true_beta) + np.random.randn(n_samples) * 0.5

    percentages = np.arange(10, 110, 10)
    n_bootstrap = 50
    target_CLs = np.linspace(0.05, 0.95, 19)
    
    calibration_results = []
    summary_metrics = []
    performance_report = []
    
    l1_convergence = np.zeros((len(percentages), n_bootstrap))
    lm_convergence = np.zeros((len(percentages), n_bootstrap))

    fig1, axes1 = plt.subplots(5, 2, figsize=(12, 20))
    fig2, axes2 = plt.subplots(5, 2, figsize=(12, 20))
    LM_color = "#3C5488"
    L1_color = "#E64B35"

    for idx, p in enumerate(percentages):
        n_censor = int(n_samples * p / 100)
        np.random.seed(42 + p) 
        censor_indices = np.random.choice(n_samples, n_censor, replace=False)
        
        y_meas = y_true.copy()
        bound_type = np.array(['both'] * n_samples, dtype=object)
        
        half_censor = n_censor // 2
        for i, c_idx in enumerate(censor_indices):
            shift = np.random.uniform(0, 4)
            if i < half_censor:
                bound_type[c_idx] = 'upper_only' 
                y_meas[c_idx] = y_true[c_idx] + shift
            else:
                bound_type[c_idx] = 'lower_only' 
                y_meas[c_idx] = y_true[c_idx] - shift
                
        X_train = X_all[:16, :]
        y_true_train = y_true[:16]
        y_meas_train = y_meas[:16]
        bound_type_train = bound_type[:16]
        
        X_test = X_all[16:, :]
        Y_test = y_true[16:]
        y_meas_test = y_meas[16:]
        bound_type_test = bound_type[16:]
        
        rng = np.random.default_rng(123 + p)
        
        # -----------------------------
        # Model 1: Linear Regression
        # -----------------------------
        deterministic_mask = (bound_type_train == 'both')
        X_train_det = X_train[deterministic_mask]
        y_train_det = y_true_train[deterministic_mask]

        test_preds_lm = []
        train_preds_lm = []

        if len(X_train_det) > 0 and p < 100:
            for b in range(n_bootstrap):
                indices = rng.choice(len(X_train_det), size=len(X_train_det), replace=True)
                Xb = X_train_det[indices]
                yb = y_train_det[indices]
                lr_model = LinearRegression().fit(Xb, yb)
                test_preds_lm.append(lr_model.predict(X_test).ravel())
                train_preds_lm.append(lr_model.predict(X_train).ravel())
        else:
            for b in range(n_bootstrap):
                test_preds_lm.append(np.zeros(len(Y_test)))
                train_preds_lm.append(np.zeros(len(X_train)))
                
        test_preds_lm_arr = np.stack(test_preds_lm, axis=0)
        train_preds_lm_arr = np.stack(train_preds_lm, axis=0)

        # -----------------------------
        # Model 2: L1-Norm
        # -----------------------------
        test_preds_l1 = []
        train_preds_l1 = []

        for b in range(n_bootstrap):
            n_train = len(X_train)
            indices = rng.choice(n_train, size=n_train, replace=True)
            X_boot = X_train[indices]
            y_meas_boot = y_meas_train[indices]
            bound_boot = bound_type_train[indices]
            
            beta_vals = train_gekko_linear(X_boot, y_meas_boot, bound_boot, X_all.shape[1])
            test_preds_l1.append(X_test.dot(beta_vals).ravel())
            train_preds_l1.append(X_train.dot(beta_vals).ravel())

        test_preds_l1_arr = np.stack(test_preds_l1, axis=0)
        train_preds_l1_arr = np.stack(train_preds_l1, axis=0)

        # -----------------------------
        # Calibration Curve Calculation
        # -----------------------------
        l1_coverages = []
        lm_coverages = []

        for cl in target_CLs:
            alpha = 1.0 - cl
            
            # L1-Norm
            l1_lower = np.percentile(test_preds_l1_arr, (alpha/2)*100, axis=0)
            l1_upper = np.percentile(test_preds_l1_arr, (1 - alpha/2)*100, axis=0)
            l1_cov = np.mean((Y_test >= l1_lower) & (Y_test <= l1_upper))
            l1_coverages.append(l1_cov)
            
            # LM
            lm_lower = np.percentile(test_preds_lm_arr, (alpha/2)*100, axis=0)
            lm_upper = np.percentile(test_preds_lm_arr, (1 - alpha/2)*100, axis=0)
            lm_cov = np.mean((Y_test >= lm_lower) & (Y_test <= lm_upper))
            lm_coverages.append(lm_cov)
            
            calibration_results.append({
                'Censorship %': p,
                'Target Confidence Level': round(cl, 3),
                'L1-Norm Empirical Coverage': round(l1_cov, 3),
                'LM Empirical Coverage': round(lm_cov, 3)
            })

        # -----------------------------
        # UQ Metrics Calculation
        # -----------------------------
        l1_w_lower = np.percentile(test_preds_l1_arr, 2.5, axis=0)
        l1_w_upper = np.percentile(test_preds_l1_arr, 97.5, axis=0)
        l1_mean_width = np.mean(l1_w_upper - l1_w_lower)
        l1_cov_95 = np.mean((Y_test >= l1_w_lower) & (Y_test <= l1_w_upper))
        
        lm_w_lower = np.percentile(test_preds_lm_arr, 2.5, axis=0)
        lm_w_upper = np.percentile(test_preds_lm_arr, 97.5, axis=0)
        lm_mean_width = np.mean(lm_w_upper - lm_w_lower)
        lm_cov_95 = np.mean((Y_test >= lm_w_lower) & (Y_test <= lm_w_upper))
        
        l1_mace = np.mean([abs(cov - cl) for cov, cl in zip(l1_coverages, target_CLs)])
        lm_mace = np.mean([abs(cov - cl) for cov, cl in zip(lm_coverages, target_CLs)])
        
        l1_rmsce = np.sqrt(np.mean([(cov - cl)**2 for cov, cl in zip(l1_coverages, target_CLs)]))
        lm_rmsce = np.sqrt(np.mean([(cov - cl)**2 for cov, cl in zip(lm_coverages, target_CLs)]))
        
        l1_crps = compute_crps(test_preds_l1_arr, Y_test)
        lm_crps = compute_crps(test_preds_lm_arr, Y_test)
        
        l1_mean = np.mean(test_preds_l1_arr, axis=0)
        lm_mean = np.mean(test_preds_lm_arr, axis=0)
        
        l1_mae_val = mean_absolute_error(Y_test, l1_mean)
        lm_mae_val = mean_absolute_error(Y_test, lm_mean)
        
        for k in range(1, n_bootstrap + 1):
            l1_mean_k = np.mean(test_preds_l1_arr[:k], axis=0)
            lm_mean_k = np.mean(test_preds_lm_arr[:k], axis=0)
            l1_convergence[idx, k-1] = mean_absolute_error(Y_test, l1_mean_k)
            lm_convergence[idx, k-1] = mean_absolute_error(Y_test, lm_mean_k)
            
        summary_metrics.append({
            'Censorship %': p,
            'L1-Norm 95% Coverage': l1_cov_95,
            'LM 95% Coverage': lm_cov_95,
            'L1-Norm 95% CI Width': l1_mean_width,
            'LM 95% CI Width': lm_mean_width,
            'L1-Norm MACE': l1_mace,
            'LM MACE': lm_mace,
            'L1-Norm RMSCE': l1_rmsce,
            'LM RMSCE': lm_rmsce,
            'L1-Norm CRPS': l1_crps,
            'LM CRPS': lm_crps,
            'L1-Norm MAE': l1_mae_val,
            'LM MAE': lm_mae_val
        })
        
        # -----------------------------
        # Performance Report (A, MACV, Train MAE)
        # -----------------------------
        l1_mean_train = np.mean(train_preds_l1_arr, axis=0)
        lm_mean_train = np.mean(train_preds_lm_arr, axis=0)
        
        l1_mae_train = mean_absolute_error(y_true_train, l1_mean_train)
        lm_mae_train = mean_absolute_error(y_true_train, lm_mean_train)

        A_l1_train, MACV_l1_train = compute_MACV_A(l1_mean_train, y_meas_train, bound_type_train, y_true_train)
        A_lm_train, MACV_lm_train = compute_MACV_A(lm_mean_train, y_meas_train, bound_type_train, y_true_train)

        A_l1_test, MACV_l1_test = compute_MACV_A(l1_mean, y_meas_test, bound_type_test, Y_test)
        A_lm_test, MACV_lm_test = compute_MACV_A(lm_mean, y_meas_test, bound_type_test, Y_test)
        
        train_censor_count = np.sum(bound_type_train != 'both')
        test_censor_count = np.sum(bound_type_test != 'both')
        
        performance_report.append({
            'Censorship %': p,
            'Train Points': len(X_train),
            'Test Points': len(X_test),
            'Train Censored Points': train_censor_count,
            'Test Censored Points': test_censor_count,
            'L1 Train MAE': l1_mae_train,
            'LM Train MAE': lm_mae_train,
            'L1 Test MAE': l1_mae_val,
            'LM Test MAE': lm_mae_val,
            'L1 Train Accuracy': A_l1_train,
            'LM Train Accuracy': A_lm_train,
            'L1 Test Accuracy': A_l1_test,
            'LM Test Accuracy': A_lm_test,
            'L1 Train MACV': MACV_l1_train,
            'LM Train MACV': MACV_lm_train,
            'L1 Test MACV': MACV_l1_test,
            'LM Test MACV': MACV_lm_test
        })

        # -----------------------------
        # Pairplot (True vs Pred + Error Bars)
        # -----------------------------
        if idx < 5:
            ax_pair = axes1[idx, 0]
            ax_cal = axes1[idx, 1]
        else:
            ax_pair = axes2[idx - 5, 0]
            ax_cal = axes2[idx - 5, 1]
        
        l1_err_low = np.maximum(0, l1_mean - l1_w_lower)
        l1_err_high = np.maximum(0, l1_w_upper - l1_mean)
        
        lm_err_low = np.maximum(0, lm_mean - lm_w_lower)
        lm_err_high = np.maximum(0, lm_w_upper - lm_mean)
        
        mn, mx = min(Y_test) - 1, max(Y_test) + 1
        ax_pair.plot([mn, mx], [mn, mx], 'k--', alpha=0.7)
        
        # Raw Bootstrap Scatters
        for b in range(n_bootstrap):
            ax_pair.scatter(Y_test, test_preds_l1_arr[b], marker='x', color=L1_color, alpha=0.3, 
                            label=r'$\ell_1$-norm Bootstraps' if ((idx==0 or idx==5) and b==0) else "")
            if p < 100:
                ax_pair.scatter(Y_test, test_preds_lm_arr[b], marker='+', color=LM_color, alpha=0.3, 
                                label='Linear Model Bootstraps' if ((idx==0 or idx==5) and b==0) else "")
        
        # Mean & Error Bars overlaid
        ax_pair.errorbar(Y_test, l1_mean, yerr=[l1_err_low, l1_err_high], fmt='o', color=L1_color, alpha=0.9, 
                         label=r'$\ell_1$-norm - 95% CI' if (idx==0 or idx==5) else "")
        if p < 100:
            ax_pair.errorbar(Y_test, lm_mean, yerr=[lm_err_low, lm_err_high], fmt='s', color=LM_color, alpha=0.9, 
                             label='Linear Model - 95% CI' if (idx==0 or idx==5) else "")
            
        ax_pair.set_title(f'{p}% Censored: True vs Predicted (95% CI)')
        ax_pair.set_xlabel('True Value')
        ax_pair.set_ylabel('Predicted Value')
        ax_pair.grid(True, linestyle='--', alpha=0.6)
        
        mae_text = f"$\ell_1$-norm MAE: {l1_mae_val:.3f}\nLinear Model MAE: {lm_mae_val:.3f}"
        ax_pair.text(0.05, 0.95, mae_text, transform=ax_pair.transAxes, 
                     fontsize=10, verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

        # -----------------------------
        # Calibration Plot
        # -----------------------------
        ax_cal.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration' if (idx==0 or idx==5) else "")
        ax_cal.plot(target_CLs, l1_coverages, marker='o', color=L1_color, label=r'$\ell_1$-norm' if (idx==0 or idx==5) else "", lw=2)
        ax_cal.plot(target_CLs, lm_coverages, marker='s', color=LM_color, label='Linear Model' if (idx==0 or idx==5) else "", lw=2)
        
        ax_cal.set_title(f'{p}% Censored: Average Calibration')
        ax_cal.set_xlabel('Target Confidence Level')
        ax_cal.set_ylabel('Empirical Coverage')
        ax_cal.set_xlim([0, 1])
        ax_cal.set_ylim([0, 1.05])
        ax_cal.grid(True, linestyle='--', alpha=0.6)

    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.legend(loc='lower center', bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=True, fontsize=12)
    plot_path_png1 = r'~/UQ_Calibration_Curve_Subplots_1.png'
    plot_path_svg1 = r'~/UQ_Calibration_Curve_Subplots_1.svg'
    fig1.savefig(plot_path_png1, dpi=300, bbox_inches='tight')
    fig1.savefig(plot_path_svg1, dpi=300, bbox_inches='tight')
    
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.legend(loc='lower center', bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=True, fontsize=12)
    plot_path_png2 = r'~/UQ_Calibration_Curve_Subplots_2.png'
    plot_path_svg2 = r'~/UQ_Calibration_Curve_Subplots_2.svg'
    fig2.savefig(plot_path_png2, dpi=300, bbox_inches='tight')
    fig2.savefig(plot_path_svg2, dpi=300, bbox_inches='tight')
    print("Saved Calibration subplots as PNG and SVG in two parts")
    
    # Save tables
    df_cal = pd.DataFrame(calibration_results)
    df_cal.to_csv(r'~/UQ_Calibration_Results.csv', index=False)
    
    df_perf = pd.DataFrame(performance_report)
    df_perf.to_csv(r'~/UQ_Performance_Report.csv', index=False)

    # -------- UQ Metrics Summary Plots --------
    df_sum = pd.DataFrame(summary_metrics)
    df_sum_lm = df_sum[df_sum['Censorship %'] < 100]
    
    # Plot 1: MPIW and CRPS
    fig_sum1, (ax_mpiw, ax_crps) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel: CI Width (Sharpness / MPIW)
    ax_mpiw.plot(df_sum['Censorship %'], df_sum['L1-Norm 95% CI Width'], marker='o', color=L1_color, label=r'$\ell_1$-norm')
    ax_mpiw.plot(df_sum_lm['Censorship %'], df_sum_lm['LM 95% CI Width'], marker='s', color=LM_color, label='Linear Model')
    ax_mpiw.set_xlabel('Percent Censored Data (%)')
    ax_mpiw.set_ylabel('95% CI Width')
    ax_mpiw.set_title('Mean Prediction Interval Width')
    ax_mpiw.legend(frameon=True)
    ax_mpiw.grid(True, linestyle='--', alpha=0.6)

    # Panel: CRPS
    ax_crps.plot(df_sum['Censorship %'], df_sum['L1-Norm CRPS'], marker='o', color=L1_color, label=r'$\ell_1$-norm')
    ax_crps.plot(df_sum_lm['Censorship %'], df_sum_lm['LM CRPS'], marker='s', color=LM_color, label='Linear Model')
    ax_crps.set_xlabel('Percent Censored Data (%)')
    ax_crps.set_ylabel('CRPS')
    ax_crps.set_title('Continuous Ranked Probability Score')
    ax_crps.legend(frameon=True)
    ax_crps.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    sum_path1_png = r'~/UQ_Metrics_MPIW_CRPS.png'
    sum_path1_svg = r'~/UQ_Metrics_MPIW_CRPS.svg'
    fig_sum1.savefig(sum_path1_png, dpi=300, bbox_inches='tight')
    fig_sum1.savefig(sum_path1_svg, dpi=300, bbox_inches='tight')
    print("Saved MPIW & CRPS plots as PNG and SVG")

    # Plot 2: MACE, RMSCE, PICP
    fig_sum2, (ax_mace, ax_rmsce, ax_picp) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel: MACE
    ax_mace.plot(df_sum['Censorship %'], df_sum['L1-Norm MACE'], marker='o', color=L1_color, label=r'$\ell_1$-norm')
    ax_mace.plot(df_sum_lm['Censorship %'], df_sum_lm['LM MACE'], marker='s', color=LM_color, label='Linear Model')
    ax_mace.set_xlabel('Percent Censored Data (%)')
    ax_mace.set_ylabel('MACE')
    ax_mace.set_title('Mean Absolute Calibration Error')
    ax_mace.legend(frameon=True)
    ax_mace.grid(True, linestyle='--', alpha=0.6)
    
    # Panel: RMSCE
    ax_rmsce.plot(df_sum['Censorship %'], df_sum['L1-Norm RMSCE'], marker='o', color=L1_color, label=r'$\ell_1$-norm')
    ax_rmsce.plot(df_sum_lm['Censorship %'], df_sum_lm['LM RMSCE'], marker='s', color=LM_color, label='Linear Model')
    ax_rmsce.set_xlabel('Percent Censored Data (%)')
    ax_rmsce.set_ylabel('RMSCE')
    ax_rmsce.set_title('Root Mean Square Calibration Error')
    ax_rmsce.legend(frameon=True)
    ax_rmsce.grid(True, linestyle='--', alpha=0.6)
    
    # Panel: Coverage (PICP)
    ax_picp.plot(df_sum['Censorship %'], df_sum['L1-Norm 95% Coverage'], marker='o', color=L1_color, label=r'$\ell_1$-norm')
    ax_picp.plot(df_sum_lm['Censorship %'], df_sum_lm['LM 95% Coverage'], marker='s', color=LM_color, label='Linear Model')
    ax_picp.axhline(0.95, color='k', linestyle='--', alpha=0.5, label='Target (0.95)')
    ax_picp.set_xlabel('Percent Censored Data (%)')
    ax_picp.set_ylabel('95% Prediction Interval Coverage')
    ax_picp.set_title('Prediction Interval Coverage Probability')
    ax_picp.legend(frameon=True)
    ax_picp.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    sum_path2_png = r'~/UQ_Metrics_MACE_RMSCE_PICP.png'
    sum_path2_svg = r'~/UQ_Metrics_MACE_RMSCE_PICP.svg'
    fig_sum2.savefig(sum_path2_png, dpi=300, bbox_inches='tight')
    fig_sum2.savefig(sum_path2_svg, dpi=300, bbox_inches='tight')
    print("Saved MACE, RMSCE & PICP plots as PNG and SVG")

    # -------- 1x3 Performance Plot --------
    df_perf_lm = df_perf[df_perf['Censorship %'] < 100]
    fig_perf, (ax_p_mae, ax_p_acc, ax_p_macv) = plt.subplots(1, 3, figsize=(18, 5))
    
    # MAE
    ax_p_mae.plot(df_perf['Censorship %'], df_perf['L1 Train MAE'], marker='o', linestyle='--', color=L1_color, label=r'$\ell_1$-norm Train')
    ax_p_mae.plot(df_perf['Censorship %'], df_perf['L1 Test MAE'], marker='o', linestyle='-', color=L1_color, label=r'$\ell_1$-norm Test')
    ax_p_mae.plot(df_perf_lm['Censorship %'], df_perf_lm['LM Train MAE'], marker='s', linestyle='--', color=LM_color, label='Linear Model Train')
    ax_p_mae.plot(df_perf_lm['Censorship %'], df_perf_lm['LM Test MAE'], marker='s', linestyle='-', color=LM_color, label='Linear Model Test')
    ax_p_mae.set_title('Mean Absolute Error')
    ax_p_mae.set_xlabel('Percent Censored Data (%)')
    ax_p_mae.set_ylabel('MAE')
    ax_p_mae.grid(True, linestyle='--', alpha=0.6)
    ax_p_mae.legend(frameon=True)
    
    # Accuracy
    ax_p_acc.plot(df_perf['Censorship %'], df_perf['L1 Train Accuracy'], marker='o', linestyle='--', color=L1_color)
    ax_p_acc.plot(df_perf['Censorship %'], df_perf['L1 Test Accuracy'], marker='o', linestyle='-', color=L1_color)
    ax_p_acc.plot(df_perf_lm['Censorship %'], df_perf_lm['LM Train Accuracy'], marker='s', linestyle='--', color=LM_color)
    ax_p_acc.plot(df_perf_lm['Censorship %'], df_perf_lm['LM Test Accuracy'], marker='s', linestyle='-', color=LM_color)
    ax_p_acc.set_title('Constraint Accuracy')
    ax_p_acc.set_xlabel('Percent Censored Data (%)')
    ax_p_acc.set_ylabel('Accuracy Proportion')
    ax_p_acc.set_ylim([-0.05, 1.05])
    ax_p_acc.grid(True, linestyle='--', alpha=0.6)
    
    # MACV
    ax_p_macv.plot(df_perf['Censorship %'], df_perf['L1 Train MACV'], marker='o', linestyle='--', color=L1_color)
    ax_p_macv.plot(df_perf['Censorship %'], df_perf['L1 Test MACV'], marker='o', linestyle='-', color=L1_color)
    ax_p_macv.plot(df_perf_lm['Censorship %'], df_perf_lm['LM Train MACV'], marker='s', linestyle='--', color=LM_color)
    ax_p_macv.plot(df_perf_lm['Censorship %'], df_perf_lm['LM Test MACV'], marker='s', linestyle='-', color=LM_color)
    ax_p_macv.set_title('Mean Absolute Constraint Violation')
    ax_p_macv.set_xlabel('Percent Censored Data (%)')
    ax_p_macv.set_ylabel('MACV')
    ax_p_macv.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    perf_path_png = r'~/UQ_Performance_Metrics.png'
    perf_path_svg = r'~/UQ_Performance_Metrics.svg'
    fig_perf.savefig(perf_path_png, dpi=300, bbox_inches='tight')
    fig_perf.savefig(perf_path_svg, dpi=300, bbox_inches='tight')
    print("Saved Performance Metrics plot as PNG and SVG")

    # -------- Ensemble Convergence Plot --------
    l1_convergence_mean = np.mean(l1_convergence, axis=0)
    lm_convergence_mean = np.mean(lm_convergence, axis=0)
    
    fig_conv, ax_conv = plt.subplots(figsize=(8, 5))
    k_vals = np.arange(1, n_bootstrap + 1)
    ax_conv.plot(k_vals, l1_convergence_mean, marker='o', label=r'$\ell_1$-norm', color=L1_color, lw=2, markersize=4)
    ax_conv.plot(k_vals, lm_convergence_mean, marker='s', label='Linear Model', color=LM_color, lw=2, markersize=4)
    ax_conv.set_xlabel('Number of Bootstrap Samples ($n$)')
    ax_conv.set_ylabel('MAE')
    ax_conv.set_title('Ensemble Convergence')
    ax_conv.legend(frameon=True)
    ax_conv.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    conv_path_png = r'~/UQ_Ensemble_Convergence.png'
    conv_path_svg = r'~/UQ_Ensemble_Convergence.svg'
    fig_conv.savefig(conv_path_png, dpi=300, bbox_inches='tight')
    fig_conv.savefig(conv_path_svg, dpi=300, bbox_inches='tight')
    print("Saved Ensemble Convergence plot as PNG and SVG")

    # -------- Final Report --------
    print("\n### Performance Metrics Final Report ###\n")
    print(df_perf[['Censorship %', 'Train Censored Points', 'Test Censored Points', 'L1 Train MAE', 'LM Train MAE', 'L1 Test MAE', 'LM Test MAE', 'L1 Train Accuracy', 'LM Train Accuracy', 'L1 Test Accuracy', 'LM Test Accuracy']].to_string(index=False))

    end_time = time.time()
    execution_time = end_time - start_time
    
    log_path = r'~/Reproducibility_Log.txt'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=== Reproducibility Log ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Execution Time (n_bootstrap={n_bootstrap}): {execution_time:.2f} seconds\n")
        f.write("\n=== System Information ===\n")
        f.write(f"Python Version: {sys.version}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write("\n=== Python Packages ===\n")
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
            f.write(result.stdout)
        except Exception as e:
            f.write(f"Could not retrieve pip freeze: {e}\n")
            
    print(f"\nReproducibility log saved to {log_path}")

if __name__ == '__main__':
    main()
