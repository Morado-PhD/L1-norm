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

def train_gekko_linear(X_boot, y_meas_boot, bound_boot, n_features):
    m = GEKKO(remote=False)
    n_samples = X_boot.shape[0]

    # Coefficients
    beta = [m.FV(value=0.0, name=f'b{j}') for j in range(n_features)]
    for b in beta:
        b.STATUS = 1  # optimize

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

    # 1. Base Noise for CRN
    base_noise = np.random.randn(n_samples)
    noise_levels = [0.0, 0.25, 0.5, 1.0, 2.0]

    percentages = np.arange(10, 110, 10)
    n_bootstrap = 50
    alpha = 0.05
    
    csv_results = []
    
    for nl in noise_levels:
        y_true = X_all.dot(true_beta) + base_noise * nl

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
            all_preds_lm = []
            metrics_lm_test = {'r2': [], 'mse': [], 'mae': []}
            metrics_lm_train = {'r2': [], 'mse': [], 'mae': []}

            if len(X_train_det) > 0 and p < 100:
                for b in range(n_bootstrap):
                    indices = rng.choice(len(X_train_det), size=len(X_train_det), replace=True)
                    Xb = X_train_det[indices]
                    yb = y_train_det[indices]
                    
                    lr_model = LinearRegression().fit(Xb, yb)
                    test_pred_lm = lr_model.predict(X_test).ravel()
                    test_preds_lm.append(test_pred_lm)
                    
                    all_pred_lm = lr_model.predict(X_all).ravel()
                    all_preds_lm.append(all_pred_lm)
                    
                    train_pred_lm = lr_model.predict(X_train).ravel()
                    
                    r2 = r2_score(Y_test, test_pred_lm)
                    mse = mean_squared_error(Y_test, test_pred_lm)
                    mae = mean_absolute_error(Y_test, test_pred_lm)
                    metrics_lm_test['r2'].append(r2)
                    metrics_lm_test['mse'].append(mse)
                    metrics_lm_test['mae'].append(mae)

                    r2_t = r2_score(y_true_train, train_pred_lm)
                    mse_t = mean_squared_error(y_true_train, train_pred_lm)
                    mae_t = mean_absolute_error(y_true_train, train_pred_lm)
                    metrics_lm_train['r2'].append(r2_t)
                    metrics_lm_train['mse'].append(mse_t)
                    metrics_lm_train['mae'].append(mae_t)
                    
                    A, MACV = compute_MACV_A(all_pred_lm, y_meas, bound_type, y_true)
                    
                    csv_results.append({
                        'Noise Level': nl, 'Subset %': f"{p}%", 'Model': 'LM', 'Type': f'Bootstrap_{b+1}',
                        'Train R2': r2_t, 'Train MSE': mse_t, 'Train MAE': mae_t,
                        'Test R2': r2, 'Test MSE': mse, 'Test MAE': mae, 'Accuracy': A, 'MACV': MACV
                    })
                    
                test_preds_lm = np.stack(test_preds_lm, axis=0)
                boot_mean_lm = test_preds_lm.mean(axis=0)
                
                all_preds_lm = np.stack(all_preds_lm, axis=0)
                boot_mean_all_lm = all_preds_lm.mean(axis=0)
                
                r2_mean = np.mean(metrics_lm_test['r2'])
                mse_mean = np.mean(metrics_lm_test['mse'])
                mae_mean = np.mean(metrics_lm_test['mae'])

                r2_mean_t = np.mean(metrics_lm_train['r2'])
                mse_mean_t = np.mean(metrics_lm_train['mse'])
                mae_mean_t = np.mean(metrics_lm_train['mae'])

                A_mean, MACV_mean = compute_MACV_A(boot_mean_all_lm, y_meas, bound_type, y_true)
                
                n_censor_train = int(16 * p / 100)
                n_lm_train = 16 - n_censor_train
                csv_results.append({
                    'Noise Level': nl, 'Subset %': f"{p}%", 'Model': 'LM', 'Type': 'Mean', 
                    'Censored Points': n_censor_train, 'Train Points': n_lm_train, 'Test Points': 4,
                    'Train R2': r2_mean_t, 'Train MSE': mse_mean_t, 'Train MAE': mae_mean_t,
                    'Test R2': r2_mean, 'Test MSE': mse_mean, 'Test MAE': mae_mean, 'Accuracy': A_mean, 'MACV': MACV_mean
                })
            else:
                for b in range(n_bootstrap):
                    csv_results.append({
                        'Noise Level': nl, 'Subset %': f"{p}%", 'Model': 'LM', 'Type': f'Bootstrap_{b+1}',
                        'Train R2': 0.0, 'Train MSE': 0.0, 'Train MAE': 0.0,
                        'Test R2': 0.0, 'Test MSE': 0.0, 'Test MAE': 0.0, 'Accuracy': np.nan, 'MACV': np.nan
                    })
                csv_results.append({
                        'Noise Level': nl, 'Subset %': f"{p}%", 'Model': 'LM', 'Type': 'Mean', 
                        'Censored Points': 16, 'Train Points': 0, 'Test Points': 4,
                        'Train R2': 0.0, 'Train MSE': 0.0, 'Train MAE': 0.0,
                        'Test R2': 0.0, 'Test MSE': 0.0, 'Test MAE': 0.0, 'Accuracy': np.nan, 'MACV': np.nan
                })

            # -----------------------------
            # Model 2: L1-Norm
            # -----------------------------
            test_preds_l1 = []
            all_preds_l1 = []
            metrics_l1_test = {'r2': [], 'mse': [], 'mae': []}
            metrics_l1_train = {'r2': [], 'mse': [], 'mae': []}

            for b in range(n_bootstrap):
                n_train = len(X_train)
                indices = rng.choice(n_train, size=n_train, replace=True)
                X_boot = X_train[indices]
                y_meas_boot = y_meas_train[indices]
                bound_boot = bound_type_train[indices]
                
                beta_vals = train_gekko_linear(X_boot, y_meas_boot, bound_boot, X_all.shape[1])
                test_pred_l1 = X_test.dot(beta_vals).ravel()
                test_preds_l1.append(test_pred_l1)
                
                all_pred_l1 = X_all.dot(beta_vals).ravel()
                all_preds_l1.append(all_pred_l1)
                
                train_pred_l1 = X_train.dot(beta_vals).ravel()
                
                r2 = r2_score(Y_test, test_pred_l1)
                mse = mean_squared_error(Y_test, test_pred_l1)
                mae = mean_absolute_error(Y_test, test_pred_l1)
                metrics_l1_test['r2'].append(r2)
                metrics_l1_test['mse'].append(mse)
                metrics_l1_test['mae'].append(mae)

                r2_t = r2_score(y_true_train, train_pred_l1)
                mse_t = mean_squared_error(y_true_train, train_pred_l1)
                mae_t = mean_absolute_error(y_true_train, train_pred_l1)
                metrics_l1_train['r2'].append(r2_t)
                metrics_l1_train['mse'].append(mse_t)
                metrics_l1_train['mae'].append(mae_t)
                
                A, MACV = compute_MACV_A(all_pred_l1, y_meas, bound_type, y_true)
                
                csv_results.append({
                    'Noise Level': nl, 'Subset %': f"{p}%", 'Model': 'L1-Norm', 'Type': f'Bootstrap_{b+1}',
                    'Train R2': r2_t, 'Train MSE': mse_t, 'Train MAE': mae_t,
                    'Test R2': r2, 'Test MSE': mse, 'Test MAE': mae, 'Accuracy': A, 'MACV': MACV
                })

            test_preds_l1 = np.stack(test_preds_l1, axis=0)
            boot_mean_l1 = test_preds_l1.mean(axis=0)
            
            all_preds_l1 = np.stack(all_preds_l1, axis=0) 
            boot_mean_all_l1 = all_preds_l1.mean(axis=0) 
            
            r2_mean = np.mean(metrics_l1_test['r2'])
            mse_mean = np.mean(metrics_l1_test['mse'])
            mae_mean = np.mean(metrics_l1_test['mae'])

            r2_mean_t = np.mean(metrics_l1_train['r2'])
            mse_mean_t = np.mean(metrics_l1_train['mse'])
            mae_mean_t = np.mean(metrics_l1_train['mae'])

            A_mean, MACV_mean = compute_MACV_A(boot_mean_all_l1, y_meas, bound_type, y_true)
            
            n_censor_train = int(16 * p / 100)
            csv_results.append({
                'Noise Level': nl, 'Subset %': f"{p}%", 'Model': 'L1-Norm', 'Type': 'Mean', 
                'Censored Points': n_censor_train, 'Train Points': 16, 'Test Points': 4,
                'Train R2': r2_mean_t, 'Train MSE': mse_mean_t, 'Train MAE': mae_mean_t,
                'Test R2': r2_mean, 'Test MSE': mse_mean, 'Test MAE': mae_mean, 'Accuracy': A_mean, 'MACV': MACV_mean
            })


    # Results to CSV
    df_res = pd.DataFrame(csv_results)
    df_res.to_csv(r'C:\Users\erick\OneDrive\PhD\Research\TL_Submission\Comments\Noise\Bootstrapped_UQ_Metrics.csv', index=False)

    df_mean = df_res[df_res['Type'] == 'Mean']
    df_mean['Subset Numeric'] = df_mean['Subset %'].str.replace('%', '').astype(int)

    LM_color = "#3C5488"
    L1_color = "#E64B35"

    # -------- Degradation Plot --------
    fig_deg, ax_deg = plt.subplots(figsize=(8, 6))
    
    cens_levels = [20, 50, 80]
    line_styles = ['-', '--', ':']
    
    for c, ls in zip(cens_levels, line_styles):
        l1_subset = df_mean[(df_mean['Model'] == 'L1-Norm') & (df_mean['Subset Numeric'] == c)]
        ax_deg.plot(l1_subset['Noise Level'], l1_subset['Test MAE'], 
                    marker='o', color=L1_color, linestyle=ls, label=f'L1-Norm ({c}% Censored)', lw=2)
        
        if c <= 80:
            lm_subset = df_mean[(df_mean['Model'] == 'LM') & (df_mean['Subset Numeric'] == c)]
            ax_deg.plot(lm_subset['Noise Level'], lm_subset['Test MAE'], 
                        marker='s', color=LM_color, linestyle=ls, alpha=0.5, label=f'Linear ({c}% Censored)', lw=1.5)

    ax_deg.set_xlabel('Noise Level (Standard Deviations)')
    ax_deg.set_ylabel('Test MAE')
    ax_deg.set_title('Test MAE Degradation vs. Noise Magnitude')
    ax_deg.legend(frameon=True, fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    ax_deg.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\erick\OneDrive\PhD\Research\TL_Submission\Comments\Noise\Noise_Degradation_Plot.png', dpi=300)
    print("Saved Degradation plot as 'C:\\Users\\erick\\OneDrive\\PhD\\Research\\TL_Submission\\Comments\\Noise\\Noise_Degradation_Plot.png'")

    # -------- Heatmap L1 --------
    fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
    
    df_l1_heat = df_mean[df_mean['Model'] == 'L1-Norm'].pivot(index='Noise Level', columns='Subset Numeric', values='Test MAE')
    df_l1_heat = df_l1_heat.sort_index(ascending=False)
    
    sns.heatmap(df_l1_heat, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax_heat, cbar_kws={'label': 'Test MAE'})
    ax_heat.set_xlabel('Percent Censored Data (%)')
    ax_heat.set_ylabel('Noise Level')
    ax_heat.set_title(r'$\ell_1$-norm')
    
    plt.tight_layout()
    fig_heat.savefig(r'C:\Users\erick\OneDrive\PhD\Research\TL_Submission\Comments\Noise\Noise_Heatmap.png', dpi=300)
    fig_heat.savefig(r'C:\Users\erick\OneDrive\PhD\Research\TL_Submission\Comments\Noise\Noise_Heatmap.svg', dpi=300)
    print("Saved L1-Norm Heatmap as PNG and SVG")

    # -------- Heatmap LM --------
    fig_heat_lm, ax_heat_lm = plt.subplots(figsize=(8, 6))
    
    df_lm_heat = df_mean[df_mean['Model'] == 'LM'].pivot(index='Noise Level', columns='Subset Numeric', values='Test MAE')
    if 100 in df_lm_heat.columns:
        df_lm_heat = df_lm_heat.drop(columns=[100])
    df_lm_heat = df_lm_heat.sort_index(ascending=False)
    
    sns.heatmap(df_lm_heat, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax_heat_lm, cbar_kws={'label': 'Test MAE'})
    ax_heat_lm.set_xlabel('Percent Censored Data (%)')
    ax_heat_lm.set_ylabel('Noise Level')
    ax_heat_lm.set_title(r'Linear Model')
    
    plt.tight_layout()
    fig_heat_lm.savefig(r'C:\Users\erick\OneDrive\PhD\Research\TL_Submission\Comments\Noise\Noise_Heatmap_LM.png', dpi=300)
    fig_heat_lm.savefig(r'C:\Users\erick\OneDrive\PhD\Research\TL_Submission\Comments\Noise\Noise_Heatmap_LM.svg', dpi=300)
    print("Saved LM Heatmap as PNG and SVG")

    # -------- Final Report --------
    report_cols = ['Noise Level', 'Subset %', 'Model', 'Censored Points', 'Train Points', 'Test Points', 'Train R2', 'Train MAE', 'Test R2', 'Test MAE']
    print("\n### Final Comparison Report (Means Only) ###\n")
    print(df_mean[report_cols].to_string(index=False))

    end_time = time.time()
    execution_time = end_time - start_time
    
    log_path = r'c:\Users\erick\OneDrive\PhD\Research\TL_Submission\Comments\Noise\Reproducibility_Log.txt'
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
