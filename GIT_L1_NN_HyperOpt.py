import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import optuna.visualization.matplotlib as ovm
import random
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unified_loss(outputs, targets, bound_type, beta=10.0):
    outputs = outputs.squeeze()
    targets = targets.squeeze()
    if outputs.dim() == 0:
        outputs = outputs.unsqueeze(0)
    if targets.dim() == 0:
        targets = targets.unsqueeze(0)
        
    bound_arr = np.asarray(bound_type)
    penalty_weight = 1.0
    
    loss_contrib = torch.zeros_like(outputs)
    
    is_upper = (bound_arr == 'upper_only')
    is_lower = (bound_arr == 'lower_only')
    is_both = (bound_arr == 'both')
    
    if np.any(is_upper):
        mask_u = torch.as_tensor(is_upper, dtype=torch.bool, device=outputs.device)
        loss_contrib[mask_u] = penalty_weight * torch.nn.functional.softplus(outputs[mask_u] - targets[mask_u], beta=beta)
        
    if np.any(is_lower):
        mask_l = torch.as_tensor(is_lower, dtype=torch.bool, device=outputs.device)
        loss_contrib[mask_l] = penalty_weight * torch.nn.functional.softplus(targets[mask_l] - outputs[mask_l], beta=beta)
        
    if np.any(is_both):
        mask_b = torch.as_tensor(is_both, dtype=torch.bool, device=outputs.device)
        loss_contrib[mask_b] = torch.abs(outputs[mask_b] - targets[mask_b])
        
    return torch.mean(loss_contrib)



class DeepRegressionNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.0, activation_name="ReLU", a=0.0, b=1.0):
        super(DeepRegressionNet, self).__init__()
        activations = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(),
            "Tanh": nn.Tanh(),
            "ELU": nn.ELU(),
            "SELU": nn.SELU(),
            "SiLU": nn.SiLU(),
        }
        activation = activations.get(activation_name, nn.ReLU())
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(activation)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self.a = a
        self.b = b

    def forward(self, x):
        z = self.net(x)
        pred = self.a + (self.b - self.a) * torch.sigmoid(z)
        return pred

if __name__ == "__main__":

    set_seed(42)
    
    name_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(name_dir)
    
    # --- Load Prepared Data to Prevent Data Leakage ---
    import pickle
    data_path = os.path.join(name_dir, 'name', 'prepared_data_splits.pkl')
    
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
        

    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']
    y_full = data_dict['y_full']
    

    xs = data_dict['xs']
    ys = data_dict['ys']
    

    inputs_test = data_dict['inputs_test']
    targets_test = data_dict['targets_test']
    bound_test = data_dict['bound_test']
    
    inputs_cv = data_dict['inputs_full']
    targets_cv = data_dict['targets_full']
    bound_cv = data_dict['bound_full']
    

    a_scaled = data_dict['a_scaled']
    b_scaled = data_dict['b_scaled']
    input_dim = data_dict['input_dim']
    
    import copy
    _current_trial_init = [None]
    _best_trial_init = [None]
    _best_trial_value = [float('inf')]
    
    def _save_best_init_callback(study, trial):
        if trial.value is not None and trial.value < _best_trial_value[0]:
            _best_trial_value[0] = trial.value
            _best_trial_init[0] = _current_trial_init[0]
    
    # --- Optuna Objective with 3-Fold CV ---
    from sklearn.model_selection import KFold
    
    def objective(trial):

        num_layers = trial.suggest_int('num_layers', 2, 4)
        

        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.6)
        

        activation_name = trial.suggest_categorical('activation_name', ['ELU', 'Tanh', 'SiLU', 'SELU', 'LeakyReLU', 'ReLU'])
        
        hidden_dims = []

        prev_dim = trial.suggest_int('layer1_dim', 16, 256, step=16)
        hidden_dims.append(prev_dim)
        
        for i in range(1, num_layers):

            dim = trial.suggest_int(f'layer{i+1}_dim', 16, prev_dim, step=16)
            hidden_dims.append(dim)
            prev_dim = dim
            

        learning_rate = trial.suggest_float('learning_rate', 10e-6, 10e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 10e-6, 10e-4, log=True)
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        fold_losses = []
        fold_best_epochs = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(inputs_cv)):
            fold_inputs_train = inputs_cv[train_idx]
            fold_targets_train = targets_cv[train_idx]
            fold_bound_train = bound_cv[train_idx]
            fold_inputs_val = inputs_cv[val_idx]
            fold_targets_val = targets_cv[val_idx]
            fold_bound_val = bound_cv[val_idx]
            
            model = DeepRegressionNet(
                input_dim=input_dim, 
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                activation_name=activation_name,
                a=a_scaled,
                b=b_scaled
            )
            
            if fold_idx == 0:
                _current_trial_init[0] = copy.deepcopy(model.state_dict())
            
            epochs = 350
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            best_val_loss_fold = float('inf')
            best_epoch_fold = 0
            patience_count = 35
            epochs_no_improve = 0
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                noise = torch.randn_like(fold_inputs_train) * 0.02
                outputs = model(fold_inputs_train + noise)
                loss = unified_loss(outputs, fold_targets_train, fold_bound_train)
                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    val_outputs = model(fold_inputs_val)
                    val_loss = unified_loss(val_outputs, fold_targets_val, fold_bound_val)
                
                scheduler.step()
                    
                if val_loss.item() < best_val_loss_fold:
                    best_val_loss_fold = val_loss.item()
                    best_epoch_fold = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= patience_count:
                    break
                    
                #
                if fold_idx == 0:
                    trial.report(val_loss.item(), epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            fold_best_epochs.append(best_epoch_fold)
            fold_losses.append(best_val_loss_fold)
        
        avg_loss = np.mean(fold_losses)
        trial.set_user_attr("best_epochs", fold_best_epochs)
        return avg_loss

    #
    from optuna.pruners import MedianPruner
    
    db_path = os.path.join(name_dir, 'optuna_study.db')
    db_conn = f"sqlite:///{db_path}"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
            
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=50)
    study = optuna.create_study(
        study_name="name_pinn_tpe",
        storage=db_conn,
        load_if_exists=True,
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )
    
    print("Phase 1: TPE Search with Pruning (500 trials)...")
    study.optimize(objective, n_trials=500, show_progress_bar=True, callbacks=[_save_best_init_callback])
    
    print(f"\nPhase 1 Best: Trial {study.best_trial.number}, Value: {study.best_trial.value:.6f}")
    
    #
    print("\nPhase 2: CMA-ES Refinement (50 trials)...")
    sampler2 = optuna.samplers.CmaEsSampler(seed=42)
    study2 = optuna.create_study(
        study_name="name_pinn_cmaes",
        storage=db_conn,
        load_if_exists=True,
        direction='minimize',
        sampler=sampler2,
        pruner=pruner
    )
    
    #
    completed_trials = [t for t in study.trials if t.value is not None]
    top_trials = sorted(completed_trials, key=lambda t: t.value)[:10]
    for t in top_trials:
        study2.enqueue_trial(t.params)
    
    study2.optimize(objective, n_trials=50, show_progress_bar=True, callbacks=[_save_best_init_callback])
    
    #
    if study2.best_trial.value < study.best_trial.value:
        best_params = study2.best_trial.params
        best_value = study2.best_trial.value
        print(f"\nPhase 2 improved! Best value: {best_value:.6f}")
    else:
        best_params = study.best_trial.params
        best_value = study.best_trial.value
        print(f"\nPhase 1 was better. Best value: {best_value:.6f}")
    #
    
    print("\n================ Best Parameters ================")
    print(json.dumps(best_params, indent=4))
    
    #
    params_path = os.path.join(name_dir, 'Name.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"\nSaved best parameters to {params_path}")

    # OPTUNA VISUALIZATIONS
    print("\nGenerating Optuna Plots...")
    try:
        fig_hist = ovm.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(name_dir, 'optuna_history.png'))
        plt.close()

        fig_imp = ovm.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(name_dir, 'optuna_importances.png'))
        plt.close()
        print("Saved optuna_history.png.")
    except Exception as e:
        print(f"Could not generate Optuna plots: {e}")
    
    #
    print(f"\n--- Final Training with Validation-based Early Stopping ---")
    
    # Split the Development Set back into Train (70%) and Validation (15%)
    inputs_train = data_dict.get('inputs_train')
    targets_train = data_dict.get('targets_train')
    bound_train = data_dict.get('bound_train')
    
    inputs_val = data_dict.get('inputs_val')
    targets_val = data_dict.get('targets_val')
    bound_val = data_dict.get('bound_val')
    
    if inputs_train is None or inputs_val is None:
        inputs_train = torch.tensor(xs.transform(X_train), dtype=torch.float32).to(inputs_cv.device)
        targets_train = torch.tensor(ys.transform(y_train[['y_meas']]), dtype=torch.float32).to(targets_cv.device)
        bound_train = y_train['bound_type'].values
        
        inputs_val = torch.tensor(xs.transform(X_val), dtype=torch.float32).to(inputs_cv.device)
        targets_val = torch.tensor(ys.transform(y_val[['y_meas']]), dtype=torch.float32).to(targets_cv.device)
        bound_val = y_val['bound_type'].values
        
    hidden_dims = [best_params[f'layer{i+1}_dim'] for i in range(best_params['num_layers'])]
    
    final_model = DeepRegressionNet(
        input_dim=input_dim, 
        hidden_dims=hidden_dims,
        dropout_rate=best_params['dropout_rate'],
        activation_name=best_params['activation_name'],
        a=a_scaled,
        b=b_scaled
    )
    
    #
    if _best_trial_init[0] is not None:
        final_model.load_state_dict(_best_trial_init[0])
        print("Loaded.")
    else:
        print("Warning.")
    
    #
    init_weights_path = os.path.join(name_dir, 'init_weights.pth')
    
    import copy
    
    #
    #
    #
    init_state = copy.deepcopy(final_model.state_dict())
    
    seed_val = 42
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    
    final_model.load_state_dict(copy.deepcopy(init_state))
    max_epochs = 500
    optimizer_f = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'], 
                             weight_decay=best_params.get('weight_decay', 1e-5))
    
    #
    scheduler_f = optim.lr_scheduler.CosineAnnealingLR(optimizer_f, T_max=max_epochs)
    train_losses_f = []
    val_losses_f = []
    test_losses_f = []
    
    best_val_loss = float('inf')
    best_weights = None
    best_epoch = 0
    patience_count = 60
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        final_model.train()
        optimizer_f.zero_grad()
        noise = torch.randn_like(inputs_train) * 0.02
        outputs = final_model(inputs_train + noise)
        loss = unified_loss(outputs, targets_train, bound_train)
        loss.backward()
        optimizer_f.step()
        
        train_losses_f.append(loss.item())
        scheduler_f.step()
 
        #
        final_model.eval()
        with torch.no_grad():
            outputs_val = final_model(inputs_val)
            val_loss = unified_loss(outputs_val, targets_val, bound_val)
            val_losses_f.append(val_loss.item())
            
            outputs_test_live = final_model(inputs_test)
            loss_test = unified_loss(outputs_test_live, targets_test, bound_test)
            test_losses_f.append(loss_test.item())
            
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_weights = copy.deepcopy(final_model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience_count:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} with validation loss: {best_val_loss:.6f}")
            break
    
    #
    if best_weights is not None:
        final_model.load_state_dict(best_weights)
        print(f"Restored model weights from best epoch {best_epoch} (val loss: {best_val_loss:.6f})")
    
    print(f"\nFinal training completed. Best Epoch: {best_epoch}, Val Loss: {best_val_loss:.6f}, Test Loss at best epoch: {test_losses_f[best_epoch]:.6f}")
    
    #
    torch.save({
        'model_state_dict': init_state,
        'best_params': best_params,
        'input_dim': input_dim,
        'a_scaled': a_scaled,
        'b_scaled': b_scaled,
        'best_seed': seed_val,
    }, init_weights_path)
    print(f"Saved initialization weights + seed to {init_weights_path}")
    
    # 
    trained_weights_path = os.path.join(name_dir, 'trained_weights.pth')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'best_params': best_params,
        'input_dim': input_dim,
        'a_scaled': a_scaled,
        'b_scaled': b_scaled,
        'epochs_trained': best_epoch,
    }, trained_weights_path)
    print(f"Saved trained weights to {trained_weights_path}")
 
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses_f, label='Train Loss', linewidth=2)
    plt.plot(val_losses_f, label='Val Loss', linewidth=2, alpha=0.8)
    plt.plot(test_losses_f, label='Test Loss', linewidth=2, alpha=0.8)
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves (Method B)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(name_dir, 'final_loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved final loss plot to {loss_plot_path}")
    
    def compute_metrics(model, inputs_tensor, y_df, scaler):
        model.eval()
        with torch.no_grad():
            outputs = model(inputs_tensor)
            preds_scaled = outputs.squeeze().cpu().numpy()
            
        preds_original = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        
        P = y_df.copy()
        if not isinstance(P, pd.DataFrame):
            P = pd.DataFrame(P)
            
        P['TL_Pred'] = preds_original
        
        #
        M = P.copy()
        M['upper_only_flag'] = ((M['bound_type'] == 'upper_only') & (M['TL_Pred'] < M['y_meas'])).astype(int)
        M['lower_only_flag'] = ((M['bound_type'] == 'lower_only') & (M['TL_Pred'] > M['y_meas'])).astype(int)
        M['both_flag']       = ((M['bound_type'] == 'both')).astype(int)
    
        M['Delta'] = np.where(
            (M['bound_type'] == 'upper_only') & (M['TL_Pred'] > M['y_meas']),
            np.abs(M['y_meas'] - M['TL_Pred']),
            0
        )
    
        m = M.copy()
        m = m[m['bound_type']=='upper_only']
    
        mc = m['upper_only_flag'].sum()
        tc = m['upper_only_flag'].count()
        A = mc / tc if tc > 0 else np.nan
        v = tc - mc
        MACV = m['Delta'].sum() / v if v > 0 else np.nan
        
        #
        P_both = P[P['bound_type'] == 'both']
        if len(P_both) > 0:
            r2 = r2_score(P_both['y_meas'], P_both['TL_Pred'])
            MSE = mean_squared_error(P_both['y_meas'], P_both['TL_Pred'])
            MAE = mean_absolute_error(P_both['y_meas'], P_both['TL_Pred'])
        else:
            r2, MSE, MAE = np.nan, np.nan, np.nan
            
        return {'R2': float(r2), 'MSE': float(MSE), 'MAE': float(MAE), 
                'Accuracy': float(A) if not np.isnan(A) else None, 
                'MACV': float(MACV) if not np.isnan(MACV) else None}, P

    train_metrics, P_train = compute_metrics(final_model, inputs_train, y_train, ys)
    val_metrics, P_val = compute_metrics(final_model, inputs_val, y_val, ys)
    test_metrics, P_test = compute_metrics(final_model, inputs_test, y_test, ys)
    
    def fmt_val(val):
        if val is None or np.isnan(val):
            return "N/A"
        return f"{val:.4f}"

    print("\n--- Final Metrics (Train vs Val vs Test) ---")
    print(f"R^2:      Train = {fmt_val(train_metrics['R2'])}  |  Val = {fmt_val(val_metrics['R2'])}  |  Test = {fmt_val(test_metrics['R2'])}")
    print(f"MSE:      Train = {fmt_val(train_metrics['MSE'])}  |  Val = {fmt_val(val_metrics['MSE'])}  |  Test = {fmt_val(test_metrics['MSE'])}")
    print(f"MAE:      Train = {fmt_val(train_metrics['MAE'])}  |  Val = {fmt_val(val_metrics['MAE'])}  |  Test = {fmt_val(test_metrics['MAE'])}")
    print(f"Accuracy: Train = {fmt_val(train_metrics['Accuracy'])}  |  Val = {fmt_val(val_metrics['Accuracy'])}  |  Test = {fmt_val(test_metrics['Accuracy'])}")
    print(f"MACV:     Train = {fmt_val(train_metrics['MACV'])}  |  Val = {fmt_val(val_metrics['MACV'])}  |  Test = {fmt_val(test_metrics['MACV'])}")
    
    #
    metrics = {
        'Train': train_metrics,
        'Val': val_metrics,
        'Test': test_metrics
    }
    metrics_path = os.path.join(name_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
    
    #
    predictions_path = os.path.join(name_dir, 'final_predictions.csv')
    P_test.to_csv(predictions_path, index=True)
    print(f"Saved predictions to {predictions_path}")
