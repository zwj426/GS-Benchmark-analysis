import os
import optuna
import numpy as np
import sys
import argparse
import pandas as pd
from sklearn import preprocessing
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import make_scorer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.model_selection import cross_validate
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.metrics import  make_scorer
from sklearn.model_selection import train_test_split


# ==============================================
# 1. pars
# ==============================================
parser = argparse.ArgumentParser(
    prog='ML pipeline', description='**** ML.py -h ****')
# Add subcommands
parser.add_argument("--Xgen", required=True, type=str,
                    help='Train geno file. (Tips: IID snp1 snp2...)')
parser.add_argument("--Xphe", required=True, type=str,
                    help='Train pheno file')
parser.add_argument("--Tgen", required=True, type=str,
                    help='Test geno file.  (Tips: IID snp1 snp2...)')
parser.add_argument("--Tphe", required=True, type=str,
                    help='Test pheno file')
parser.add_argument("-m", "--model", required=True, type=str,
                    help='ML model [SVR, RF, KRR]')
parser.add_argument("-o", "--out", required=False, default='./',
                    type=str, help='Output path')
args = parser.parse_args(None if sys.argv[1:] else ['-h'])

Xgen = args.Xgen
Xphe = args.Xphe
Tgen = args.Tgen
Tphe = args.Tphe
model = args.model
out = args.out
pca = None


# ==============================================
# 2. read datasets
# ==============================================
def read_data(gen_file, phe_file, fit_scaler=None, pca_transformer=None, pca=None):
    # 1.1 reference data
    gen = pd.read_csv(gen_file, index_col=0, sep='\s+', low_memory=False) #header=None, , sep='\s+'
    gen.columns = gen.columns.str.split('_').str[0]
    phe = pd.read_csv(phe_file, index_col=0, sep='\s+')

    # check NA
    if gen.isnull().any().any():
        raise ValueError("There are missing values in the genotype data!")
    if phe.isnull().any().any():
        raise ValueError("There are missing values in the phenotype data!")
        
    if gen.index.equals(phe.index):
        print("The order of individuals is consistent.")
    else:
        print("Error: The order of individuals is unconsistent.")
        
    # Genotype standardization
    if fit_scaler is None:
        geno_scaler = MinMaxScaler().fit(gen)
        pca_transformer = PCA(n_components=pca).fit(geno_scaler.transform(gen)) if pca else None
    else:
        geno_scaler = fit_scaler
    
    gen_scaled = geno_scaler.transform(gen)
    if pca_transformer:
        gen_scaled = pca_transformer.transform(gen_scaled)
    
    # phenotype standardization
    phe_scaled = StandardScaler().fit_transform(phe.iloc[:, 1].values.reshape(-1, 1)).ravel()
    
    return gen_scaled, phe, phe_scaled


# training data
print('Reading training datasets...')
all = read_data(Xgen, Xphe)
gen, phe, phe_scale = all

# ref 
print('Reading refer datasets...')
all = read_data(Xgen, Xphe)
ref_gen, ref_phe, ref_phe_scale = all

# tst
print('Reading test datasets...')
all = read_data(Tgen, Tphe)
tst_gen, tst_phe, tst_phe_scale = all

#### Divide the training set and the validation set
train_idx, val_idx = train_test_split(range(len(phe_scale)), test_size=0.2, random_state=42)
train_geno = gen[train_idx]
train_pheno = phe_scale[train_idx]
valid_geno = gen[val_idx]
valid_pheno = phe_scale[val_idx]


# ==============================================
# 2. Training model
# ==============================================
start_time = time.time()
def correlation_score(y_true, y_pred):
    if np.var(y_true) < 1e-6 or np.var(y_pred) < 1e-6:
        return 0.0  
    corr, _ = pearsonr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0

def unbiasedness_score(y_true, y_pred):
    var_pred = np.var(y_pred, ddof=1)
    if var_pred < 1e-6:  
        return 0.0
    bias = np.cov(y_true, y_pred, ddof=1)[0, 1] / var_pred
    return bias 

scoring = {
    'correlation': make_scorer(correlation_score),
    'unbiasedness': make_scorer(unbiasedness_score)
}


cv_result = []
def train_model(trial):
    if model == 'KRR':
        max_kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]) 
        max_gamma = trial.suggest_float("gamma", 1e-3, 10, log=True)  
        max_alpha = trial.suggest_float("alpha", 0, 1)
        reg = KernelRidge(kernel=max_kernel, alpha=max_alpha, gamma=max_gamma)
        reg.fit(train_geno, train_pheno)
        pred_def = reg.predict(valid_geno)
        corr = correlation_score(valid_pheno, pred_def)
        unbasis = unbiasedness_score(valid_pheno, pred_def)
        score = - 0.8 * corr + 0.2 * abs(1- unbasis)
        cv_result.append({
            'kernel': max_kernel,
            'gamma': max_gamma,
            'alpha': max_alpha,
            'correlation': corr,
            'unbiasedness': unbasis
        })
        return score
    
    elif model == 'SVR':
        max_kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])  
        max_gamma = trial.suggest_float("gamma", 1e-3, 10, log=True)  
        max_C = trial.suggest_float("C", 0.1, 100, log=True)
        reg = SVR(kernel=max_kernel, gamma=max_gamma, C=max_C)
        reg.fit(train_geno, train_pheno)
        pred_def = reg.predict(valid_geno)
        corr = correlation_score(valid_pheno, pred_def)
        unbasis = unbiasedness_score(valid_pheno, pred_def)
        score = - 0.8 * corr + 0.2 * abs(1- unbasis)
        cv_result.append({
            'kernel': max_kernel,
            'gamma': max_gamma,
            'max_C': max_C,
            'correlation': corr,
            'unbiasedness': unbasis
        })
        return score
    
    elif model == 'RF':
        n_estimators = trial.suggest_int("n_estimators", 100, 1000, 50)
        max_depth = trial.suggest_int("max_depth", 3, 30, 2)
        max_features = trial.suggest_float("max_features", 0.1, 1)
        reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,max_features=max_features)
        reg.fit(train_geno, train_pheno)
        pred_def = reg.predict(valid_geno)
        corr = correlation_score(valid_pheno, pred_def)
        unbasis = unbiasedness_score(valid_pheno, pred_def)
        score = - 0.8 * corr + 0.2 * abs(1- unbasis)
        cv_result.append({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'max_features': max_features,
            'correlation': corr,
            'unbiasedness': unbasis
        })
        return score


    
#### EarlyStopping
def optimizer_optuna(max_trials=100, patience=10):
    class EarlyStoppingCallback:
        def __init__(self, patience):
            self.patience = patience
            self.best_value = float('inf') 
            self.counter = 0
        def __call__(self, study, trial):
            current_value = study.best_value
            if current_value < self.best_value:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    study.stop()
    def optuna_objective(trial):
        try:
            score = train_model(trial)
            return round(score, 4)
        except Exception as e:  # 捕获所有异常类型
            print(f"Error in trial {trial.number}: {str(e)}")
            raise optuna.exceptions.TrialPruned()        
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    early_stop = EarlyStoppingCallback(patience)
    study.optimize(optuna_objective, n_trials=max_trials, callbacks=[early_stop])
    return study, study.best_params

#### Training
study, best_params = optimizer_optuna() 
end_time = time.time()
traning_time = end_time - start_time
print(f"Training time: {traning_time} s")

#### result
print(f"Total number of trials: {len(study.trials)}")
completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
print(f"Completed trials: {len(completed_trials)}")
best_trial = study.best_trial
print("Best trial:")
print(f"  Value: {best_trial.value}")  
print(f"  Params: {best_trial.params}")  

cv_df = pd.DataFrame(cv_result)
cv_df.to_csv(out + '_cv_result')

final = pd.DataFrame(list(best_params.items()), columns=['param', 'value'])
final.to_csv(out + '_best_pars', index=None)  


# ==============================================
# 5. genomic prediction using Hyperparameter
# ==============================================
hyper = final
if model == 'KRR':
    rf2 = KernelRidge(kernel=hyper.loc['kernel', 'value'], gamma=float(hyper.loc['gamma', 'value']), alpha=float(hyper.loc['alpha', 'value']))
elif model == 'SVR':
    rf2 = SVR(kernel=hyper.loc['kernel', 'value'] , gamma=float(hyper.loc['gamma', 'value']), C=float(hyper.loc['C', 'value']))
elif model == 'RF':
    rf2 = RandomForestRegressor(n_estimators=float(hyper.loc['n_estimators', 'value']), max_depth=float(hyper.loc['max_depth', 'value']),
                                max_features=float(hyper.loc['max_features', 'value']))

start_time = time.time()
rf2.fit(ref_gen, ref_phe_scale)
pred = rf2.predict(tst_gen)
end_time = time.time()
hyper_time = end_time - start_time
print(f"Running time: {hyper_time} s")

df = pd.DataFrame(pred)
df.columns = ['dgv']
df.index = tst_phe.index
df[['rel', 'drp']] = tst_phe
tst_phe_scale = pd.Series(tst_phe_scale)
tst_phe_scale.index = tst_phe.index
df['drp_scale'] = tst_phe_scale
corr = (df['dgv'].corr(df['drp_scale'])) / np.sqrt(df['rel']).mean()
basis = (np.cov(df['dgv'], df['drp_scale'])[0, 1] / np.var(df['dgv'], ddof=1))
print("Prediction accuracy: %f" % corr)
print("Prediction unbiasedness: %f" % basis)
df.to_csv(out + '_gebv_hyper', sep='\t')


# =========================================================
# 6. genomic prediction using default parameter (Optional)
# =========================================================
if model == 'KRR':
   rf2_def = KernelRidge()
elif model == 'SVR':
   rf2_def = SVR()
elif model == 'RF':
   rf2_def = RandomForestRegressor()

start_time = time.time()
rf2_def.fit(ref_gen, ref_phe_scale)
pred_def = rf2_def.predict(tst_gen)
end_time = time.time()
default_time = end_time - start_time
print(f"Running time: {default_time} s")

df_def = pd.DataFrame(pred_def)
df_def.columns = ['dgv']
df_def.index = tst_phe.index
df_def[['rel', 'drp']] = tst_phe
df_def['drp_scale'] = tst_phe_scale
corr_def = (df_def['dgv'].corr(df_def['drp_scale'])) / np.sqrt(df_def['rel']).mean()
basisdf_def = (np.cov(df_def['dgv'], df_def['drp_scale'])[0, 1] / np.var(df_def['dgv'], ddof=1))
print("Prediction accuracy: %f" % corr_def)
print("Prediction unbiasedness: %f" % basisdf_def)
df_def.to_csv(out + '_gebv_default', sep='\t')

#### summary
res = {'file': out, 'cor_hyper': corr, 'basis_hyper' : basis, 'cor_default': corr_def, 'basis_default' : basisdf_def, 'hyper_time':hyper_time,'default_time':default_time}
res = pd.DataFrame([res])
res.to_csv(out + '_statistics', sep='\t')