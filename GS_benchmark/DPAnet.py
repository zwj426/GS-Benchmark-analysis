import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import OneCycleLR
import optuna
from scipy.stats import pearsonr
import warnings
import sys
import argparse
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')
import shap
import plotly.express as px
import random

# seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==============================
# 1. load data
# ==============================
def read_data(gen_file, phe_file):
    # 1.1 reference data
    gen = pd.read_csv(gen_file, index_col=0, sep='\s+', low_memory=False) #header=None, , sep='\s+'
    gen.columns = ['_'.join(col.split('_')[:-1]) if len(col.split('_')) >= 2 else col for col in gen.columns]
    phe = pd.read_csv(phe_file, index_col=0, sep='\s+')
    snp_sort = gen.columns
    
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
    gen_scaled = gen.to_numpy().astype(np.float32)
    # phenotype standardization
    phe_scaled = StandardScaler().fit_transform(phe.iloc[:, 1].values.reshape(-1, 1)).ravel().astype(np.float32)
    return gen_scaled, phe, phe_scaled, snp_sort


# ==============================
# Dynamic attention layer
# ==============================
class DynamicAttentionLayer(nn.Module):
    """Dynamic attention layer"""
    def __init__(self, input_dim, gwas_weights, maf, bayesB_squared, hidden_dim=32):
        super().__init__()
        self.register_buffer('gwas_weights', torch.tensor(gwas_weights, dtype=torch.float32))
        self.register_buffer('maf', torch.tensor(maf, dtype=torch.float32))
        self.register_buffer('bayesB_effects', torch.tensor(bayesB_squared, dtype=torch.float32))
        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        if x.shape[1] == self.gwas_weights.shape[0]:
            prior_features = torch.cat([
                self.gwas_weights.unsqueeze(1),
                self.maf.unsqueeze(1),
                self.bayesB_effects.unsqueeze(1)
            ], dim=1)   
            a_j = self.attention_net(prior_features).squeeze()
            return x * a_j.unsqueeze(0)
        else:
            return self.attention_net(x).squeeze()


class BayesianResidualLayer(nn.Module):
    """Prior Bayesian residual layer"""
    def __init__(self, input_dim, output_dim, combined_effects, use_individual_init=True):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if not isinstance(combined_effects, torch.Tensor):
            combined_effects = torch.tensor(combined_effects, dtype=torch.float32)
        self.register_buffer('combined_effects', combined_effects)
        self.use_individual_init = use_individual_init
        self._init_weights()
    def _init_weights(self):
        with torch.no_grad():
            sigma = torch.abs(self.combined_effects) 
            if self.use_individual_init:
                sigma_expanded = sigma.unsqueeze(1).expand(-1, self.W.size(1))
                self.W.data = torch.normal(mean=0.0, std=sigma_expanded)
            else:
                self.W.data.normal_(mean=0.0, std=sigma.mean().item())
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return torch.matmul(x, self.W)


class DPANet(nn.Module):
    """DPANet"""
    def __init__(self, input_dim, p_values, maf, bayesB_effects, 
                 hidden_dims=[128,64], dropout=0.5):
        super().__init__()
        if isinstance(bayesB_effects, np.ndarray):
            bayesB_effects = torch.tensor(bayesB_effects, dtype=torch.float32)
        # Attention layer
        w_j = -torch.log10(torch.tensor(p_values, dtype=torch.float32) + 1e-10)
        self.attention = DynamicAttentionLayer(input_dim, w_j, maf, bayesB_effects ** 2)
        # Residual block
        self.block1 = nn.Sequential(
            BayesianResidualLayer(input_dim, hidden_dims[0], bayesB_effects),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout)
        )
        self.projection = nn.Linear(input_dim, hidden_dims[0])
        layers = []
        for in_dim, out_dim in zip(hidden_dims, hidden_dims[1:]):
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout/2)
            ]
        self.hidden = nn.Sequential(*layers) 
        self.fc = nn.Linear(hidden_dims[-1], 1)
    def forward(self, x):
        x_att = self.attention(x)
        identity = self.projection(x)
        x = self.block1(x_att) + identity
        x = self.hidden(x)
        return self.fc(x).squeeze()



# ==============================
# Custom evaluation indicators
# ==============================
def pearson_loss(y_pred, y_true):
    if y_pred.dtype != torch.float32:
        y_pred = y_pred.to(torch.float32)
    if y_true.dtype != torch.float32:
        y_true = y_true.to(torch.float32)
    # correlation
    def pearson_corr(y_pred, y_true):
        y_pred_mean = y_pred - y_pred.mean()
        y_true_mean = y_true - y_true.mean()
        numerator = (y_pred_mean * y_true_mean).sum()
        denominator = torch.sqrt((y_pred_mean ** 2).sum() * (y_true_mean ** 2).sum()) + 1e-8
        return numerator / (denominator + 1e-6)  
    # unbiasedness
    def regression_slope(y_pred, y_true):
        cov = torch.cov(torch.stack([y_true, y_pred]))[0, 1]
        var_pred = torch.var(y_pred, unbiased=True)
        return cov / (var_pred + 1e-8) 
    
    corr_loss = - pearson_corr(y_pred, y_true)
    slope = regression_slope(y_pred, y_true)
    bias_loss = torch.abs(slope - 1.0) 
    return 0.8 * corr_loss + 0.2 * bias_loss 



# ==============================
# Hyperparameter optimization
# ==============================
class HyperparameterOptimizer:
    def __init__(self, X, y, p, maf, effects, out, n_trials=100, patience=15, epochs=100):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.float32)
        self.p = torch.tensor(p, dtype=torch.float32)
        self.maf = torch.tensor(maf, dtype=torch.float32)
        self.effects = torch.tensor(effects, dtype=torch.float32)
        self.n_trials = n_trials
        self.patience = patience
        self.out = out
        self.epochs = epochs
        self.best_params = None  
        self.best_model_state = None  
        self.global_best_params = None  
        self.global_best_model_state = None  
        self.global_best_val_loss = float('inf')  
    
    #### Hyperparameter optimization
    def objective(self, trial):
        params = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512]),
            'hidden_dims': [trial.suggest_int('h1', 64, 512), trial.suggest_int('h2', 32, 256)],
            'dropout': trial.suggest_float('dropout', 0.1, 0.7),
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
        }
        # Model initialization
        model = DPANet(
            input_dim=self.X.shape[1],
            p_values=self.p,
            maf=self.maf,
            bayesB_effects=self.effects,
            hidden_dims=params['hidden_dims'],
            dropout=params['dropout']
        )
        # Optimizer configuration
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        # Training
        train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=params['batch_size'],
            shuffle=True, drop_last=True
        )
        best_val_loss = float('inf')
        no_improve  = 0
        for epoch in range(self.epochs):  
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = pearson_loss(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(self.X_val)
                val_loss = pearson_loss(val_pred, self.y_val).item()
            
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                self.best_params = params
                self.best_model_state = model.state_dict()
                # If the optimal value of the current trial is better than the global optimal value, update the global optimal value
                if best_val_loss < self.global_best_val_loss:
                    self.global_best_val_loss = best_val_loss
                    self.global_best_params = params
                    self.global_best_model_state = model.state_dict()
                    # save
                    torch.save({'model_state_dict': self.best_model_state, 'best_params': self.best_params}, self.out + '_best_model_trail.pth')
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        return best_val_loss

    #### early stop
    def optimize(self):
        class EarlyStoppingCallback:
            def __init__(self, patience=20):
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
       
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(n_startup_trials=5),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, #5,
                n_warmup_steps=5,  #10,
                interval_steps=1
            )
        )
        early_stop = EarlyStoppingCallback(patience=20)
        study.optimize(self.objective, n_trials=self.n_trials, callbacks=[early_stop], n_jobs=18)  #, n_jobs=20
        return study.best_params


# ==============================
# evaluate
# ==============================
def evaluate(y_pred, y_true):
    metrics = {
        'Pearson': pearsonr(y_pred, y_true)[0],
        'Unbasis': np.cov(y_pred, y_true)[0, 1] / np.var(y_pred, ddof=1),
        'R2': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    return metrics


# =====================================================================================
# Load the model corresponding to the optimal hyperparameters for thorough training
# =====================================================================================
def train_model(ref_gen, ref_phe_scale, p_values, maf, bayesB_effect, out, epochs= 200, patience=30):
    print("\nTrain the final model with the best parameters...")
    # Load the model corresponding to the optimal hyperparameters for thorough training
    checkpoint = torch.load(out + '_best_model_trail.pth')
    best_params = checkpoint['best_params']
    final_model = DPANet(
        input_dim=ref_gen.shape[1],
        p_values=p_values,
        maf=maf,
        bayesB_effects=bayesB_effect,
        hidden_dims=[best_params['hidden_dims'][0], best_params['hidden_dims'][1]],
        dropout=best_params['dropout']
    )

    final_model.load_state_dict(checkpoint['model_state_dict'])
    X_train, X_val, y_train, y_val = train_test_split(
        torch.tensor(ref_gen, dtype=torch.float32),
        torch.tensor(ref_phe_scale, dtype=torch.float32),
        test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True, drop_last=True
    )

    optimizer = optim.AdamW(
        final_model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=best_params['lr'],
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )

    best_val_loss = float('inf')
    counter = 0
  
    for epoch in range(epochs):
        final_model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = final_model(batch_X)
            loss = pearson_loss(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        
        final_model.eval()
        with torch.no_grad():
            val_pred = final_model(X_val)
            val_loss = pearson_loss(val_pred, y_val).item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={epoch_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}")
       
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(final_model.state_dict(), out + '.best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stop trigger, optimal verification loss: {best_val_loss:.4f}")
                break
    final_model.load_state_dict(torch.load(out + '.best_model.pth'))
    return final_model


# ========================================
# SNP screening based on attention weight
# ========================================
def snp_importance_selection(model, X, top_k=1000):
    """SNP screening based on attention weight"""
    model.eval()
    with torch.no_grad():
        prior_features = torch.cat([
            model.attention.gwas_weights.unsqueeze(1),
            model.attention.maf.unsqueeze(1),
            model.attention.bayesB_effects.unsqueeze(1)
        ], dim=1)
        
        attn_weights = model.attention.attention_net(prior_features).squeeze().numpy()
    
    sorted_indices = np.argsort(attn_weights)[::-1]
    selected_snps = sorted_indices[:top_k]
    return selected_snps, attn_weights



# ==============================
# Genome-wide attention heat map
# ==============================
def plot_genome_attention(attn_weights, snp_positions, chromosome_lengths, out):
    """Genome-wide attention heat map"""
    data = []
    for chrom, length in chromosome_lengths.items():
        chrom_snps = [i for i, pos in enumerate(snp_positions) if pos[0] == int(chrom)]
        chrom_weights = attn_weights[chrom_snps]
        chrom_positions = [pos[1] for pos in snp_positions if pos[0] == int(chrom)]
        data.extend(zip([int(chrom)]*len(chrom_snps), chrom_positions, chrom_weights))
    df = pd.DataFrame(data, columns=['Chromosome', 'Position', 'Attention'])
    df['pos'] = df['Chromosome'].astype(str) + '_' + df['Position'].astype(str)

    fig = px.scatter(
        df, x='pos', y='Attention', color='Attention',
        title='Genome-wide Attention Weights',
        labels={'Position': 'Genomic Position', 'Attention': 'Attention Weight'}
    )
    fig.write_html(out + '_attention_heatmap.html')



# ==============================
# main
# ==============================
def main(args):
    Xgen = args.Xgen
    Xphe = args.Xphe
    Tgen = args.Tgen
    Tphe = args.Tphe
    out = args.out
    gwas_file = args.gwas
    bayesB_effect_file = args.effect

    # ==============================
    # load data
    # ==============================
    # ref
    print('Reading refer datasets...')
    ref_gen, _, ref_phe_scale, snp_sort = read_data(Xgen, Xphe)
    # tst
    print('Reading test datasets...')
    tst_gen, tst_phe, tst_phe_scale, _ = read_data(Tgen, Tphe)

    # load prior
    print("load GWAS...")
    gwas = pd.read_csv(gwas_file, index_col=1, sep='\s+')

    valid_indices = [x for x in snp_sort if x in gwas.index]
    gwas = gwas.loc[valid_indices]
    print(gwas.index.equals(snp_sort))
    p_values = gwas['p'].values.astype(np.float32)
    maf = gwas['Freq'].values.astype(np.float32)

    print("load SNP effect...")
    bayesB = pd.read_csv(bayesB_effect_file, index_col=0, sep='\s+')
    bayesB.index = ['_'.join(col.split('_')[:-1]) if len(col.split('_')) >= 2 else col for col in bayesB.index]
    valid_indices = [x for x in snp_sort if x in bayesB.index]
    bayesB = bayesB.loc[valid_indices]
    print(bayesB.index.equals(snp_sort))
    bayesB_effect = bayesB['effect'].values.astype(np.float32) 

    # ==============================
    # Optuna hyperparameter optimization
    # ==============================
    print("Optuna hyperparameter optimization...")
    start_time = time.time()
    optimizer = HyperparameterOptimizer(ref_gen, ref_phe_scale, p_values, maf, bayesB_effect, out)
    best_params = optimizer.optimize()
    traning_time = time.time() - start_time
    print(f"Hyperparameter training time: {traning_time} s")
    print("\nOptimal hyperparameter:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    # ==============================
    # Train the final model with the best parameters
    # ==============================
    start_time = time.time()
    final_model = train_model(ref_gen, ref_phe_scale, p_values, maf, bayesB_effect, out)
    model_time = time.time() - start_time
    print(f"Model training time: {model_time} s")

    # ==============================
    # Model evaluation
    # ==============================
    print("\nModel evaluation...")
    start_time = time.time()
    final_model.eval()
    with torch.no_grad():
        val_pred = final_model(torch.tensor(tst_gen, dtype=torch.float32)).detach().numpy()
        val_true = tst_phe_scale
        val_metrics = evaluate(val_pred, val_true)
    valid_time = time.time() - start_time
    print(f"Model evaluation time: {valid_time} 秒")

    #### result
    df = pd.DataFrame(val_pred)
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
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")
    df.to_csv(out + '_gebv_hyper', sep='\t')

    #### summary
    res = {
        'file': out,
        'best_params': best_params,
        'val_metrics': val_metrics,
        'cor_hyper': corr, 
        'basis_hyper' : basis,
        'hyper_time': traning_time,
        'train_time': model_time,
        'valid_time': valid_time
    }
    res = pd.DataFrame([res])
    print(res)
    res.to_csv(out + '_statistics', sep='\t')

    # Screen important SNPS
    selected_snps,  attn_weights = snp_importance_selection(final_model, ref_gen)
    snp_positions = list(zip(gwas['Chr'], gwas['bp']))
    chromosome_lengths = {
        '1': 158534110,'2': 136231102,'3': 121005158,'4': 120000601,'5': 120089316,'6': 117806340,'7': 110682743,'8': 113319770,
        '9': 105454467,'10': 103308737,'11': 106982474,'12': 87216183,'13': 83472345,'14': 82403003,'15': 85007780,'16': 81013979,
        '17': 73167244,'18': 65820629,'19': 63449741,'20': 71974595,'21': 69862954,'22': 60773035,'23': 52498615,'24': 62317253,
        '25': 42350435,'26': 51992305,'27': 45612108,'28': 45940150,'29': 51098607}    
    plot_genome_attention(attn_weights, snp_positions, chromosome_lengths, out)
    weights = gwas.loc[:, ['Chr','bp','Freq','b','p']]
    weights['weight'] = attn_weights
    weights.to_csv(out + '_SNP_weight', sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='DPAnet pipeline', description='**** DPAnet.py -h ****')
    # Add subcommands
    parser.add_argument("--Xgen", required=True, type=str,
                        help='Train geno file. (Tips: IID snp1 snp2...)')
    parser.add_argument("--Xphe", required=True, type=str,
                        help='Train pheno file')
    parser.add_argument("--Tgen", required=True, type=str,
                        help='Test geno file.  (Tips: IID snp1 snp2...)')
    parser.add_argument("--Tphe", required=True, type=str,
                        help='Test pheno file')
    parser.add_argument("--gwas", required=True, type=str,
                        help='GWAS summary by GCTA64')
    parser.add_argument("--effect", required=True, type=str,
                        help='SNP effect by Bayes')
    parser.add_argument("--out", required=False, default='./',
                        type=str, help='Output path')
    args = parser.parse_args(None if sys.argv[1:] else ['-h'])
    main(args)
