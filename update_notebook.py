# -*- coding: utf-8 -*-
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

# ── Cellule 0 : Titre ────────────────────────────────────────────────────
cells.append(md("""# Prediction de la Consommation Energetique
**Dataset principal :** energydata_ready_for_machine_learning.csv (3 098 observations horaires, jan-mai 2016)
**Types de batiments :** Residentiel / Commercial / Industriel
**Contexte local :** Donnees reelles Senelec 2022-2024
**Objectif :** Predire la consommation horaire (kWh) a partir de variables meteorologiques, comportementales et temporelles.
"""))

# ── Cellule 1 : Section imports ──────────────────────────────────────────
cells.append(md("## 1. Import des bibliotheques"))

cells.append(code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

print('Bibliotheques importees avec succes')
"""))

# ── Cellule 2 : Chargement ────────────────────────────────────────────────
cells.append(md("## 2. Chargement des donnees"))

cells.append(code("""# Charger le dataset enrichi (ML-ready)
df = pd.read_csv('energydata_ready_for_machine_learning.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f'Dimensions         : {df.shape[0]} observations x {df.shape[1]} variables')
print(f'Periode            : {df["date"].min().date()} -> {df["date"].max().date()}')
print(f'Frequence          : horaire')
print(f'Types de batiment  : {df["Type_batiment"].unique()}')
print(f'Valeurs manquantes : {df.isnull().sum().sum()}')
df.head(3)
"""))

# ── Cellule 3 : EDA ───────────────────────────────────────────────────────
cells.append(md("## 3. Analyse Exploratoire (EDA)"))

cells.append(code("""# Statistiques descriptives de la variable cible
print('=== Variable cible : Consommation_kWh ===')
print(df['Consommation_kWh'].describe().round(4))
print(f'\\nCoefficient de variation : {df["Consommation_kWh"].std()/df["Consommation_kWh"].mean()*100:.1f}%')
print(f'\\nRepartition par type de batiment :')
print(df.groupby('Type_batiment')['Consommation_kWh'].agg(['count','mean','std']).round(4))
"""))

cells.append(code("""# Fig 1 : Distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(df['Consommation_kWh'], bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(df['Consommation_kWh'].mean(), color='red', linestyle='--',
                label=f'Moyenne = {df["Consommation_kWh"].mean():.3f} kWh')
axes[0].axvline(df['Consommation_kWh'].median(), color='orange', linestyle='--',
                label=f'Mediane = {df["Consommation_kWh"].median():.3f} kWh')
axes[0].set_title('Distribution de la consommation')
axes[0].set_xlabel('Consommation (kWh)')
axes[0].legend(fontsize=9)

axes[1].boxplot(df['Consommation_kWh'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red', linewidth=2))
axes[1].set_title('Boxplot - Valeurs aberrantes')
axes[1].set_ylabel('Consommation (kWh)')

colors_type = {'Residentiel':'steelblue','Commercial':'darkorange','Industriel':'green'}
for typ, grp in df.groupby('Type_batiment'):
    axes[2].hist(grp['Consommation_kWh'], bins=30, alpha=0.6,
                 color=colors_type.get(typ,'gray'), label=typ, edgecolor='white')
axes[2].set_title('Distribution par type de batiment')
axes[2].set_xlabel('Consommation (kWh)')
axes[2].legend(fontsize=9)

plt.suptitle('Distribution de la Consommation Energetique', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig1_distribution.png', dpi=120, bbox_inches='tight')
plt.show()
"""))

cells.append(code("""# Fig 2 : Serie temporelle
fig, ax = plt.subplots(figsize=(15, 4))
ax.plot(df['date'], df['Consommation_kWh'], color='steelblue', alpha=0.5, linewidth=0.6)
df_daily = df.set_index('date')['Consommation_kWh'].resample('D').mean()
ax.plot(df_daily.index, df_daily.values, color='red', linewidth=2, label='Moyenne journaliere')
ax.set_title('Evolution temporelle de la consommation (jan-mai 2016)', fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Consommation (kWh)'); ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.tight_layout()
plt.savefig('fig2_serie_temporelle.png', dpi=120, bbox_inches='tight')
plt.show()
"""))

cells.append(code("""# Fig 3 : Patterns horaires et hebdomadaires
jours_fr = ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Par heure
hourly = df.groupby('Heure')['Consommation_kWh'].mean()
axes[0].bar(hourly.index, hourly.values, color='steelblue', alpha=0.85)
axes[0].set_title('Consommation moyenne par heure', fontweight='bold')
axes[0].set_xlabel('Heure de la journee'); axes[0].set_ylabel('Consommation moyenne (kWh)')
axes[0].set_xticks(range(0, 24, 2))

# Par jour de la semaine (1=Lun, 7=Dim)
weekly = df.groupby('Jour_semaine')['Consommation_kWh'].mean().sort_index()
labels_jours = [jours_fr[(j-1)%7] for j in weekly.index]
colors_w = ['steelblue' if j <= 5 else 'coral' for j in weekly.index]
axes[1].bar(labels_jours, weekly.values, color=colors_w, alpha=0.85)
axes[1].set_title('Consommation moyenne par jour', fontweight='bold')
axes[1].set_ylabel('Consommation moyenne (kWh)')
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle('Patterns Temporels de Consommation', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig3_patterns_temporels.png', dpi=120, bbox_inches='tight')
plt.show()
"""))

cells.append(code("""# Fig 4 : Matrice de correlation
features_corr = ['Consommation_kWh','Temperature','Humidite','Surface','Occupants',
                  'conso_H_1','conso_J_1','rolling_mean_6h','rolling_mean_24h','is_weekend']
features_corr = [c for c in features_corr if c in df.columns]
corr_matrix = df[features_corr].corr()

plt.figure(figsize=(11, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
            mask=mask, center=0, square=True, linewidths=0.5, cbar_kws={'shrink':0.8})
plt.title('Matrice de Correlation - Variables Principales', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('fig4_correlation.png', dpi=120, bbox_inches='tight')
plt.show()

print('\\nTop 5 correlations avec Consommation_kWh :')
print(corr_matrix['Consommation_kWh'].drop('Consommation_kWh').abs().sort_values(ascending=False).head(5))
"""))

cells.append(code("""# Fig 5 : Relation meteo / consommation
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(df['Temperature'], df['Consommation_kWh'], alpha=0.2, s=8, color='steelblue')
axes[0].set_xlabel('Temperature exterieure (C)')
axes[0].set_ylabel('Consommation (kWh)')
axes[0].set_title('Temperature vs Consommation')

axes[1].scatter(df['Humidite'], df['Consommation_kWh'], alpha=0.2, s=8, color='darkorange')
axes[1].set_xlabel('Humidite (%)')
axes[1].set_ylabel('Consommation (kWh)')
axes[1].set_title('Humidite vs Consommation')

plt.suptitle('Relations Meteo - Consommation', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig5_meteo_conso.png', dpi=120, bbox_inches='tight')
plt.show()
"""))

# ── Cellule 4 : Preprocessing ─────────────────────────────────────────────
cells.append(md("## 4. Preprocessing & Feature Engineering\n\nLe dataset enrichi contient deja le feature engineering (lags, moyennes mobiles, encodage cyclique). On effectue uniquement l'encodage one-hot du type de batiment."))

cells.append(code("""# Encodage one-hot du type de batiment
df_model = df.copy()
df_model = pd.get_dummies(df_model, columns=['Type_batiment'], prefix='type', drop_first=False)
type_cols = [c for c in df_model.columns if c.startswith('type_')]
for c in type_cols:
    df_model[c] = df_model[c].astype(int)

print(f'Dataset apres encodage : {df_model.shape}')
print(f'Colonnes type : {type_cols}')
"""))

cells.append(code("""# Detection des outliers (IQR x 3)
Q1 = df_model['Consommation_kWh'].quantile(0.25)
Q3 = df_model['Consommation_kWh'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 3 * IQR
upper = Q3 + 3 * IQR
outliers_count = ((df_model['Consommation_kWh'] < lower) | (df_model['Consommation_kWh'] > upper)).sum()
print(f'Outliers (IQR x 3) : {outliers_count} observations ({outliers_count/len(df_model)*100:.2f}%)')
print(f'Seuil superieur : {upper:.3f} kWh')
print('-> Valeurs conservees (pics reels de consommation)')
"""))

cells.append(code("""# Selection des features finales
feature_cols = [
    'Temperature', 'Humidite', 'Jour_ferie', 'Surface', 'Occupants',
    'Heure', 'Jour_semaine', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'conso_H_1', 'conso_H_2', 'conso_H_6', 'conso_J_1', 'conso_J_7',
    'rolling_mean_6h', 'rolling_mean_24h',
] + type_cols
feature_cols = [c for c in feature_cols if c in df_model.columns]

X = df_model[feature_cols]
y = df_model['Consommation_kWh']

# Split temporel chronologique (80/20)
split_idx = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f'Features utilisees : {len(feature_cols)}')
print(f'Train : {len(X_train)} obs ({len(X_train)/len(X)*100:.0f}%)')
print(f'Test  : {len(X_test)} obs ({len(X_test)/len(X)*100:.0f}%)')

# Normalisation pour les modeles lineaires
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print('Normalisation StandardScaler appliquee')
"""))

# ── Cellule 5 : Modelisation ──────────────────────────────────────────────
cells.append(md("## 5. Modelisation\n\nValidation par TimeSeriesSplit (5 folds chronologiques) — plus rigoureuse qu'une CV classique pour les series temporelles."))

cells.append(code("""def evaluate_model(model, X_tr, y_tr, X_te, y_te, name):
    \"\"\"Entraine, evalue et retourne les metriques d'un modele.\"\"\"
    model.fit(X_tr, y_tr)
    y_pred = np.maximum(model.predict(X_te), 0)

    r2   = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    mape = np.mean(np.abs((y_te - y_pred) / (y_te + 1e-9))) * 100

    # Validation croisee temporelle (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_r2 = []
    for tr_idx, te_idx in tscv.split(X_tr):
        import copy
        mc = copy.deepcopy(model)
        mc.fit(X_tr[tr_idx] if hasattr(X_tr,'iloc') is False else X_tr.iloc[tr_idx],
               y_tr.iloc[tr_idx] if hasattr(y_tr,'iloc') else y_tr[tr_idx])
        p = np.maximum(mc.predict(
            X_tr[te_idx] if hasattr(X_tr,'iloc') is False else X_tr.iloc[te_idx]), 0)
        cv_r2.append(r2_score(y_tr.iloc[te_idx] if hasattr(y_tr,'iloc') else y_tr[te_idx], p))

    cv_mean = np.mean(cv_r2)
    cv_std  = np.std(cv_r2)

    print(f'\\n{"="*52}')
    print(f'  {name}')
    print(f'{"="*52}')
    print(f'  R2   (test)     : {r2:.4f}')
    print(f'  RMSE (test)     : {rmse:.4f} kWh')
    print(f'  MAE  (test)     : {mae:.4f} kWh')
    print(f'  MAPE (test)     : {mape:.2f}%')
    print(f'  CV R2 (5 folds) : {cv_mean:.4f} +/- {cv_std:.4f}')

    return {
        'Modele': name, 'R2': round(r2,4), 'RMSE': round(rmse,4),
        'MAE': round(mae,4), 'MAPE (%)': round(mape,2),
        'CV R2 moyen': round(cv_mean,4)
    }, model, y_pred

results = []
models_trained = {}
"""))

cells.append(code("""# Modele 1 : Regression Lineaire (baseline)
lr = LinearRegression()
res, m, preds_lr = evaluate_model(lr, X_train_sc, y_train, X_test_sc, y_test, 'Regression Lineaire')
results.append(res)
models_trained['Regression Lineaire'] = (m, preds_lr)
"""))

cells.append(code("""# Modele 2 : Ridge (regularisation L2)
ridge = Ridge(alpha=100)
res, m, preds_ridge = evaluate_model(ridge, X_train_sc, y_train, X_test_sc, y_test, 'Ridge (L2)')
results.append(res)
models_trained['Ridge (L2)'] = (m, preds_ridge)
"""))

cells.append(code("""# Modele 3 : Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5,
                           random_state=42, n_jobs=-1)
res, m, preds_rf = evaluate_model(rf, X_train, y_train, X_test, y_test, 'Random Forest')
results.append(res)
models_trained['Random Forest'] = (m, preds_rf)
"""))

cells.append(code("""# Modele 4 : Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5,
                                subsample=0.8, min_samples_leaf=4, random_state=42)
res, m, preds_gb = evaluate_model(gb, X_train, y_train, X_test, y_test, 'Gradient Boosting')
results.append(res)
models_trained['Gradient Boosting'] = (m, preds_gb)
"""))

# ── Cellule 6 : Comparaison ───────────────────────────────────────────────
cells.append(md("## 6. Comparaison des Modeles"))

cells.append(code("""df_results = pd.DataFrame(results)
print('\\n=== TABLEAU COMPARATIF DES MODELES ===')
print(df_results.to_string(index=False))

best_name = df_results.loc[df_results['R2'].idxmax(), 'Modele']
print(f'\\nMeilleur modele : {best_name} (R2 = {df_results["R2"].max():.4f})')
"""))

cells.append(code("""# Fig 6 : Comparaison des modeles
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
model_names = df_results['Modele'].tolist()
palette = ['#7B2FBE','#D4227A','#F15A24','#A855F7']

# R2
bars0 = axes[0].barh(model_names, df_results['R2'], color=palette, alpha=0.85)
axes[0].set_title('R2 (Test)', fontweight='bold')
axes[0].set_xlim(0, 0.65)
for bar, val in zip(bars0, df_results['R2']):
    axes[0].text(val+0.005, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)

# RMSE
bars1 = axes[1].barh(model_names, df_results['RMSE'], color=palette, alpha=0.85)
axes[1].set_title('RMSE - kWh (bas = meilleur)', fontweight='bold')
for bar, val in zip(bars1, df_results['RMSE']):
    axes[1].text(val+0.002, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)

# CV R2
bars2 = axes[2].barh(model_names, df_results['CV R2 moyen'], color=palette, alpha=0.85)
axes[2].set_title('CV R2 moyen (TimeSeriesSplit)', fontweight='bold')
axes[2].set_xlim(0, 0.65)
for bar, val in zip(bars2, df_results['CV R2 moyen']):
    axes[2].text(val+0.005, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)

plt.suptitle('Comparaison des Modeles de Regression', fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('fig6_comparaison_modeles.png', dpi=120, bbox_inches='tight')
plt.show()
"""))

# ── Cellule 7 : Analyse meilleur modele ──────────────────────────────────
cells.append(md("## 7. Analyse du Meilleur Modele"))

cells.append(code("""# Recuperer le meilleur modele
best_name = df_results.loc[df_results['R2'].idxmax(), 'Modele']
best_preds = models_trained[best_name][1]
print(f'Meilleur modele : {best_name}')

# Fig 7 : Diagnostic
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test.values, best_preds, alpha=0.3, s=10, color='steelblue')
lim = max(y_test.max(), best_preds.max()) * 1.05
axes[0].plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Prediction parfaite')
axes[0].set_xlabel('Valeurs reelles (kWh)')
axes[0].set_ylabel('Valeurs predites (kWh)')
axes[0].set_title(f'Reel vs Predit - {best_name}')
axes[0].legend()

residuals = y_test.values - best_preds
axes[1].scatter(best_preds, residuals, alpha=0.3, s=10, color='darkorange')
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Valeurs predites (kWh)')
axes[1].set_ylabel('Residus (kWh)')
axes[1].set_title('Analyse des Residus')

plt.suptitle('Diagnostic du Meilleur Modele', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig7_diagnostic.png', dpi=120, bbox_inches='tight')
plt.show()
"""))

cells.append(code("""# Fig 8 : Importance des variables
best_model_obj = models_trained[best_name][0]
if hasattr(best_model_obj, 'feature_importances_'):
    importances = pd.Series(best_model_obj.feature_importances_, index=feature_cols)
    top_features = importances.sort_values(ascending=True).tail(15)
    plt.figure(figsize=(10, 6))
    colors_imp = ['#D4227A' if i >= 10 else '#7B2FBE' for i in range(len(top_features))]
    top_features.plot(kind='barh', color=colors_imp, alpha=0.85)
    plt.title(f'Top 15 Variables - {best_name}', fontweight='bold')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('fig8_importance.png', dpi=120, bbox_inches='tight')
    plt.show()
    print('\\nTop 5 variables :')
    print(importances.sort_values(ascending=False).head(5))
elif hasattr(best_model_obj, 'coef_'):
    importances = pd.Series(np.abs(best_model_obj.coef_), index=feature_cols)
    top_features = importances.sort_values(ascending=True).tail(15)
    plt.figure(figsize=(10, 6))
    top_features.plot(kind='barh', color='#7B2FBE', alpha=0.85)
    plt.title(f'Coefficients |beta| - {best_name}', fontweight='bold')
    plt.xlabel('|Coefficient|')
    plt.tight_layout()
    plt.savefig('fig8_importance.png', dpi=120, bbox_inches='tight')
    plt.show()
"""))

cells.append(code("""# Fig 9 : Predictions sur les 7 premiers jours du jeu de test
n_points = min(7*24, len(y_test))
dates_test = df_model['date'].iloc[split_idx:split_idx+n_points]

plt.figure(figsize=(15, 5))
plt.plot(dates_test.values, y_test.values[:n_points],
         color='steelblue', alpha=0.8, linewidth=1.5, label='Reel')
plt.plot(dates_test.values, best_preds[:n_points],
         color='red', alpha=0.8, linewidth=1.5, linestyle='--', label=f'Predit ({best_name})')
plt.title(f'Prediction vs Reel - 7 premiers jours du jeu de test', fontweight='bold')
plt.xlabel('Date'); plt.ylabel('Consommation (kWh)'); plt.legend()
plt.tight_layout()
plt.savefig('fig9_prediction_semaine.png', dpi=120, bbox_inches='tight')
plt.show()
"""))

# ── Cellule 8 : Contexte Senelec ──────────────────────────────────────────
cells.append(md("## 8. Contexte Senegalais - Analyse Senelec 2022-2024"))

cells.append(code("""# Donnees reelles Senelec
pointes_senelec = {
    2022: [677,679,684,808,875,900,910,915,920,957,900,841],
    2023: [784,771,873,893,948,995,1021,1018,1006,1075,1039,896],
    2024: [899,917,935,1084,1062,1092,1109,1159,1090,1141,1090,1015],
}
production_senelec = {
    2022: {'Thermique':4645.79,'Hydraulique':485.78,'Solaire':381.19,'Eolien':395.56},
    2023: {'Thermique':5357.80,'Hydraulique':540.35,'Solaire':373.48,'Eolien':382.47},
    2024: {'Thermique':6029.92,'Hydraulique':651.73,'Solaire':410.38,'Eolien':373.83},
}
mois = ['Jan','Fev','Mar','Avr','Mai','Jun','Jul','Aou','Sep','Oct','Nov','Dec']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_yr = {2022:'#7B2FBE', 2023:'#D4227A', 2024:'#F15A24'}

for yr, pts in pointes_senelec.items():
    axes[0].plot(mois, pts, marker='o', label=str(yr), linewidth=2, color=colors_yr[yr])
axes[0].set_title('Pointe Maximale Mensuelle - Reseau Senelec (MW)', fontweight='bold')
axes[0].set_ylabel('MW'); axes[0].legend(); axes[0].tick_params(axis='x', rotation=30)

labels = list(production_senelec[2024].keys())
vals   = list(production_senelec[2024].values())
colors_pie = ['#D4227A','#7B2FBE','#F15A24','#2ca02c']
wedges, texts, autotexts = axes[1].pie(
    vals, labels=labels, autopct='%1.1f%%', colors=colors_pie,
    startangle=90, pctdistance=0.82)
for at in autotexts: at.set_fontsize(10)
axes[1].set_title('Mix Energetique Senelec 2024\\n(Total : 7 466 GWh)', fontweight='bold')

plt.suptitle('Contexte Energetique Senegal - Donnees Reelles Senelec', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig10_senelec.png', dpi=120, bbox_inches='tight')
plt.show()
print('Source : Rapports Annuels Senelec 2022, 2023, 2024')
"""))

# ── Cellule 9 : Resume ────────────────────────────────────────────────────
cells.append(md("## 9. Resume et Conclusions"))

cells.append(code("""print('=' * 62)
print('   RESUME - PROJET 2 : PREDICTION CONSOMMATION ENERGETIQUE')
print('=' * 62)

print('\\nDONNEES :')
print('  Dataset : energydata_ready_for_machine_learning.csv')
print(f'  Observations : {len(df)} (horaires) | Jan-Mai 2016')
print(f'  Types : Residentiel / Commercial / Industriel')
print(f'  Features : {len(feature_cols)} variables apres encodage')
print(f'  Split : 80% train ({split_idx} obs) / 20% test ({len(X_test)} obs)')

print('\\nMODELES COMPARES (TimeSeriesSplit - 5 folds) :')
print(df_results[['Modele','R2','RMSE','MAPE (%)','CV R2 moyen']].to_string(index=False))

best = df_results.loc[df_results['R2'].idxmax()]
print(f'\\nMEILLEUR MODELE : {best["Modele"]}')
print(f'   R2={best["R2"]:.4f} | RMSE={best["RMSE"]:.4f} kWh | MAPE={best["MAPE (%)"]:.2f}% | CV R2={best["CV R2 moyen"]:.4f}')

print('\\nCONTEXTE SENEGAL (Senelec) :')
print('  Production totale 2024 : 7 466 GWh (+26% vs 2022)')
print('  Pointe max 2024        : 1 159 MW (aout 2024)')
print('  Part renouvelable 2024 : 19.2%')

print('\\nLivrables :')
print('  - Notebook Jupyter commente (ce fichier)')
print('  - Dashboard interactif Streamlit (dashboard.py)')
print('  - Rapport ecrit (Rapport_projet.docx)')
print('  - 10 figures de visualisation (fig1-fig10)')
print('  - Dataset ML-ready (energydata_ready_for_machine_learning.csv)')
print('=' * 62)
"""))

nb.cells = cells

# Sauvegarder
with open('Projet_Prediction_Consommation_Energetique.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Notebook mis a jour avec succes !')
print(f'Nombre de cellules : {len(cells)}')
