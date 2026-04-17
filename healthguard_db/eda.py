"""
HealthGuard — Exploratory Data Analysis (EDA)
Datasets: Heart Disease, Diabetes, Kidney Disease
Generates: eda_report/ folder with PNG graphs + eda_summary.txt
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data")
OUT    = os.path.join(BASE, "eda_report")
os.makedirs(OUT, exist_ok=True)

# ── PALETTE ───────────────────────────────────────────────
BLUE   = "#1565c0"
RED    = "#c62828"
GREEN  = "#2e7d32"
ORANGE = "#e65100"
PURPLE = "#6a1b9a"
TEAL   = "#00695c"
PINK   = "#ad1457"
DARK   = "#212121"

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#fafafa",
    "axes.edgecolor":   "#bdbdbd",
    "axes.grid":        True,
    "grid.color":       "#e0e0e0",
    "font.family":      "DejaVu Sans",
})

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}")

# ══════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════
heart   = pd.read_csv(os.path.join(DATA, "heart.csv"))
diabetes= pd.read_csv(os.path.join(DATA, "diabetes.csv"))
kidney  = pd.read_csv(os.path.join(DATA, "kidney.csv"))

# Clean kidney
kidney.replace('?', np.nan, inplace=True)
kidney.drop(columns=['id'], errors='ignore', inplace=True)
kidney['classification'] = kidney['classification'].str.strip()
kidney['age'] = pd.to_numeric(kidney['age'], errors='coerce')
kidney['bp']  = pd.to_numeric(kidney['bp'],  errors='coerce')
kidney['bgr'] = pd.to_numeric(kidney['bgr'], errors='coerce')
kidney['hemo']= pd.to_numeric(kidney['hemo'],errors='coerce')
kidney['sc']  = pd.to_numeric(kidney['sc'],  errors='coerce')
kidney['label'] = (kidney['classification'] == 'ckd').astype(int)

heart['label']    = heart['condition']
diabetes['label'] = diabetes['Outcome']

print("=== DATASET SHAPES ===")
print(f"Heart:    {heart.shape}")
print(f"Diabetes: {diabetes.shape}")
print(f"Kidney:   {kidney.shape}")

# ══════════════════════════════════════════════════════════
# 1. DATASET OVERVIEW — Target Distribution (all 3)
# ══════════════════════════════════════════════════════════
print("\n[1] Target distribution overview...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("HealthGuard — Target Class Distribution Across Datasets",
             fontsize=15, fontweight='bold', color=DARK, y=1.01)

datasets = [
    (heart,    "condition", ["No Disease", "Heart Disease"], [TEAL, RED],   "Heart Disease"),
    (diabetes, "Outcome",   ["No Diabetes", "Diabetes"],     [BLUE, ORANGE],"Diabetes"),
]

for ax, (df, col, labels, clrs, title) in zip(axes[:2], datasets):
    counts = df[col].value_counts().sort_index()
    bars = ax.bar(labels, counts.values, color=clrs, edgecolor='white', linewidth=1.5, width=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                f"{v}\n({v/len(df)*100:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', color=DARK)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_ylim(0, max(counts.values)*1.2)

# Kidney pie
ax3 = axes[2]
kcounts = kidney['classification'].value_counts()
wedges, texts, autotexts = ax3.pie(
    kcounts, labels=['CKD','Not CKD'], colors=[RED, GREEN],
    autopct='%1.1f%%', startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor='white', linewidth=2))
for at in autotexts:
    at.set_fontsize(11); at.set_fontweight('bold')
ax3.set_title("Kidney Disease", fontsize=13, fontweight='bold', color=DARK)

plt.tight_layout()
save(fig, "01_target_distribution.png")

# ══════════════════════════════════════════════════════════
# 2. HEART — Feature Distributions
# ══════════════════════════════════════════════════════════
print("[2] Heart feature distributions...")
num_cols = ['age','trestbps','chol','thalach','oldpeak']
fig, axes = plt.subplots(2, 5, figsize=(20, 9))
fig.suptitle("Heart Disease — Numeric Feature Distributions by Class",
             fontsize=14, fontweight='bold', color=DARK)

for i, col in enumerate(num_cols):
    # Top row: histogram
    ax = axes[0, i]
    for val, color, label in [(0, TEAL, "No Disease"), (1, RED, "Disease")]:
        subset = heart[heart['condition']==val][col].dropna()
        ax.hist(subset, bins=20, alpha=0.65, color=color, label=label, edgecolor='white')
    ax.set_title(col.upper(), fontsize=11, fontweight='bold')
    ax.set_xlabel(col, fontsize=9)
    if i == 0: ax.set_ylabel("Frequency", fontsize=9)
    ax.legend(fontsize=8)

    # Bottom row: KDE
    ax2 = axes[1, i]
    for val, color in [(0, TEAL), (1, RED)]:
        subset = heart[heart['condition']==val][col].dropna()
        subset.plot.kde(ax=ax2, color=color, linewidth=2)
    ax2.set_xlabel(col, fontsize=9)
    if i == 0: ax2.set_ylabel("Density", fontsize=9)

plt.tight_layout()
save(fig, "02_heart_feature_distributions.png")

# ══════════════════════════════════════════════════════════
# 3. HEART — Correlation Heatmap
# ══════════════════════════════════════════════════════════
print("[3] Heart correlation heatmap...")
fig, ax = plt.subplots(figsize=(12, 9))
corr = heart.select_dtypes(include=np.number).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
ax.set_title("Heart Disease — Feature Correlation Matrix",
             fontsize=14, fontweight='bold', color=DARK, pad=15)
plt.tight_layout()
save(fig, "03_heart_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════
# 4. HEART — Age & Chest Pain Analysis
# ══════════════════════════════════════════════════════════
print("[4] Heart age & chest pain analysis...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Heart Disease — Key Risk Factor Analysis", fontsize=14, fontweight='bold', color=DARK)

# Age boxplot
ax1 = axes[0]
heart.boxplot(column='age', by='condition', ax=ax1,
              boxprops=dict(color=BLUE), medianprops=dict(color=RED, linewidth=2),
              whiskerprops=dict(color=DARK), capprops=dict(color=DARK))
ax1.set_title("Age Distribution by Disease Status", fontsize=11, fontweight='bold')
ax1.set_xlabel("0 = No Disease  |  1 = Disease", fontsize=9)
ax1.set_ylabel("Age", fontsize=9)
plt.sca(ax1); plt.title("")

# Chest pain type
ax2 = axes[1]
cp_counts = heart.groupby(['cp','condition']).size().unstack(fill_value=0)
cp_counts.plot(kind='bar', ax=ax2, color=[TEAL, RED], edgecolor='white', width=0.6)
ax2.set_title("Chest Pain Type vs Disease", fontsize=11, fontweight='bold')
ax2.set_xlabel("Chest Pain Type (0-3)", fontsize=9)
ax2.set_ylabel("Count", fontsize=9)
ax2.legend(["No Disease", "Disease"], fontsize=9)
ax2.tick_params(axis='x', rotation=0)

# Sex vs disease
ax3 = axes[2]
sex_counts = heart.groupby(['sex','condition']).size().unstack(fill_value=0)
sex_counts.index = ['Female', 'Male']
sex_counts.plot(kind='bar', ax=ax3, color=[TEAL, RED], edgecolor='white', width=0.5)
ax3.set_title("Sex vs Heart Disease", fontsize=11, fontweight='bold')
ax3.set_xlabel("Sex", fontsize=9); ax3.set_ylabel("Count", fontsize=9)
ax3.legend(["No Disease", "Disease"], fontsize=9)
ax3.tick_params(axis='x', rotation=0)

plt.tight_layout()
save(fig, "04_heart_risk_factors.png")

# ══════════════════════════════════════════════════════════
# 5. DIABETES — Feature Distributions
# ══════════════════════════════════════════════════════════
print("[5] Diabetes feature distributions...")
d_num = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age','DiabetesPedigreeFunction']
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle("Diabetes — Feature Distributions by Outcome",
             fontsize=14, fontweight='bold', color=DARK)
axes_flat = axes.flatten()
for i, col in enumerate(d_num):
    ax = axes_flat[i]
    for val, color, label in [(0, BLUE, "No Diabetes"), (1, ORANGE, "Diabetes")]:
        subset = diabetes[diabetes['Outcome']==val][col].dropna()
        ax.hist(subset, bins=22, alpha=0.65, color=color, label=label, edgecolor='white')
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7)

axes_flat[-1].axis('off')  # hide last empty
plt.tight_layout()
save(fig, "05_diabetes_feature_distributions.png")

# ══════════════════════════════════════════════════════════
# 6. DIABETES — Correlation Heatmap
# ══════════════════════════════════════════════════════════
print("[6] Diabetes correlation heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
corr = diabetes.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
ax.set_title("Diabetes — Feature Correlation Matrix",
             fontsize=14, fontweight='bold', color=DARK, pad=15)
plt.tight_layout()
save(fig, "06_diabetes_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════
# 7. DIABETES — Scatter & Box Plots
# ══════════════════════════════════════════════════════════
print("[7] Diabetes scatter & box plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Diabetes — Key Variable Relationships", fontsize=14, fontweight='bold', color=DARK)

# Glucose vs BMI scatter
ax1 = axes[0]
for val, color, label in [(0, BLUE, "No Diabetes"), (1, ORANGE, "Diabetes")]:
    sub = diabetes[diabetes['Outcome']==val]
    ax1.scatter(sub['Glucose'], sub['BMI'], alpha=0.4, color=color, s=20, label=label)
ax1.set_xlabel("Glucose", fontsize=10); ax1.set_ylabel("BMI", fontsize=10)
ax1.set_title("Glucose vs BMI", fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)

# Age boxplot
ax2 = axes[1]
diabetes.boxplot(column='Age', by='Outcome', ax=ax2,
                 boxprops=dict(color=PURPLE),
                 medianprops=dict(color=RED, linewidth=2),
                 whiskerprops=dict(color=DARK))
ax2.set_title("Age by Diabetes Outcome", fontsize=11, fontweight='bold')
ax2.set_xlabel("0 = No Diabetes  |  1 = Diabetes", fontsize=9)
ax2.set_ylabel("Age", fontsize=9); plt.sca(ax2); plt.title("")

# Insulin distribution violin
ax3 = axes[2]
parts = ax3.violinplot(
    [diabetes[diabetes['Outcome']==0]['Insulin'].dropna(),
     diabetes[diabetes['Outcome']==1]['Insulin'].dropna()],
    positions=[0, 1], showmedians=True, showextrema=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor([BLUE, ORANGE][i]); pc.set_alpha(0.7)
parts['cmedians'].set_color(RED); parts['cmedians'].set_linewidth(2)
ax3.set_xticks([0, 1]); ax3.set_xticklabels(["No Diabetes", "Diabetes"])
ax3.set_title("Insulin Level Distribution", fontsize=11, fontweight='bold')
ax3.set_ylabel("Insulin", fontsize=10)

plt.tight_layout()
save(fig, "07_diabetes_relationships.png")

# ══════════════════════════════════════════════════════════
# 8. KIDNEY — Missing Value Heatmap
# ══════════════════════════════════════════════════════════
print("[8] Kidney missing values heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Kidney Disease — Missing Values Analysis", fontsize=14, fontweight='bold', color=DARK)

# Missing % bar
ax1 = axes[0]
miss = kidney.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)
colors_bar = [RED if v > kidney.shape[0]*0.2 else ORANGE for v in miss.values]
bars = ax1.barh(miss.index, miss.values / len(kidney) * 100, color=colors_bar, edgecolor='white')
ax1.set_xlabel("Missing %", fontsize=10)
ax1.set_title("Missing Values per Feature (%)", fontsize=12, fontweight='bold')
ax1.axvline(20, color=RED, linestyle='--', linewidth=1.5, label="20% threshold")
ax1.legend(fontsize=9)
for bar, v in zip(bars, miss.values):
    ax1.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
             f"{v/len(kidney)*100:.1f}%", va='center', fontsize=8)

# Missingness heatmap (sample)
ax2 = axes[1]
sample = kidney.isnull().astype(int).sample(min(100, len(kidney)), random_state=42)
sns.heatmap(sample.T, ax=ax2, cmap=['#e8f5e9', RED], cbar=False,
            xticklabels=False, yticklabels=True)
ax2.set_title("Missingness Pattern (100 sample rows)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Samples", fontsize=9)

plt.tight_layout()
save(fig, "08_kidney_missing_values.png")

# ══════════════════════════════════════════════════════════
# 9. KIDNEY — Feature Distributions
# ══════════════════════════════════════════════════════════
print("[9] Kidney feature distributions...")
k_num = ['age','bp','bgr','hemo','sc']
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle("Kidney Disease — Numeric Feature Distributions by Class",
             fontsize=14, fontweight='bold', color=DARK)

for i, col in enumerate(k_num):
    ax_top = axes[0, i]
    ax_bot = axes[1, i]
    for val, color, label in [(1, RED, "CKD"), (0, GREEN, "Not CKD")]:
        sub = kidney[kidney['label']==val][col].dropna()
        if len(sub) == 0: continue
        ax_top.hist(sub, bins=20, alpha=0.65, color=color, label=label, edgecolor='white')
        sub.plot.kde(ax=ax_bot, color=color, linewidth=2)
    ax_top.set_title(col.upper(), fontsize=11, fontweight='bold')
    if i == 0: ax_top.set_ylabel("Frequency", fontsize=9)
    ax_top.legend(fontsize=8)
    ax_bot.set_xlabel(col, fontsize=9)
    if i == 0: ax_bot.set_ylabel("Density", fontsize=9)

plt.tight_layout()
save(fig, "09_kidney_feature_distributions.png")

# ══════════════════════════════════════════════════════════
# 10. KIDNEY — Categorical Features
# ══════════════════════════════════════════════════════════
print("[10] Kidney categorical features...")
cat_cols = ['htn','dm','cad','appet','pe','ane','rbc','pc']
cat_cols = [c for c in cat_cols if c in kidney.columns]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle("Kidney Disease — Categorical Feature vs CKD Status",
             fontsize=14, fontweight='bold', color=DARK)
axes_flat = axes.flatten()

for i, col in enumerate(cat_cols[:8]):
    ax = axes_flat[i]
    ct = kidney.groupby([col, 'classification']).size().unstack(fill_value=0)
    if ct.empty: ax.axis('off'); continue
    ct.plot(kind='bar', ax=ax, color=[RED, GREEN], edgecolor='white', width=0.6)
    ax.set_title(col.upper(), fontsize=10, fontweight='bold')
    ax.set_xlabel(""); ax.set_ylabel("Count", fontsize=8)
    ax.legend(["CKD", "Not CKD"], fontsize=7)
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
save(fig, "10_kidney_categorical_features.png")

# ══════════════════════════════════════════════════════════
# 11. ALL DATASETS — Outlier Detection (Boxplots)
# ══════════════════════════════════════════════════════════
print("[11] Outlier detection boxplots...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Outlier Detection — Boxplots for All Datasets",
             fontsize=14, fontweight='bold', color=DARK)

# Heart
ax1 = axes[0]
h_num = heart.select_dtypes(include=np.number).drop(columns=['condition'], errors='ignore')
h_norm = (h_num - h_num.mean()) / h_num.std()
h_norm.boxplot(ax=ax1, boxprops=dict(color=RED),
               medianprops=dict(color=DARK, linewidth=2),
               whiskerprops=dict(color=TEAL))
ax1.set_title("Heart Disease", fontsize=12, fontweight='bold', color=RED)
ax1.tick_params(axis='x', rotation=45)
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)

# Diabetes
ax2 = axes[1]
d_num2 = diabetes.select_dtypes(include=np.number).drop(columns=['Outcome'], errors='ignore')
d_norm = (d_num2 - d_num2.mean()) / d_num2.std()
d_norm.boxplot(ax=ax2, boxprops=dict(color=ORANGE),
               medianprops=dict(color=DARK, linewidth=2),
               whiskerprops=dict(color=BLUE))
ax2.set_title("Diabetes", fontsize=12, fontweight='bold', color=ORANGE)
ax2.tick_params(axis='x', rotation=45)
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)

# Kidney
ax3 = axes[2]
k_num2 = kidney[['age','bp','bgr','hemo','sc']].dropna()
k_norm = (k_num2 - k_num2.mean()) / k_num2.std()
k_norm.boxplot(ax=ax3, boxprops=dict(color=GREEN),
               medianprops=dict(color=DARK, linewidth=2),
               whiskerprops=dict(color=TEAL))
ax3.set_title("Kidney Disease", fontsize=12, fontweight='bold', color=GREEN)
ax3.tick_params(axis='x', rotation=30)
ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)

plt.tight_layout()
save(fig, "11_outlier_detection_boxplots.png")

# ══════════════════════════════════════════════════════════
# 12. ALL DATASETS — Summary Statistics Dashboard
# ══════════════════════════════════════════════════════════
print("[12] Summary statistics dashboard...")
fig = plt.figure(figsize=(18, 10))
fig.suptitle("HealthGuard — Dataset Summary Statistics Dashboard",
             fontsize=15, fontweight='bold', color=DARK)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Dataset sizes
ax1 = fig.add_subplot(gs[0, 0])
names  = ["Heart\nDisease", "Diabetes", "Kidney\nDisease"]
sizes  = [len(heart), len(diabetes), len(kidney)]
clrs   = [RED, ORANGE, GREEN]
bars   = ax1.bar(names, sizes, color=clrs, edgecolor='white', linewidth=1.5, width=0.5)
for bar, v in zip(bars, sizes):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
             str(v), ha='center', fontsize=11, fontweight='bold')
ax1.set_title("Dataset Sizes", fontsize=12, fontweight='bold')
ax1.set_ylabel("Samples")

# Feature count
ax2 = fig.add_subplot(gs[0, 1])
feats = [heart.shape[1]-1, diabetes.shape[1]-1, kidney.shape[1]-1]
bars2 = ax2.bar(names, feats, color=[TEAL, BLUE, PURPLE], edgecolor='white', linewidth=1.5, width=0.5)
for bar, v in zip(bars2, feats):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
             str(v), ha='center', fontsize=11, fontweight='bold')
ax2.set_title("Number of Features", fontsize=12, fontweight='bold')
ax2.set_ylabel("Features")

# Missing values
ax3 = fig.add_subplot(gs[0, 2])
miss_pct = [
    heart.isnull().sum().sum() / heart.size * 100,
    diabetes.isnull().sum().sum() / diabetes.size * 100,
    kidney.isnull().sum().sum() / kidney.size * 100,
]
bars3 = ax3.bar(names, miss_pct, color=[RED, ORANGE, GREEN], edgecolor='white', linewidth=1.5, width=0.5)
for bar, v in zip(bars3, miss_pct):
    ax3.text(bar.get_x()+bar.get_width()/2, v+0.1,
             f"{v:.1f}%", ha='center', fontsize=11, fontweight='bold')
ax3.set_title("Overall Missing Values %", fontsize=12, fontweight='bold')
ax3.set_ylabel("Missing %")

# Positive class %
ax4 = fig.add_subplot(gs[1, 0])
pos_pct = [
    heart['condition'].mean()*100,
    diabetes['Outcome'].mean()*100,
    kidney['label'].mean()*100,
]
bars4 = ax4.bar(names, pos_pct, color=[RED, ORANGE, GREEN], edgecolor='white', linewidth=1.5, width=0.5)
for bar, v in zip(bars4, pos_pct):
    ax4.text(bar.get_x()+bar.get_width()/2, v+0.5,
             f"{v:.1f}%", ha='center', fontsize=11, fontweight='bold')
ax4.set_title("Positive Class %", fontsize=12, fontweight='bold')
ax4.set_ylabel("% Positive Cases")

# Model accuracy
ax5 = fig.add_subplot(gs[1, 1])
acc = [88.33, 74.03, 100.0]
bars5 = ax5.bar(names, acc, color=[TEAL, BLUE, PURPLE], edgecolor='white', linewidth=1.5, width=0.5)
for bar, v in zip(bars5, acc):
    ax5.text(bar.get_x()+bar.get_width()/2, v+0.3,
             f"{v}%", ha='center', fontsize=11, fontweight='bold')
ax5.set_title("ML Model Accuracy", fontsize=12, fontweight='bold')
ax5.set_ylabel("Accuracy %")
ax5.set_ylim(0, 115)

# Numeric vs categorical features
ax6 = fig.add_subplot(gs[1, 2])
num_f  = [heart.select_dtypes(include=np.number).shape[1]-1,
          diabetes.select_dtypes(include=np.number).shape[1]-1,
          kidney.select_dtypes(include=np.number).shape[1]]
cat_f  = [heart.select_dtypes(exclude=np.number).shape[1],
          diabetes.select_dtypes(exclude=np.number).shape[1],
          kidney.select_dtypes(exclude=np.number).shape[1]-1]
x = np.arange(3)
ax6.bar(x-0.2, num_f, 0.35, label="Numeric",     color=[TEAL,TEAL,TEAL],   edgecolor='white')
ax6.bar(x+0.2, cat_f, 0.35, label="Categorical",  color=[ORANGE,ORANGE,ORANGE], edgecolor='white')
ax6.set_xticks(x); ax6.set_xticklabels(names)
ax6.set_title("Numeric vs Categorical Features", fontsize=12, fontweight='bold')
ax6.set_ylabel("Count"); ax6.legend(fontsize=9)

save(fig, "12_summary_dashboard.png")

# ══════════════════════════════════════════════════════════
# 13. PAIRPLOT — Heart (key features)
# ══════════════════════════════════════════════════════════
print("[13] Heart pairplot...")
h_pair = heart[['age','chol','thalach','oldpeak','condition']].copy()
h_pair['condition'] = h_pair['condition'].map({0:'No Disease', 1:'Disease'})
g = sns.pairplot(h_pair, hue='condition', palette={'No Disease': TEAL, 'Disease': RED},
                 plot_kws={'alpha': 0.5, 's': 20}, diag_kind='kde')
g.figure.suptitle("Heart Disease — Pairplot of Key Features",
                   fontsize=13, fontweight='bold', y=1.02)
save(g.figure, "13_heart_pairplot.png")

# ══════════════════════════════════════════════════════════
# 14. DIABETES — Glucose & BMI Deep Dive
# ══════════════════════════════════════════════════════════
print("[14] Diabetes glucose & BMI deep dive...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Diabetes — Glucose & BMI Deep Dive", fontsize=14, fontweight='bold', color=DARK)

# Glucose distribution
ax1 = axes[0]
diabetes['Glucose'].hist(bins=30, color=ORANGE, edgecolor='white', ax=ax1, alpha=0.8)
ax1.axvline(diabetes[diabetes['Outcome']==1]['Glucose'].mean(), color=RED,
            linewidth=2, linestyle='--', label='Diabetic mean')
ax1.axvline(diabetes[diabetes['Outcome']==0]['Glucose'].mean(), color=BLUE,
            linewidth=2, linestyle='--', label='Non-diabetic mean')
ax1.set_title("Glucose Distribution", fontsize=11, fontweight='bold')
ax1.set_xlabel("Glucose Level"); ax1.set_ylabel("Count")
ax1.legend(fontsize=8)

# BMI categories
ax2 = axes[1]
def bmi_cat(b):
    if b < 18.5: return "Underweight"
    elif b < 25: return "Normal"
    elif b < 30: return "Overweight"
    else: return "Obese"
diabetes['bmi_cat'] = diabetes['BMI'].apply(bmi_cat)
bmi_order = ["Underweight", "Normal", "Overweight", "Obese"]
bmi_ct = diabetes.groupby(['bmi_cat','Outcome']).size().unstack(fill_value=0)
bmi_ct = bmi_ct.reindex([b for b in bmi_order if b in bmi_ct.index])
bmi_ct.plot(kind='bar', ax=ax2, color=[BLUE, ORANGE], edgecolor='white', width=0.6)
ax2.set_title("BMI Category vs Diabetes", fontsize=11, fontweight='bold')
ax2.set_xlabel("BMI Category"); ax2.set_ylabel("Count")
ax2.legend(["No Diabetes", "Diabetes"], fontsize=9)
ax2.tick_params(axis='x', rotation=30)

# Age groups
ax3 = axes[2]
diabetes['age_group'] = pd.cut(diabetes['Age'], bins=[0,30,45,60,100],
                                labels=['<30','30-45','45-60','>60'])
ag = diabetes.groupby(['age_group','Outcome']).size().unstack(fill_value=0)
ag.plot(kind='bar', ax=ax3, color=[BLUE, ORANGE], edgecolor='white', width=0.6)
ax3.set_title("Age Group vs Diabetes", fontsize=11, fontweight='bold')
ax3.set_xlabel("Age Group"); ax3.set_ylabel("Count")
ax3.legend(["No Diabetes", "Diabetes"], fontsize=9)
ax3.tick_params(axis='x', rotation=0)

plt.tight_layout()
save(fig, "14_diabetes_glucose_bmi_deepdive.png")

# ══════════════════════════════════════════════════════════
# WRITE SUMMARY
# ══════════════════════════════════════════════════════════
summary = f"""
HealthGuard — EDA Summary
=========================

HEART DISEASE ({len(heart)} samples, {heart.shape[1]-1} features)
  Positive cases : {heart['condition'].sum()} ({heart['condition'].mean()*100:.1f}%)
  Avg age        : {heart['age'].mean():.1f} years
  Avg cholesterol: {heart['chol'].mean():.1f}
  Missing values : {heart.isnull().sum().sum()}

DIABETES ({len(diabetes)} samples, {diabetes.shape[1]-1} features)
  Positive cases : {diabetes['Outcome'].sum()} ({diabetes['Outcome'].mean()*100:.1f}%)
  Avg glucose    : {diabetes['Glucose'].mean():.1f}
  Avg BMI        : {diabetes['BMI'].mean():.1f}
  Missing values : {diabetes.isnull().sum().sum()}

KIDNEY DISEASE ({len(kidney)} samples, {kidney.shape[1]-2} features)
  CKD cases      : {kidney['label'].sum()} ({kidney['label'].mean()*100:.1f}%)
  Avg age        : {kidney['age'].mean():.1f} years
  Missing values : {kidney.isnull().sum().sum()}

GRAPHS GENERATED (14 total):
  01_target_distribution.png
  02_heart_feature_distributions.png
  03_heart_correlation_heatmap.png
  04_heart_risk_factors.png
  05_diabetes_feature_distributions.png
  06_diabetes_correlation_heatmap.png
  07_diabetes_relationships.png
  08_kidney_missing_values.png
  09_kidney_feature_distributions.png
  10_kidney_categorical_features.png
  11_outlier_detection_boxplots.png
  12_summary_dashboard.png
  13_heart_pairplot.png
  14_diabetes_glucose_bmi_deepdive.png
"""
with open(os.path.join(OUT, "eda_summary.txt"), "w") as f:
    f.write(summary)

print("\n✅  EDA Complete!")
print(f"   Output folder: {OUT}")
print(f"   Total graphs : 14 PNG files")
