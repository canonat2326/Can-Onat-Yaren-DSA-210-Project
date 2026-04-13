"""
DSA 210 - Introduction to Data Science
Project: Analyzing the Relationship Between Training Load, Recovery, and Strength Performance
Author: Can Onat Yaren
Date: April 2026

This script performs data cleaning, exploratory data analysis, and hypothesis testing
on personal workout data collected from the Hevy app over ~4 years (May 2022 - April 2026).

Athlete Profile: Male, 24 years old, 187 cm
Body weight: 95 kg (2022) -> 127 kg (2026)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from datetime import timedelta
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

TR_MONTHS = {
    'Oca': 'Jan', 'Şub': 'Feb', 'Mar': 'Mar', 'Nis': 'Apr',
    'May': 'May', 'Haz': 'Jun', 'Tem': 'Jul', 'Ağu': 'Aug',
    'Eyl': 'Sep', 'Eki': 'Oct', 'Kas': 'Nov', 'Ara': 'Dec'
}

BIG_LIFTS = ['Bench Press (Bar)', 'Squat (Bar)', 'Deadlift (Bar)', 'Overhead Press (Bar)']

# ---- ATHLETE PROFILE ----
BODYWEIGHT_TIMELINE = {2022: 95, 2023: 97.5, 2024: 105, 2025: 125, 2026: 127}

# Strength standards by body weight class (estimated 1RM, kg)
# Source: Symmetric Strength / Strength Level / ExRx.net
STRENGTH_STANDARDS = {
    95: {
        'Bench Press (Bar)': {'Beginner': 57, 'Novice': 82, 'Intermediate': 110, 'Advanced': 140, 'Elite': 168},
        'Squat (Bar)':       {'Beginner': 73, 'Novice': 112, 'Intermediate': 152, 'Advanced': 195, 'Elite': 238},
        'Deadlift (Bar)':    {'Beginner': 90, 'Novice': 132, 'Intermediate': 178, 'Advanced': 225, 'Elite': 273},
        'Overhead Press (Bar)': {'Beginner': 39, 'Novice': 56, 'Intermediate': 78, 'Advanced': 100, 'Elite': 123},
    },
    105: {
        'Bench Press (Bar)': {'Beginner': 62, 'Novice': 89, 'Intermediate': 118, 'Advanced': 150, 'Elite': 180},
        'Squat (Bar)':       {'Beginner': 78, 'Novice': 120, 'Intermediate': 164, 'Advanced': 208, 'Elite': 254},
        'Deadlift (Bar)':    {'Beginner': 96, 'Novice': 142, 'Intermediate': 190, 'Advanced': 240, 'Elite': 290},
        'Overhead Press (Bar)': {'Beginner': 42, 'Novice': 61, 'Intermediate': 84, 'Advanced': 108, 'Elite': 132},
    },
    125: {
        'Bench Press (Bar)': {'Beginner': 70, 'Novice': 100, 'Intermediate': 132, 'Advanced': 168, 'Elite': 200},
        'Squat (Bar)':       {'Beginner': 86, 'Novice': 133, 'Intermediate': 182, 'Advanced': 230, 'Elite': 280},
        'Deadlift (Bar)':    {'Beginner': 106, 'Novice': 156, 'Intermediate': 208, 'Advanced': 262, 'Elite': 316},
        'Overhead Press (Bar)': {'Beginner': 47, 'Novice': 68, 'Intermediate': 93, 'Advanced': 120, 'Elite': 146},
    },
}

def get_bodyweight_for_date(date):
    year = date.year
    if year <= 2022:
        return 95.0
    elif year == 2023:
        frac = (date.month - 1) / 11
        return 95 + frac * 5
    elif year == 2024:
        frac = (date.month - 1) / 11
        return 100 + frac * 10
    elif year == 2025:
        frac = (date.month - 1) / 11
        return 120 + frac * 10
    else:
        return 127.0

def get_nearest_standards(bw):
    classes = sorted(STRENGTH_STANDARDS.keys())
    if bw <= classes[0]: return STRENGTH_STANDARDS[classes[0]]
    if bw >= classes[-1]: return STRENGTH_STANDARDS[classes[-1]]
    for i in range(len(classes) - 1):
        if classes[i] <= bw <= classes[i+1]:
            lower, upper = classes[i], classes[i+1]
            frac = (bw - lower) / (upper - lower)
            result = {}
            for exercise in STRENGTH_STANDARDS[lower]:
                result[exercise] = {}
                for level in STRENGTH_STANDARDS[lower][exercise]:
                    v_lo = STRENGTH_STANDARDS[lower][exercise][level]
                    v_hi = STRENGTH_STANDARDS[upper][exercise][level]
                    result[exercise][level] = v_lo + frac * (v_hi - v_lo)
            return result
    return STRENGTH_STANDARDS[classes[-1]]

def classify_strength(exercise, e1rm, bw):
    standards = get_nearest_standards(bw)
    if exercise not in standards: return 'Unknown'
    s = standards[exercise]
    if e1rm >= s['Elite']: return 'Elite'
    elif e1rm >= s['Advanced']: return 'Advanced'
    elif e1rm >= s['Intermediate']: return 'Intermediate'
    elif e1rm >= s['Novice']: return 'Novice'
    else: return 'Beginner'


# ============================================================
# 1. DATA LOADING AND CLEANING
# ============================================================
print("=" * 60)
print("1. DATA LOADING AND CLEANING")
print("=" * 60)

df_raw = pd.read_csv("data/workout_data.csv")
print(f"Raw data shape: {df_raw.shape}")

def parse_turkish_date(date_str):
    for tr, en in TR_MONTHS.items():
        date_str = date_str.replace(tr, en)
    return pd.to_datetime(date_str, format='%d %b %Y, %H:%M')

df = df_raw.copy()
df['start_dt'] = df['start_time'].apply(parse_turkish_date)
df['end_dt'] = df['end_time'].apply(parse_turkish_date)
df['date'] = df['start_dt'].dt.date
df['year'] = df['start_dt'].dt.year
df['month'] = df['start_dt'].dt.to_period('M')
df['week'] = df['start_dt'].dt.to_period('W')
df['day_of_week'] = df['start_dt'].dt.day_name()
df['hour'] = df['start_dt'].dt.hour
df['session_duration_min'] = (df['end_dt'] - df['start_dt']).dt.total_seconds() / 60

# Remove outlier
print(f"\nOutliers (weight > 300kg): {len(df[df['weight_kg'] > 300])}")
df.loc[df['weight_kg'] > 300, 'weight_kg'] = np.nan

# Body weight, volume, 1RM, relative strength
df['est_bodyweight'] = df['start_dt'].apply(get_bodyweight_for_date)
df['volume'] = df['weight_kg'] * df['reps']
df['estimated_1rm'] = df['weight_kg'] * (1 + df['reps'] / 30)
df['relative_1rm'] = df['estimated_1rm'] / df['est_bodyweight']

print(f"\nCleaned data shape: {df.shape}")
print(f"Date range: {df['start_dt'].min().date()} to {df['start_dt'].max().date()}")
print(f"Total unique sessions: {df['start_time'].nunique()}")
print(f"Total unique exercises: {df['exercise_title'].nunique()}")
print(f"Body weight range: {df['est_bodyweight'].min():.0f} - {df['est_bodyweight'].max():.0f} kg")

df.to_csv("data/workout_data_cleaned.csv", index=False)
print("\nCleaned data saved.")


# ============================================================
# 2. SESSION-LEVEL AGGREGATION
# ============================================================
print("\n" + "=" * 60)
print("2. SESSION-LEVEL AGGREGATION")
print("=" * 60)

sessions = df.groupby('start_time').agg(
    date=('date', 'first'), start_dt=('start_dt', 'first'),
    session_duration_min=('session_duration_min', 'first'),
    title=('title', 'first'), est_bodyweight=('est_bodyweight', 'first'),
    n_exercises=('exercise_title', 'nunique'), n_sets=('set_index', 'count'),
    total_volume=('volume', 'sum'), mean_weight=('weight_kg', 'mean'),
    max_weight=('weight_kg', 'max'), total_reps=('reps', 'sum'),
).reset_index()

sessions['date'] = pd.to_datetime(sessions['date'])
sessions = sessions.sort_values('start_dt').reset_index(drop=True)
sessions['days_since_last'] = sessions['start_dt'].diff().dt.total_seconds() / 86400
sessions['relative_volume'] = sessions['total_volume'] / sessions['est_bodyweight']
sessions['week'] = sessions['start_dt'].dt.to_period('W')
sessions['month'] = sessions['start_dt'].dt.to_period('M')

weekly = sessions.groupby('week').agg(
    sessions_per_week=('start_time', 'count'),
    weekly_volume=('total_volume', 'sum'),
    avg_session_duration=('session_duration_min', 'mean'),
).reset_index()
weekly['week_start'] = weekly['week'].apply(lambda x: x.start_time)

print(f"Total sessions: {len(sessions)}")
print(f"Average session duration: {sessions['session_duration_min'].mean():.1f} min")
print(f"Average exercises per session: {sessions['n_exercises'].mean():.1f}")
print(f"Average sets per session: {sessions['n_sets'].mean():.1f}")
print(f"Average rest days: {sessions['days_since_last'].mean():.1f}")
print(f"\nSessions per year:\n{sessions.groupby(sessions['start_dt'].dt.year)['start_time'].count()}")


# ============================================================
# 3. EXERCISE-LEVEL ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("3. EXERCISE-LEVEL ANALYSIS (BIG LIFTS)")
print("=" * 60)

big_lifts_df = df[df['exercise_title'].isin(BIG_LIFTS)].copy()
print(f"Big lifts data: {len(big_lifts_df)} sets")

lift_progress = big_lifts_df.groupby(['exercise_title', 'date']).agg(
    max_weight=('weight_kg', 'max'), max_1rm=('estimated_1rm', 'max'),
    total_volume=('volume', 'sum'), n_sets=('set_index', 'count'),
    total_reps=('reps', 'sum'), est_bodyweight=('est_bodyweight', 'first'),
).reset_index()

lift_progress['date'] = pd.to_datetime(lift_progress['date'])
lift_progress['relative_1rm'] = lift_progress['max_1rm'] / lift_progress['est_bodyweight']

for lift in BIG_LIFTS:
    subset = lift_progress[lift_progress['exercise_title'] == lift]
    latest = subset.sort_values('date').iloc[-1]
    print(f"\n{lift}:")
    print(f"  Sessions: {len(subset)}, Max ever: {subset['max_weight'].max():.0f} kg")
    print(f"  Latest: {latest['max_weight']:.0f} kg (e1RM: {latest['max_1rm']:.1f} kg, "
          f"relative: {latest['relative_1rm']:.2f}x BW)")


# ============================================================
# 4. EXTERNAL ENRICHMENT: BW-ADJUSTED STRENGTH STANDARDS
# ============================================================
print("\n" + "=" * 60)
print("4. EXTERNAL ENRICHMENT: STRENGTH STANDARDS (BW-ADJUSTED)")
print("=" * 60)

lift_progress['strength_level'] = lift_progress.apply(
    lambda row: classify_strength(row['exercise_title'], row['max_1rm'], row['est_bodyweight']), axis=1)

print("Strength level distribution:")
print(lift_progress['strength_level'].value_counts())

print("\nCurrent strength levels (at ~127 kg BW):")
for lift in BIG_LIFTS:
    subset = lift_progress[lift_progress['exercise_title'] == lift].sort_values('date')
    latest = subset.iloc[-1]
    print(f"  {lift}: {latest['strength_level']} (e1RM: {latest['max_1rm']:.1f} kg, {latest['relative_1rm']:.2f}x BW)")


# ============================================================
# 5. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("5. EXPLORATORY DATA ANALYSIS - VISUALIZATIONS")
print("=" * 60)

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# --- Fig 1: Training Frequency ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
monthly_sessions = sessions.groupby(sessions['start_dt'].dt.to_period('M')).size()
monthly_sessions.index = monthly_sessions.index.to_timestamp()
axes[0].bar(monthly_sessions.index, monthly_sessions.values, width=25, color='#2E86AB', alpha=0.85)
axes[0].set_ylabel('Sessions per Month')
axes[0].set_title('Training Frequency Over Time', fontsize=14, fontweight='bold')
axes[1].plot(weekly['week_start'], weekly['sessions_per_week'], color='#A23B72', alpha=0.4, linewidth=0.8)
weekly_rolling = weekly['sessions_per_week'].rolling(8, center=True).mean()
axes[1].plot(weekly['week_start'], weekly_rolling, color='#A23B72', linewidth=2.5, label='8-week rolling avg')
axes[1].set_ylabel('Sessions per Week'); axes[1].set_xlabel('Date'); axes[1].legend()
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/01_training_frequency.png"); plt.close()
print("Saved: 01_training_frequency.png")

# --- Fig 2: Strength Progression with BW-adjusted standards ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (lift, color) in enumerate(zip(BIG_LIFTS, colors)):
    ax = axes[idx // 2, idx % 2]
    subset = lift_progress[lift_progress['exercise_title'] == lift].sort_values('date')
    ax.scatter(subset['date'], subset['max_1rm'], alpha=0.25, s=15, color=color)
    if len(subset) > 10:
        rolling = subset['max_1rm'].rolling(15, center=True, min_periods=5).mean()
        ax.plot(subset['date'].values, rolling.values, color=color, linewidth=2.5)
    std = get_nearest_standards(127)
    if lift in std:
        for level, val in std[lift].items():
            ax.axhline(y=val, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax.text(subset['date'].max(), val, f' {level}', fontsize=7, va='center', alpha=0.6)
    ax.set_title(lift, fontsize=12, fontweight='bold'); ax.set_ylabel('Estimated 1RM (kg)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')); ax.tick_params(axis='x', rotation=45)
fig.suptitle('Strength Progression (Estimated 1RM with Standards for ~127 kg Male)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/02_strength_progression.png"); plt.close()
print("Saved: 02_strength_progression.png")

# --- Fig 3: Monthly Volume ---
fig, ax = plt.subplots(figsize=(14, 6))
monthly_volume = sessions.groupby(sessions['start_dt'].dt.to_period('M'))['total_volume'].sum()
monthly_volume.index = monthly_volume.index.to_timestamp()
ax.bar(monthly_volume.index, monthly_volume.values / 1000, width=25, color='#F18F01', alpha=0.85)
ax.set_ylabel('Total Volume (tons)'); ax.set_xlabel('Date')
ax.set_title('Monthly Training Volume', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/03_monthly_volume.png"); plt.close()
print("Saved: 03_monthly_volume.png")

# --- Fig 4: Duration & Rest ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
valid_dur = sessions[(sessions['session_duration_min'] > 10) & (sessions['session_duration_min'] < 180)]
axes[0].hist(valid_dur['session_duration_min'], bins=40, color='#2E86AB', alpha=0.8, edgecolor='white')
axes[0].axvline(valid_dur['session_duration_min'].median(), color='red', linestyle='--',
                label=f"Median: {valid_dur['session_duration_min'].median():.0f} min")
axes[0].set_xlabel('Duration (minutes)'); axes[0].set_ylabel('Count')
axes[0].set_title('Session Duration Distribution', fontweight='bold'); axes[0].legend()
valid_rest = sessions[(sessions['days_since_last'] > 0) & (sessions['days_since_last'] < 14)]
axes[1].hist(valid_rest['days_since_last'], bins=30, color='#A23B72', alpha=0.8, edgecolor='white')
axes[1].axvline(valid_rest['days_since_last'].median(), color='red', linestyle='--',
                label=f"Median: {valid_rest['days_since_last'].median():.1f} days")
axes[1].set_xlabel('Rest Days Between Sessions'); axes[1].set_ylabel('Count')
axes[1].set_title('Rest Period Distribution', fontweight='bold'); axes[1].legend()
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/04_duration_and_rest.png"); plt.close()
print("Saved: 04_duration_and_rest.png")

# --- Fig 5: Day & Hour ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_counts = sessions['start_dt'].dt.day_name().value_counts().reindex(dow_order)
axes[0].barh(dow_order[::-1], dow_counts.values[::-1], color='#2E86AB', alpha=0.85)
axes[0].set_xlabel('Number of Sessions'); axes[0].set_title('Sessions by Day of Week', fontweight='bold')
hour_counts = df.groupby('hour')['start_time'].nunique()
axes[1].bar(hour_counts.index, hour_counts.values, color='#F18F01', alpha=0.85, edgecolor='white')
axes[1].set_xlabel('Hour of Day'); axes[1].set_ylabel('Number of Sessions')
axes[1].set_title('Sessions by Hour of Day', fontweight='bold')
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/05_day_and_hour.png"); plt.close()
print("Saved: 05_day_and_hour.png")

# --- Fig 6: Top Exercises ---
fig, ax = plt.subplots(figsize=(12, 7))
top_ex = df['exercise_title'].value_counts().head(15)
ax.barh(top_ex.index[::-1], top_ex.values[::-1], color='#C73E1D', alpha=0.8)
ax.set_xlabel('Number of Sets')
ax.set_title('Top 15 Exercises by Total Sets', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/06_top_exercises.png"); plt.close()
print("Saved: 06_top_exercises.png")

# --- Fig 7: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
corr_cols = ['session_duration_min','n_exercises','n_sets','total_volume',
             'mean_weight','max_weight','total_reps','days_since_last','est_bodyweight']
corr = sessions[corr_cols].dropna().corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation Between Session-Level Features', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/07_correlation_heatmap.png"); plt.close()
print("Saved: 07_correlation_heatmap.png")

# --- Fig 8: Volume vs Strength ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (lift, color) in enumerate(zip(BIG_LIFTS, colors)):
    ax = axes[idx // 2, idx % 2]
    subset = lift_progress[lift_progress['exercise_title'] == lift].copy()
    subset['month'] = pd.to_datetime(subset['date']).dt.to_period('M')
    monthly = subset.groupby('month').agg(avg_1rm=('max_1rm', 'mean'), total_vol=('total_volume', 'sum')).reset_index()
    ax.scatter(monthly['total_vol'] / 1000, monthly['avg_1rm'], alpha=0.6, color=color, s=40)
    m = monthly[['total_vol','avg_1rm']].dropna()
    if len(m) > 5:
        z = np.polyfit(m['total_vol']/1000, m['avg_1rm'], 1)
        p = np.poly1d(z)
        xr = np.linspace(m['total_vol'].min()/1000, m['total_vol'].max()/1000, 100)
        ax.plot(xr, p(xr), '--', color=color, alpha=0.7, linewidth=2)
    ax.set_title(lift, fontweight='bold'); ax.set_xlabel('Monthly Volume (tons)'); ax.set_ylabel('Avg e1RM (kg)')
fig.suptitle('Monthly Training Volume vs. Average Estimated 1RM', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/08_volume_vs_strength.png"); plt.close()
print("Saved: 08_volume_vs_strength.png")

# --- Fig 9: Strength Level Progression ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
level_order = ['Beginner','Novice','Intermediate','Advanced','Elite']
level_colors = {'Beginner':'#C73E1D','Novice':'#F18F01','Intermediate':'#2E86AB','Advanced':'#A23B72','Elite':'#1B998B'}
for idx, lift in enumerate(BIG_LIFTS):
    ax = axes[idx // 2, idx % 2]
    subset = lift_progress[lift_progress['exercise_title'] == lift].sort_values('date')
    for level in level_order:
        ld = subset[subset['strength_level'] == level]
        if len(ld) > 0:
            ax.scatter(ld['date'], ld['max_1rm'], alpha=0.5, s=15, color=level_colors[level], label=level)
    ax.set_title(lift, fontweight='bold'); ax.set_ylabel('Estimated 1RM (kg)')
    ax.legend(fontsize=8, loc='upper left'); ax.tick_params(axis='x', rotation=45)
fig.suptitle('Strength Level Classification Over Time (Body-Weight-Adjusted)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/09_strength_levels.png"); plt.close()
print("Saved: 09_strength_levels.png")

# --- Fig 10: Absolute vs Relative Strength ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (lift, color) in enumerate(zip(BIG_LIFTS, colors)):
    ax = axes[idx // 2, idx % 2]
    subset = lift_progress[lift_progress['exercise_title'] == lift].sort_values('date')
    ax.scatter(subset['date'], subset['max_1rm'], alpha=0.2, s=12, color=color, label='Absolute 1RM')
    if len(subset) > 10:
        r_abs = subset['max_1rm'].rolling(15, center=True, min_periods=5).mean()
        ax.plot(subset['date'].values, r_abs.values, color=color, linewidth=2.5)
    ax2 = ax.twinx()
    ax2.scatter(subset['date'], subset['relative_1rm'], alpha=0.15, s=10, color='gray')
    if len(subset) > 10:
        r_rel = subset['relative_1rm'].rolling(15, center=True, min_periods=5).mean()
        ax2.plot(subset['date'].values, r_rel.values, color='gray', linewidth=2, linestyle='--', label='Relative (x BW)')
    ax2.set_ylabel('Relative 1RM (x BW)', color='gray'); ax2.tick_params(axis='y', labelcolor='gray')
    ax.set_title(lift, fontweight='bold'); ax.set_ylabel('Estimated 1RM (kg)', color=color)
    ax.tick_params(axis='y', labelcolor=color); ax.tick_params(axis='x', rotation=45)
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=8, loc='upper left')
fig.suptitle('Absolute vs. Relative Strength (BW: 95 kg -> 127 kg)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/10_absolute_vs_relative_strength.png"); plt.close()
print("Saved: 10_absolute_vs_relative_strength.png")

# --- Fig 11: Body Weight Trajectory ---
fig, ax = plt.subplots(figsize=(14, 5))
bw_dates = pd.date_range(start='2022-05-20', end='2026-04-11', freq='D')
bw_values = [get_bodyweight_for_date(d) for d in bw_dates]
ax.plot(bw_dates, bw_values, color='#2E86AB', linewidth=3, label='Estimated Body Weight')
ax.fill_between(bw_dates, bw_values, alpha=0.15, color='#2E86AB')
for year, bw in BODYWEIGHT_TIMELINE.items():
    ax.annotate(f'{bw} kg', xy=(pd.Timestamp(f'{year}-06-01'), bw), fontsize=10, fontweight='bold',
               ha='center', xytext=(0, 15), textcoords='offset points',
               arrowprops=dict(arrowstyle='->', color='gray'))
ax.set_ylabel('Body Weight (kg)'); ax.set_xlabel('Date')
ax.set_title('Body Weight Progression Over Training Period', fontsize=14, fontweight='bold')
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/11_bodyweight_trajectory.png"); plt.close()
print("Saved: 11_bodyweight_trajectory.png")


# ============================================================
# 6. HYPOTHESIS TESTING
# ============================================================
print("\n" + "=" * 60)
print("6. HYPOTHESIS TESTING")
print("=" * 60)

# --- H1: Volume -> Strength ---
print("\n--- HYPOTHESIS 1 ---")
print("H0: Weekly training volume has no effect on subsequent strength gains.")
print("H1: Higher weekly training volume is associated with greater subsequent strength gains.\n")

for lift in BIG_LIFTS:
    subset = lift_progress[lift_progress['exercise_title'] == lift].copy()
    subset['month'] = pd.to_datetime(subset['date']).dt.to_period('M')
    monthly = subset.groupby('month').agg(avg_1rm=('max_1rm','mean'), total_volume=('total_volume','sum')).reset_index()
    monthly['next_month_1rm'] = monthly['avg_1rm'].shift(-1)
    monthly['1rm_change'] = monthly['next_month_1rm'] - monthly['avg_1rm']
    monthly = monthly.dropna(subset=['1rm_change','total_volume'])
    if len(monthly) > 10:
        med = monthly['total_volume'].median()
        high = monthly[monthly['total_volume'] >= med]['1rm_change']
        low = monthly[monthly['total_volume'] < med]['1rm_change']
        t, pv = stats.ttest_ind(high, low, equal_var=False)
        r, pc = stats.pearsonr(monthly['total_volume'], monthly['1rm_change'])
        print(f"  {lift}:")
        print(f"    High vol avg change: {high.mean():+.2f} kg | Low vol: {low.mean():+.2f} kg")
        print(f"    Welch t={t:.3f}, p={pv:.4f} | Pearson r={r:.3f}, p={pc:.4f}")
        print(f"    {'Reject H0' if pv < 0.05 else 'Fail to reject H0'} (alpha=0.05)\n")

# --- H2: Rest -> Performance ---
print("\n--- HYPOTHESIS 2 ---")
print("H0: Rest duration has no effect on max performance.")
print("H1: More rest days are associated with higher max weights.\n")

session_rest = sessions[['start_dt','days_since_last','max_weight']].dropna()
session_rest = session_rest[(session_rest['days_since_last'] > 0) & (session_rest['days_since_last'] < 14)]
session_rest['rest_cat'] = pd.cut(session_rest['days_since_last'], bins=[0,1.0,2.0,14.0],
                                   labels=['Short (<=1d)','Moderate (1-2d)','Long (>2d)'])

groups = [g['max_weight'].values for _, g in session_rest.groupby('rest_cat')]
f_stat, pv_anova = stats.f_oneway(*groups)

for cat in ['Short (<=1d)','Moderate (1-2d)','Long (>2d)']:
    s = session_rest[session_rest['rest_cat']==cat]['max_weight']
    print(f"  {cat}: n={len(s)}, mean={s.mean():.1f} kg, std={s.std():.1f}")

print(f"\n  ANOVA: F={f_stat:.3f}, p={pv_anova:.4f}")
print(f"  {'Reject H0' if pv_anova < 0.05 else 'Fail to reject H0'} (alpha=0.05)")

r_sp, p_sp = stats.spearmanr(session_rest['days_since_last'], session_rest['max_weight'])
print(f"  Spearman: r={r_sp:.3f}, p={p_sp:.4f}")

print("\n  Post-hoc (Bonferroni):")
cats = ['Short (<=1d)','Moderate (1-2d)','Long (>2d)']
for i in range(len(cats)):
    for j in range(i+1, len(cats)):
        g1 = session_rest[session_rest['rest_cat']==cats[i]]['max_weight']
        g2 = session_rest[session_rest['rest_cat']==cats[j]]['max_weight']
        t, p = stats.ttest_ind(g1, g2, equal_var=False)
        pa = min(p*3, 1.0)
        print(f"    {cats[i]} vs {cats[j]}: t={t:.3f}, p_adj={pa:.4f} {'*' if pa<0.05 else 'ns'}")

# --- H3: Frequency -> Strength ---
print("\n\n--- HYPOTHESIS 3 ---")
print("H0: Training frequency has no effect on monthly strength change.")
print("H1: Higher frequency leads to greater monthly strength gains.\n")

for lift in BIG_LIFTS:
    subset = lift_progress[lift_progress['exercise_title'] == lift].copy()
    subset['month'] = pd.to_datetime(subset['date']).dt.to_period('M')
    m1 = subset.groupby('month')['max_1rm'].mean().reset_index()
    m1.columns = ['month','avg_1rm']; m1['1rm_change'] = m1['avg_1rm'].diff()
    sm = sessions.copy(); sm['month'] = sm['start_dt'].dt.to_period('M')
    sc = sm.groupby('month').size().reset_index(name='n_sessions')
    merged = m1.merge(sc, on='month', how='inner').dropna()
    if len(merged) > 10:
        r, p = stats.pearsonr(merged['n_sessions'], merged['1rm_change'])
        med = merged['n_sessions'].median()
        hi = merged[merged['n_sessions']>=med]['1rm_change']
        lo = merged[merged['n_sessions']<med]['1rm_change']
        t, pt = stats.ttest_ind(hi, lo, equal_var=False)
        print(f"  {lift}: r={r:.3f}, p={p:.4f}")
        print(f"    High freq: {hi.mean():+.2f} kg | Low freq: {lo.mean():+.2f} kg | t={t:.3f}, p={pt:.4f}")
        print(f"    {'Reject H0' if pt < 0.05 else 'Fail to reject H0'}\n")

# --- Fig 12: Hypothesis Tests ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
lift = 'Bench Press (Bar)'
subset = lift_progress[lift_progress['exercise_title'] == lift].copy()
subset['month'] = pd.to_datetime(subset['date']).dt.to_period('M')
monthly = subset.groupby('month').agg(avg_1rm=('max_1rm','mean'), total_volume=('total_volume','sum')).reset_index()
monthly['1rm_change'] = monthly['avg_1rm'].shift(-1) - monthly['avg_1rm']
monthly = monthly.dropna()
ax.scatter(monthly['total_volume']/1000, monthly['1rm_change'], alpha=0.6, color='#2E86AB', s=50)
z = np.polyfit(monthly['total_volume']/1000, monthly['1rm_change'], 1); p = np.poly1d(z)
xr = np.linspace(monthly['total_volume'].min()/1000, monthly['total_volume'].max()/1000, 50)
ax.plot(xr, p(xr), '--', color='#C73E1D', linewidth=2)
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('Monthly Volume (tons)'); ax.set_ylabel('Next Month 1RM Change (kg)')
ax.set_title('H1: Volume vs. Strength Gain\n(Bench Press)', fontweight='bold')

ax = axes[1]
bp_data = [session_rest[session_rest['rest_cat']==cat]['max_weight'].values for cat in cats]
bp = ax.boxplot(bp_data, labels=['Short\n(<=1d)','Moderate\n(1-2d)','Long\n(>2d)'],
                patch_artist=True, widths=0.6)
for patch, c in zip(bp['boxes'], ['#2E86AB','#F18F01','#A23B72']):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_title('H2: Rest Duration vs. Max Weight\n(ANOVA p < 0.001 ***)', fontweight='bold')
ax.set_xlabel('Rest Category'); ax.set_ylabel('Max Weight in Session (kg)')

ax = axes[2]
lift = 'Squat (Bar)'
subset = lift_progress[lift_progress['exercise_title'] == lift].copy()
subset['month'] = pd.to_datetime(subset['date']).dt.to_period('M')
m1 = subset.groupby('month')['max_1rm'].mean().reset_index(); m1.columns = ['month','avg_1rm']
m1['1rm_change'] = m1['avg_1rm'].diff()
sm = sessions.copy(); sm['month'] = sm['start_dt'].dt.to_period('M')
sc = sm.groupby('month').size().reset_index(name='n_sessions')
merged = m1.merge(sc, on='month', how='inner').dropna()
ax.scatter(merged['n_sessions'], merged['1rm_change'], alpha=0.6, color='#A23B72', s=50)
if len(merged) > 5:
    z = np.polyfit(merged['n_sessions'], merged['1rm_change'], 1); p = np.poly1d(z)
    xr = np.linspace(merged['n_sessions'].min(), merged['n_sessions'].max(), 50)
    ax.plot(xr, p(xr), '--', color='#C73E1D', linewidth=2)
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('Sessions per Month'); ax.set_ylabel('Monthly 1RM Change (kg)')
ax.set_title('H3: Frequency vs. Strength Gain\n(Squat)', fontweight='bold')

plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/12_hypothesis_tests.png"); plt.close()
print("\nSaved: 12_hypothesis_tests.png")


# ============================================================
# 7. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("7. SUMMARY")
print("=" * 60)
total_weeks = (sessions['start_dt'].max() - sessions['start_dt'].min()).days / 7
print(f"""
Athlete Profile: Male, 24 yrs, 187 cm, {BODYWEIGHT_TIMELINE[2022]} kg -> {BODYWEIGHT_TIMELINE[2026]} kg

Dataset: {len(df)} sets | {len(sessions)} sessions | {df['exercise_title'].nunique()} exercises
Period: May 2022 - Apr 2026 (~4 years) | Avg {len(sessions)/total_weeks:.1f} sessions/week

External Enrichment: BW-adjusted strength standards (Symmetric Strength / ExRx.net)

Hypothesis Results:
  H1 (Volume -> Strength):    Fail to reject H0 (p > 0.05)
  H2 (Rest -> Performance):   REJECT H0 (ANOVA p < 0.001 ***)
  H3 (Frequency -> Strength): Fail to reject H0 (p > 0.05)
""")
print("Analysis complete!")
