# DSA 210 - Analyzing the Relationship Between Training Load, Recovery, and Strength Performance

## Project Overview

This project analyzes **~4 years of personal strength training data** (May 2022 – April 2026) to investigate how training load, recovery patterns, and training frequency relate to strength performance over time. The data was collected using the [Hevy](https://www.hevyapp.com/) workout tracking app and contains **19,627 set-level records** across **975 workout sessions** and **145 exercises**.

The analysis focuses on four major compound lifts: **Bench Press, Squat, Deadlift, and Overhead Press**.

**Athlete profile:** Male, 24 years old, 187 cm. Body weight increased from ~95 kg (2022) to ~127 kg (2026), which is accounted for through body-weight-adjusted strength standards and relative strength analysis.

## Motivation

Strength training is a long-term endeavor where progress depends on the interplay between training stimulus (volume, intensity) and recovery. By applying data science methods to my own multi-year training logs, I aim to uncover patterns and test common training beliefs with statistical rigor.

## Hypotheses

1. **Training Volume → Strength Gains:** Higher monthly training volume is associated with greater estimated 1RM improvements in the following month.
2. **Rest Duration → Session Performance:** Longer rest periods between sessions are associated with higher maximum weights lifted.
3. **Training Frequency → Strength Progression:** Higher monthly training frequency leads to greater monthly strength gains.

## Dataset

| Property | Value |
|---|---|
| Source | Hevy app (personal export) |
| Period | May 2022 – April 2026 |
| Total records (sets) | 19,627 |
| Workout sessions | 975 |
| Unique exercises | 145 |
| Avg sessions/week | 4.8 |
| Avg session duration | ~77 minutes |

**Key variables:** date, exercise name, set index, weight (kg), reps, session duration.

**Derived features:** training volume (weight × reps), estimated 1RM (Epley formula), estimated body weight (interpolated from yearly records: 95→100→110→127 kg), relative strength (1RM / body weight), rest days between sessions, weekly/monthly aggregations.

**External enrichment:** Body-weight-adjusted strength standards from Symmetric Strength / ExRx.net, interpolated for the lifter's actual weight at each period (95 kg, 105 kg, 125 kg weight classes). Performance classified as Beginner → Elite.

## Project Structure

```
├── README.md                          # This file
├── data/
│   ├── workout_data.csv               # Raw data exported from Hevy
│   └── workout_data_cleaned.csv       # Cleaned and feature-engineered data
├── eda_and_hypothesis_testing.py      # Main analysis script (EDA + hypothesis tests)
├── figures/                           # Generated visualizations
│   ├── 01_training_frequency.png
│   ├── 02_strength_progression.png
│   ├── 03_monthly_volume.png
│   ├── 04_duration_and_rest.png
│   ├── 05_day_and_hour.png
│   ├── 06_top_exercises.png
│   ├── 07_correlation_heatmap.png
│   ├── 08_volume_vs_strength.png
│   ├── 09_strength_levels.png
│   ├── 10_absolute_vs_relative_strength.png
│   ├── 11_bodyweight_trajectory.png
│   └── 12_hypothesis_tests.png
├── revised_proposal.pdf               # Revised project proposal
└── requirements.txt                   # Python dependencies
```

## How to Reproduce

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis:**
   ```bash
   python eda_and_hypothesis_testing.py
   ```
   This will generate all figures in the `figures/` directory and print statistical results to the console.

## Methods

### Data Cleaning
- Parsed Turkish-locale date strings to proper datetime objects
- Removed data entry outliers (e.g., an erroneous 860 kg bench press entry)
- Handled missing weight values for bodyweight exercises (pull-ups, dips, etc.)

### Exploratory Data Analysis
- Training frequency trends (monthly and weekly)
- Strength progression over time using estimated 1RM (Epley formula)
- Session duration and rest period distributions
- Training patterns by day of week and time of day
- Correlation analysis across session-level features
- Volume vs. strength relationship

### Hypothesis Testing
- **Welch's t-test** for comparing group means (high vs. low volume/frequency)
- **One-way ANOVA** for comparing performance across rest duration categories
- **Pearson correlation** for linear relationships
- **Spearman rank correlation** for monotonic relationships

### External Enrichment
Performance classified against publicly available strength standards for male lifters, **interpolated for the lifter's actual body weight at each time period** (95 kg → 105 kg → 125 kg weight classes) from Symmetric Strength / ExRx.net:
- Beginner, Novice, Intermediate, Advanced, Elite
- Relative strength (1RM / body weight) tracked to separate genuine strength gains from body mass increases

## Key Findings (EDA & Hypothesis Tests)

- **H1 (Volume → Gains):** No statistically significant relationship between monthly training volume and next-month 1RM changes across all four lifts (p > 0.05).
- **H2 (Rest → Performance):** Statistically significant effect of rest duration on session max weight (ANOVA F=13.9, p < 0.001). Longer rest periods associated with higher max weights.
- **H3 (Frequency → Gains):** No statistically significant relationship between monthly session count and monthly strength change (p > 0.05).

## Tools Used

- **Python**: pandas, numpy, scipy, matplotlib, seaborn
- **Data Source**: Hevy workout tracking app
- **Version Control**: Git / GitHub

## Timeline

| Deadline | Deliverable | Status |
|---|---|---|
| 17 March | GitHub repo created | ✅ |
| 31 March | Project proposal | ✅ |
| 14 April | Data collection, EDA, hypothesis tests | ✅ |
| 5 May | Machine learning methods | Upcoming |
| 18 May | Final report and code | Upcoming |

## Acknowledgments

- DSA 210 course at Sabancı University (Spring 2025-2026)
- Strength standards sourced from Symmetric Strength and ExRx.net
- Data collected using the Hevy app

## LLM Usage Disclosure

In accordance with the course policy on academic integrity, I used Claude (Anthropic) as an AI assistant during this project. It was used for:
- Getting advice on appropriate statistical tests for my hypotheses
- Debugging and structuring Python code for data processing and visualization
- Guidance on project organization and README structure

All analysis decisions, interpretations, and written content reflect my own understanding of the data and methods.
