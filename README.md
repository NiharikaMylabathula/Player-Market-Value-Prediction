# Predicting Football Player Market Values (EPL & La Liga, 2023–24)

Predict a player's Transfermarkt market value from his on-pitch statistics, then use the model to recommend young, affordable signings within a club's transfer budget.

The project covers the full data science pipeline: web scraping → cleaning and merging → exploratory analysis → dimensionality reduction → six regression approaches → a budget-constrained player recommender.

---

## Highlights

- **Dataset:** 620 outfield players from the English Premier League and La Liga (2023–24), with ~115 performance features each plus market value in EUR.
- **Best model:** XGBoost, with a test R² of about 0.53 (RMSE ≈ 0.78 on standardized market value).
- **What drives value:** shots on target, goals and expected goals (xG) correlate most strongly with market value. Age is the strongest negative correlate.
- **Recommender:** a position-aware K-Means + budget optimizer that picks the best-value squad additions within a given budget (demonstrated on synthetic youth-player data).

---

## Data

| Source | What we took | Tool |
|---|---|---|
| [FBref](https://fbref.com) | Per-player 2023–24 stats for EPL and La Liga: shooting, passing, goal & shot creation, defense, possession, miscellaneous | Selenium + `pandas.read_html` |
| [Transfermarkt](https://www.transfermarkt.us) | Market values for players in the top-5 leagues | `requests` + BeautifulSoup |

Market values were scraped for all top-5 leagues (EPL, La Liga, Bundesliga, Serie A, Ligue 1), not just the two we model. Some players changed league during the season, and this raised the match rate when joining stats to values.

### Cleaning steps (`Data_preprocessing.ipynb`)

1. Dropped the FBref divider rows that repeat inside each table.
2. Converted string columns to numeric and turned percentages into fractions.
3. Imputed `NaN` with 0. The `NaN`s were all undefined ratios, such as a pass-completion rate for a player with zero passes. This was checked column by column.
4. Players who moved clubs mid-season appear twice. We kept the row with the most minutes (`90s`).
5. Removed goalkeepers, because their stats and valuations behave differently from outfield players'.
6. Cleaned the `Nation` codes and kept only a player's primary position (`FW`, `MF` or `DF`).
7. Merged the six stat tables on `Player, Nation, Pos, Squad, Age, 90s`, added a `League` column, and combined EPL and La Liga.
8. Resolved name clashes before joining to market values (for example the two Juan Cruz players, Danilo and Vitinha).
9. Parsed market values (`€12m`, `€800k`) into numbers.

The result is `final_df_new.csv` with 620 rows and 119 columns.

---

## Exploratory analysis (`Final_EDA.ipynb`)

- **League gap:** the 336 EPL players are worth about €9.33B in total (average ≈ €27.8M). The 284 La Liga players are worth about €3.77B (average ≈ €13.3M).
- **Position:** forwards are valued highest in the EPL, while midfielders carry the most total value in La Liga. Defenders are the cheapest in both leagues.
- **Age:** value peaks for players in their early-to-mid 20s and declines steadily afterwards.
- **Correlation with market value:** shots on target (0.50), goals (0.49) and xG (0.46) are the top positive correlates. Age (−0.31), La Liga membership (−0.29) and being a defender (−0.18) are among the top negative ones.
- **Outliers:** the target is heavily right-skewed. An IQR rule flags 30 outliers (620 → 590 players), and many models were evaluated both with and without them.

---

## Modeling

All models predict market value after standardizing both features and target. Categorical variables (position, league, and a nationality group of ESP / ENG / Other) were one-hot encoded. Hyperparameters were tuned with 5-fold cross-validation.

| Notebook | Approach |
|---|---|
| `PCA.ipynb` | PCA exploration, scree and cumulative-variance plots, interpretation of the top components, and Lasso/Ridge baselines on the raw and PCA-transformed data |
| `Regression_and_PCR.ipynb` | Ridge and Lasso on raw features, plus principal component regression (PCR) with 25 components |
| `PCR__Regression_and_Player_Recommender_Algorithm.ipynb` | Lasso/PCR variant, model saving, and the player recommender |
| `SVM_v2.ipynb` | Support vector regression (RBF kernel) |
| `Decision_Trees__Random_Forest.ipynb` | Decision tree and random forest regressors, with and without outliers |
| `XGB.ipynb` | XGBoost with grid search over four feature/outlier variants, plus SHAP imports for interpretation |

### Results

Test-set R² on standardized market value. Splits and random seeds differ between notebooks, so treat the comparisons as indicative.

| Model | Variant | Test R² | Train R² |
|---|---|---|---|
| **XGBoost** | Hand-picked feature subset | **0.535** | 0.928 |
| XGBoost | All features | 0.525 | 0.799 |
| XGBoost | All features, outliers removed | 0.449 | 0.959 |
| XGBoost | Selected features, outliers removed | 0.461 | 0.816 |
| Random forest | Outliers removed | 0.469 | 0.563 |
| Random forest | With outliers | 0.383 | 0.554 |
| Lasso | Raw features (tuned α) | 0.471 | n/a |
| Support vector regression | RBF, C=1 | 0.416 | 0.697 |
| Decision tree | With outliers | 0.424 | 0.428 |
| Decision tree | Outliers removed | 0.144 | 0.437 |

### Findings

- **Boosting wins, but the ceiling is low (about 0.5).** Market value depends on things that box-score stats don't capture, such as brand, contract length, club, injuries and potential.
- **Overfitting.** XGBoost, SVR and the unrestricted-feature models show large train/test gaps, which more data or stronger regularization would help.
- **Lasso picks a sensible, sparse feature set.** It keeps age (negative), goals, xG, progressive carrying distance, pass completion and EPL membership, and shrinks the rest.
- **PCA helps interpretation more than prediction.** The first components separate ball-progressing midfielders from defenders, then shooters from creators. Regression on the PCA scores did not beat regression on the raw features.

---

## Player recommender (`PCR__Regression_and_Player_Recommender_Algorithm.ipynb`)

Given a squad budget, it picks players for each position.

1. **Predict value:** the saved Lasso model estimates each candidate's market value.
2. **Cluster by role:** per position (FW / MF / DF), K-Means on position-specific features. The number of clusters (2–3) is chosen by silhouette score, with a minimum cluster size.
3. **Score players:** a weighted value score is computed from the features that matter for that position.
4. **Optimize:** the best-scoring combination of players that fits the position budget is chosen by exhaustive search, falling back to a greedy value-per-cost heuristic if nothing fits.
5. **Visualize:** clusters are plotted in 3D PCA space with the selected players highlighted.

**Demo:** the function was tested on 200 synthetic players aged 15–20, built to mimic the FBref feature distributions, with a €350M budget. It selected 4 forwards, 3 midfielders and 3 defenders for roughly €166M in total. This is a proof of concept on synthetic data, not a transfer recommendation.

---

## Repository structure

```
.
├── README.md
├── data/
│   └── final_df_new.csv                    # cleaned dataset (not included, see below)
├── notebooks/
│   ├── 01_Web_scraping_stats_values.ipynb
│   ├── 02_Data_preprocessing.ipynb
│   ├── 03_Final_EDA.ipynb
│   ├── 04_PCA.ipynb
│   ├── 05_Regression_and_PCR.ipynb
│   ├── 06_SVM_v2.ipynb
│   ├── 07_Decision_Trees__Random_Forest.ipynb
│   ├── 08_XGB.ipynb
│   └── 09_PCR__Regression_and_Player_Recommender_Algorithm.ipynb
└── requirements.txt
```

## Getting started

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
jupyter lab
```

**Requirements:** Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `scipy`, `joblib`, `requests`, `beautifulsoup4`, `selenium`, `webdriver-manager`, plus Chrome for scraping.

**Running order:**

1. `Web_scraping_stats_values` produces the raw CSVs. Skip it if you already have the data.
2. `Data_preprocessing` produces `final_df_new.csv`.
3. All other notebooks read `final_df_new.csv`, so run them in any order.

> **Note:** several notebooks contain absolute paths from the authors' machines (for example `/Users/.../final_df_new.csv`). Update them to point to your local `data/` folder before running.

Please be considerate when scraping. Respect each site's terms of service and robots rules, and add delays between requests.

---

## Limitations and future work

- Only two leagues and one season, with no multi-season trends.
- Market value is partly subjective, and goalkeepers are excluded.
- Missing context: injuries, contract length, transfer fees, social-media reach and club tier.
- The recommender is validated only on synthetic data, so real scouting data and a proper backtest are the next steps.
- Replace the hand-picked feature subset with a principled selection, and try stacking or a log-transformed target to handle the skew.
- Use a fixed random seed and a shared train/test split across notebooks for a fair comparison.


## Acknowledgments

Data from [FBref](https://fbref.com) (Sports Reference) and [Transfermarkt](https://www.transfermarkt.us).
