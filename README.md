# Laptop Reviews - Aspect-Based Sentiment Analysis (ABSA)

A data science portfolio project analyzing Flipkart laptop reviews to understand customer sentiment at both an overall and (eventually) aspect level - covering data cleaning, exploratory analysis, statistical hypothesis testing, and a baseline machine learning sentiment model, with aspect-level sentiment extraction planned as the next phase.

This project is part of a mentor-led Data Science Special Interest Group (SIG), following a structured weekly curriculum.

## Project Goal

Most laptop reviews mix multiple opinions in a single sentence, for example, praising battery life while criticizing the display. A single overall star rating hides this nuance. This project works toward Aspect-Based Sentiment Analysis (ABSA): identifying specific product aspects mentioned in a review (battery, display, build quality, performance, etc.) and determining the sentiment expressed toward each one individually, rather than relying on a single overall rating.

## Dataset

- **Source:** Flipkart laptop reviews
- **Size:** 16,991 reviews after cleaning, spanning 20 brands
- **Key columns:** `product_name`, `rating`, `review`, `no_ratings`, `no_reviews`
- **Derived columns:** `brand`, `popularity_tier`, `review_length`

## Project Structure

```
laptop-absa-project/
├── data/
│   ├── raw/
│   └── processed/
│       ├── processed_data.csv
│       └── processed_data_enriched.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_statistical_analysis.ipynb
│   └── 05_model_baseline.ipynb
├── reports/
│   ├── data_cleaning_report.pdf
│   ├── eda_summary.pdf
│   ├── statistical_analysis_report.pdf
│   └── model_baseline_report.pdf
├── requirements.txt
└── README.md
```

## Progress by Week

| Week | Focus | Status | Notebook | Report |
|---|---|---|---|---|
| 03 | Data Understanding | Complete | `01_data_understanding.ipynb` | - |
| 04 | Data Cleaning | Complete | `02_data_cleaning.ipynb` | `data_cleaning_report.pdf` |
| 05 | Exploratory Data Analysis | Complete | `03_exploratory_data_analysis.ipynb` | `eda_summary.pdf` |
| 06 | Statistical Hypothesis Testing | Complete | `04_statistical_analysis.ipynb` | `statistical_analysis_report.pdf` |
| 07 | Baseline ML Sentiment Model | Complete | `05_model_baseline.ipynb` | `model_baseline_report.pdf` |
| Next | Aspect Extraction | Planned | - | - |

## Key Findings So Far

**Statistical Analysis (Week 06)**
- Average rating differs significantly across brands (Kruskal-Wallis, p < 0.001), holding even after excluding brands with fewer than 30 reviews
- Rating is significantly associated with product popularity tier (Chi-square, p < 0.001) - high-popularity products show more consistently positive ratings
- Review length has a statistically significant but practically negligible relationship with rating (Spearman's rho = -0.112) - a clear example of statistical significance not implying business significance

**Baseline Sentiment Model (Week 07)**
- Reviews were labeled Positive (4-5 stars) or Negative (1-3 stars) and classified using review text alone (TF-IDF features + Logistic Regression)
- A majority-class baseline reached 82% accuracy but 0% recall on negative reviews, showing accuracy alone is misleading on this imbalanced dataset (81.6% Positive / 18.4% Negative)
- The trained model improved accuracy to 87% and, more importantly, achieved 80% recall on negative reviews, confirming review text carries a genuine, learnable sentiment signal

## Methodology Notes

- **Git workflow:** one branch per week/feature, PRs merged into `main`, no direct commits to `main`
- **Statistical testing:** non-parametric tests (Kruskal-Wallis, Spearman) were preferred over their parametric equivalents due to the skewed rating distribution
- **Data leakage prevention:** TF-IDF vectorization is fit only on training data, never on the full dataset, to avoid leaking test information into the model
- **Class imbalance:** addressed using `class_weight="balanced"` in the baseline model rather than resampling, given this is an initial baseline

## Tech Stack

- Python (pandas, numpy, scipy, scikit-learn, matplotlib, seaborn)
- Jupyter Notebooks
- Git / GitHub

## Next Steps

- Discuss the 2-class (Positive/Negative) vs 3-class (with Neutral) sentiment labeling decision with the SIG mentor
- Begin Aspect Extraction: identifying specific product aspects mentioned within reviews
- Extend sentiment classification to the aspect level
- Add model performance metrics and a live demo link once later phases are complete

## Author

Sameera Ekanayaka
GitHub: [sameera-ekanayaka](https://github.com/sameera-ekanayaka)
