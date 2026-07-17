# 🛒 Customer Segmentation & A/B Testing for Targeted Marketing

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0.3-blue.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Solution Approach](#-solution-approach)
- [Key Results](#-key-results)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Visualizations](#-visualizations)
- [Results](#-results)
- [Future Work](#-future-work)
- [Author](#-author)
- [License](#-license)

---

## 📊 Project Overview

This project demonstrates an end-to-end data science solution for customer analytics in e-commerce. By combining **unsupervised learning (K-Means clustering)** with **hypothesis testing (A/B testing)**, this project develops a personalized marketing strategy that significantly improves conversion rates.

The pipeline generates synthetic customer data, performs segmentation, runs A/B tests with statistical rigor, and produces actionable business recommendations through an interactive dashboard.

**Why this matters:** One-size-fits-all marketing is inefficient. Different customer segments respond differently to promotions. This project shows how data science can help businesses:
- Identify who their customers really are
- Test what works for each segment
- Scale what works and stop what doesn't

---

## 💼 Business Problem

A retail company is running uniform marketing campaigns across all customers. This leads to:
- Wasted marketing spend on customers who would buy anyway
- Missed opportunities to re-engage dormant customers
- Inefficient discounting that erodes margins

**The goal:** Develop a personalized marketing strategy that maximizes ROI by targeting the right customers with the right message.

---

## 🧠 Solution Approach

### 1. Data Generation
Created synthetic customer data with realistic e-commerce behavior patterns:
- Pareto distribution for spend (80/20 rule)
- Mix of recency patterns (active, dormant, at-risk)
- Intentional missing values to mimic real-world data

### 2. Customer Segmentation (K-Means Clustering)
Grouped customers into 4 distinct segments based on:
- Total spend
- Days since last purchase (recency)
- Login frequency
- Average order value

**Segments identified:**
- **High Value** — Big spenders, very active (9.4%)
- **Loyal** — Regular, engaged customers (18.8%)
- **At-Risk** — Previously high-spending, now dormant (15.6%)
- **Occasional** — Low engagement, infrequent buyers (56.2%)

### 3. A/B Testing
Simulated a test where half the customers received a promotional offer (Treatment) and half received no offer (Control). Analyzed the lift within each segment using:
- Two-sample t-tests
- **Bonferroni correction** for multiple testing (p < 0.05 threshold)
- Lift percentage calculation

### 4. Business Recommendations
Provided clear, actionable recommendations for each segment based on statistical evidence.

---

## 📈 Key Results

### A/B Test Results by Segment

| Segment | Control Conv. | Treatment Conv. | Lift | p-value (adj) | Significant? | Recommendation |
|---------|--------------|----------------|------|---------------|--------------|----------------|
| **At-Risk** | 8.0% | 12.8% | **+60.0%** | 0.046 | ✅ Yes | **Deploy** |
| **Loyal** | 22.0% | 30.1% | **+36.8%** | 0.038 | ✅ Yes | **Deploy** |
| **High Value** | 30.0% | 34.6% | +15.3% | 1.000 | ❌ No | Re-evaluate |
| **Occasional** | 12.0% | 12.2% | +1.9% | 1.000 | ❌ No | Re-evaluate |

### Business Impact
- **At-Risk customers** show the highest response to re-engagement offers (+60% lift)
- **Loyal customers** respond well to rewards programs (+36.8% lift)
- **High Value customers** don't need discounts — they buy regardless
- **Occasional customers** need a different approach (longer test or alternative strategy)

### Recommended Strategy
1. **Immediate action** — Launch re-engagement campaigns for At-Risk customers
2. **Short-term** — Implement a loyalty rewards program for Loyal customers
3. **Avoid** — Stop discounting High Value customers (wastes revenue)
4. **Investigate** — Test different approaches for Occasional customers

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (K-Means, PCA, StandardScaler)
- **Statistical Testing**: SciPy (t-tests, Bonferroni correction)
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebooks

---

## 📁 Project Structure

```
08-customer-analytics/segmentation-ab-testing/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
│
├── data/
│   ├── raw/
│   │   └── customers_raw.csv          # Generated customer data
│   ├── processed/
│   │   ├── customers_cleaned.csv      # Cleaned data
│   │   └── customers_segmented.csv    # Data with segment labels
│   └── outputs/
│       ├── figures/
│       │   ├── distributions.png      # Data distributions
│       │   ├── segment_radar.png      # Segment comparison
│       │   ├── segments_2d.png        # PCA visualization
│       │   ├── segment_heatmap.png    # Segment characteristics
│       │   ├── ab_results.png         # A/B test results
│       │   └── dashboard.png          # Business dashboard
│       └── results/
│           └── ab_test_results.csv    # Detailed test results
│
├── src/                                # Source code
│   ├── __init__.py
│   ├── helpers.py                     # Utility functions
│   ├── generate_data.py               # Data generation
│   ├── clean_data.py                  # Data cleaning
│   ├── clustering.py                  # K-Means segmentation
│   ├── ab_test.py                     # A/B testing
│   ├── visualize.py                   # All visualizations
│   └── pipeline.py                    # Main pipeline runner
│
│
└── tests/
    └── (empty - placeholder for future tests)
```

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/MuhammadZahidRWTH/DataScience-Projects.git
cd DataScience-Projects/08-customer-analytics/segmentation-ab-testing
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
cd src
python pipeline.py
```

---

## 💻 Usage

### Run the Full Pipeline
```bash
cd src
python pipeline.py
```

This will:
1. Generate 5,000 synthetic customers
2. Clean the data (handle missing values, outliers)
3. Segment customers using K-Means clustering
4. Run A/B test analysis
5. Generate all visualizations
6. Save outputs to `data/outputs/`

### Explore Results
After running the pipeline, check:
- **Visualizations**: `data/outputs/figures/`
- **Results**: `data/outputs/results/`
- **Processed Data**: `data/processed/`

### Modify Parameters
In `pipeline.py`, you can adjust:
```python
# Number of customers
run_pipeline(n_customers=10000)

# Number of segments
run_pipeline(k_clusters=4)
```
---

## 🎨 Visualizations

1. **Data Distributions** (`distributions.png`) — Shows the distribution of key metrics: total spend (heavy right skew, typical in e-commerce), days inactive (mix of recent and dormant customers), and login frequency (zero-inflated distribution).
2. **Segment Radar Chart** (`segment_radar.png`) — Compares segment profiles across total spend, days inactive, login frequency, and average order value.
3. **Segment Visualization / PCA** (`segments_2d.png`) — 2D projection of customer segments using PCA, showing clear separation between groups.
4. **Segment Characteristics Heatmap** (`segment_heatmap.png`) — Average values per segment across all features, making it easy to spot differences.
5. **A/B Test Results** (`ab_results.png`) — Left: conversion rates for Control vs. Treatment by segment. Right: lift percentage with statistical significance indicators.
6. **Business Dashboard** (`dashboard.png`) — Executive summary showing segment distribution, key metrics per segment, business recommendations, and actionable insights.

---

## 📊 Results

### Customer Segments Found

| Segment | Count | % | Avg Spend | Days Inactive | Avg Logins | Description |
|---------|-------|---|-----------|---------------|------------|--------------|
| Occasional | 2,808 | 56.2% | $121 | 18.7 | 3.1 | Low engagement, infrequent buyers |
| Loyal | 942 | 18.8% | $804 | 8.8 | 10.3 | Regular, engaged customers |
| At-Risk | 778 | 15.6% | $915 | 104.4 | 2.2 | Dormant, previously high spend |
| High Value | 472 | 9.4% | $2,668 | 2.8 | 20.3 | Big spenders, very active |

### Statistical Findings
After applying Bonferroni correction (p < 0.05 threshold):
- **At-Risk**: Significant positive lift (+60.0%, p = 0.046)
- **Loyal**: Significant positive lift (+36.8%, p = 0.038)
- **High Value**: Not significant (p = 1.000)
- **Occasional**: Not significant (p = 1.000)

### Business Recommendations

| Segment | Recommendation | Rationale |
|---------|----------------|-----------|
| At-Risk | 🚀 Deploy re-engagement offers | Highest lift, statistically significant |
| Loyal | 🎯 Deploy loyalty rewards | Strong positive response |
| High Value | ⏸️ Re-evaluate strategy | Not significant, try alternative |
| Occasional | ⏸️ Re-evaluate strategy | No effect, test different approach |

---

## 🔮 Future Work

- **Real data integration** — Replace synthetic data with actual customer data
- **ROI calculation** — Add revenue impact analysis
- **Time-series analysis** — Track segment changes over time
- **Alternative models** — Try hierarchical clustering or DBSCAN
- **Multi-armed bandit** — Implement adaptive testing for faster optimization
- **API deployment** — Create an API for real-time segment assignment
- **MLflow integration** — Track experiments and models
- **Dashboard app** — Build an interactive Streamlit dashboard

---

## 👤 Author

**Muhammad Zahid**
- GitHub: [@MuhammadZahidRWTH](https://github.com/MuhammadZahidRWTH)
---

## 📝 License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Built as part of a data science portfolio
- Inspired by real-world e-commerce analytics challenges
- Thanks to the open-source community for the amazing tools

---

## ⭐️ Show Your Support

If you found this project helpful, please give it a ⭐️ on GitHub!

## 📞 Contact

Questions or suggestions? Feel free to reach out — open an issue on GitHub, connect on LinkedIn, or send an email.

---

*Built with ❤️ for data-driven decision making*