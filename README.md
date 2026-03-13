# 💼 San Francisco Employee Compensation Predictor

> *Can we predict what a city pays its people just from their job title and department? Turns out, yes. Pretty accurately.*

This project is a full end-to-end machine learning solution built to predict employee compensation across San Francisco's public workforce. It started as a curiosity and turned into a proper data product — complete with exploratory analysis, feature engineering, multiple trained models, and an interactive web app where you can get predictions in real time.

## 🧠 Why I Built This

Compensation data is one of those datasets that *sounds* dry until you start pulling on the threads. What does the city actually pay its people? Which departments are generous? What's the relationship between overtime reliance and total comp? Is benefits-to-salary ratio a meaningful signal?

I wanted to answer those questions — and build something that goes beyond a notebook. The Streamlit app is the proof of concept: a recruiter, a policy analyst, or a curious SF resident could open it and get real, interpretable predictions without touching a line of code.

## 📁 Project Structure

```
sf_employee_compensation/
├── compensation.ipynb                  # The full ML pipeline — EDA to deployment
├── streamlit_app.py                    # Interactive prediction web app
├── cleaned_Employee_Compensation.csv   # Source data
├── requirements.txt                    # All dependencies
│
├── best_model.pkl                      # Trained Random Forest (generated after running notebook)
├── scaler.pkl                          # Feature scaler
├── label_encoders.pkl                  # Categorical encoders
├── feature_names.pkl                   # Feature list
├── model_summary_report.txt            # Performance summary
└── README.md                           # You're reading it
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### 1. Clone / navigate to the project
```bash
cd "/Users/marykinya/Desktop/04_Study/Data Science Practice/sf_employee_compensation"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (run the notebook first)
```bash
jupyter notebook compensation.ipynb
```
Run all cells. This generates the model artifacts the app depends on:
- `best_model.pkl`
- `scaler.pkl`
- `label_encoders.pkl`
- `feature_names.pkl`
- `model_summary_report.txt`

### 4. Launch the web app
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`

## 📊 The Notebook — What's Inside

### Section 1 · Data Loading & Exploration
First things first — understanding what we're working with. This section covers:
- Statistical summaries across all compensation fields
- Missing value audit and handling strategy
- Distribution analysis to flag skews and outliers early

### Section 2 · Exploratory Data Analysis
This is where the storytelling happens. Key analyses include:
- **Top-paying job titles** — which roles command the highest total comp
- **Department-level benchmarking** — average compensation across city departments
- **Organization group comparisons** — seeing how comp varies across the org hierarchy
- **Compensation component breakdown** — salaries vs. overtime vs. benefits visualized

### Section 3 · Feature Engineering
Raw data rarely tells the full story. I engineered four additional features that turned out to be among the most predictive:

| Feature | Formula |
|--------|---------|
| `salary_component` | Salaries + Overtime + Other Salaries |
| `benefits_component` | Retirement + Health & Dental + Other Benefits |
| `benefits_to_salary_ratio` | Benefits Component ÷ Salary Component |
| `overtime_to_salary_ratio` | Overtime ÷ Base Salaries |

These ratios capture *how* comp is structured — not just how much — which added meaningful signal.

### Section 4 · Model Training & Comparison
I trained and compared four models to find the best fit:

| Model | Type | Performance |
|-------|------|-------------|
| **Random Forest** | Ensemble — 200 trees | ⭐⭐⭐⭐⭐ Best overall |
| **Gradient Boosting** | Sequential boosting | ⭐⭐⭐⭐ Very strong |
| **Voting Ensemble** | RF + GB combined | ⭐⭐⭐⭐ Very strong |
| **Ridge Regression** | Regularized linear | ⭐⭐⭐ Good baseline |

Random Forest won — both on accuracy and interpretability.

### Section 5 · Evaluation
Every model was evaluated across multiple dimensions:
- **R² Score** — how much variance is explained
- **RMSE** — dollar-level prediction error
- **MAE** — mean absolute error
- **MAPE** — percentage error (more intuitive for stakeholders)
- **5-Fold Cross-Validation** — to catch overfitting

### Section 6 · Analysis & Interpretation
- Actual vs. Predicted scatter plots
- Residuals distribution
- Feature importance rankings
- Error percentage breakdowns per job category

## 🌐 The Streamlit App

The app has four tabs:

### 🔮 Predict
Input any job details and compensation components to get an instant prediction. Also shows:
- How the prediction compares to historical averages for that job
- Quick stats for the selected role

### 📊 Data Analysis
Four lenses on the data:
1. **Jobs** — Top-paying positions with headcounts
2. **Departments** — Average comp by department
3. **Organization** — Comp across org groups
4. **Compensation** — Distribution and component breakdown

### 📈 Model Performance
Full transparency on how the model was built and how well it performs — metrics, architecture, cross-validation results, and feature importance.

### ℹ️ About
Background on the project, data source, and tech stack.

## 📈 Model Performance

The final Random Forest model delivers:

| Metric | Result |
|--------|--------|
| R² Score | > 0.90 |
| RMSE | ~$5,000 – $8,000 |
| MAPE | ~3 – 5% |
| Cross-Validation | Consistent across all folds |

Explaining 90%+ of compensation variance from job attributes alone is a strong result for this kind of public sector data.

## 🔧 Customisation

### Tune the Random Forest
Edit Section 4 of the notebook:
```python
rf_model = RandomForestRegressor(
    n_estimators=200,       # More trees = more stable, slower
    max_depth=20,           # Controls overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

### Add new features
Edit the feature engineering block in Section 3. Any new column added to the training data needs to be reflected in `feature_names.pkl` and the Streamlit input form.


## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn |
| Visualisation | Plotly |
| Web app | Streamlit |
| Model persistence | joblib |


## ⚠️ A Few Notes

- **Run the notebook before the app.** The Streamlit app depends on the `.pkl` files that the notebook generates. Skipping this step will throw a "model not found" error.
- **Memory:** Training on the full dataset may require 2GB+ RAM. If you're on a limited machine, consider sampling the data first.
- **Python version:** Developed and tested on Python 3.9+.

## 🐛 Troubleshooting

**"Model files not found"**
→ Run all cells in `compensation.ipynb` first. The `.pkl` files need to exist in the project directory before the app can load them.

**"Data file not found"**
→ Make sure `cleaned_Employee_Compensation.csv` is in the same folder as the notebook and app.

**Slow first load**
→ The Random Forest model is large. First-load is slower; subsequent predictions are cached and fast.

## 🔭 What's Next

This project has a solid foundation — here's where I'd take it next:

- [ ] **SHAP values** for explainable, individual-level predictions
- [ ] **Hyperparameter tuning** with Optuna for further performance gains
- [ ] **Time-series layer** — compensation trend analysis over years
- [ ] **Salary benchmarking** — compare SF public sector vs. private market
- [ ] **Export to CSV/PDF** — downloadable prediction reports
- [ ] **Model retraining pipeline** — automated refresh when new data drops
- [ ] **REST API** — expose predictions as an endpoint for downstream integrations


## 📝 Notes

This project was built as a data science practice exercise. The dataset is sourced from SF's public employee compensation records. All analysis is for educational purposes.

---

*Built by Mary Kinya· Last updated: March 2026 · Model v2.0*