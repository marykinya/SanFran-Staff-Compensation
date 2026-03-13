"""
San Francisco Employee Compensation Prediction - Streamlit App
A web app to predict employee compensation based on job roles and organizational attributes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="SF Employee Compensation Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ARTIFACTS ====================
@st.cache_resource
def load_model_artifacts():
    """Load the trained model and preprocessing artifacts"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, scaler, label_encoders, feature_names
    except FileNotFoundError as e:
        st.error(f"❌ Error loading model files: {e}")
        st.info("Please ensure you have run the Jupyter notebook to generate model artifacts.")
        return None, None, None, None

@st.cache_data
def load_raw_data():
    """Load the original data for reference and visualizations"""
    try:
        df = pd.read_csv('data/cleaned_Employee_Compensation.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Error loading raw data")
        return None

# Load artifacts
model, scaler, label_encoders, feature_names = load_model_artifacts()
df_raw = load_raw_data()

if model is None or df_raw is None:
    st.stop()

# ==================== HELPER FUNCTIONS ====================
def prepare_prediction_input(year_type, year, organization_group, department, job_family, 
                            job, union, salaries, overtime, other_salaries, 
                            retirement, health_dental, other_benefits):
    """Prepare input data for model prediction"""
    try:
        # Store original year_type before encoding
        is_fiscal = (year_type == 'Fiscal')
        
        # Create input dataframe
        input_data = pd.DataFrame([{
            'year_type': year_type,
            'year': year,
            'organization_group': organization_group,
            'department': department,
            'job_family': job_family,
            'job': job,
            'union': union,
            'salaries': salaries,
            'overtime': overtime,
            'other_salaries': other_salaries,
            'retirement': retirement,
            'health_and_dental': health_dental,
            'other_benefits': other_benefits,
        }])
        
        # Encode categorical variables
        for col in ['year_type', 'organization_group', 'department', 'job_family', 'job', 'union']:
            if col in label_encoders:
                try:
                    input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
                except (ValueError, KeyError):
                    # Use default value for unknown categories
                    input_data[col] = 0
        
        # Handle duplicate 'job' column from training data (pandas renamed it to 'job.1')
        input_data['job.1'] = input_data['job']

        # Calculate engineered features
        input_data['salary_component'] = input_data['salaries'] + input_data['overtime'] + input_data['other_salaries']
        input_data['benefits_component'] = input_data['retirement'] + input_data['health_and_dental'] + input_data['other_benefits']
        input_data['benefits_salary_ratio'] = input_data['benefits_component'] / (input_data['salary_component'] + 1)
        input_data['overtime_salary_ratio'] = input_data['overtime'] / (input_data['salaries'] + 1)
        input_data['is_fiscal_year'] = int(is_fiscal)
        
        # Select only required features in correct order
        input_processed = input_data[feature_names].astype('float64')
        
        return input_processed, input_data
    except Exception as e:
        st.error(f"❌ Error preparing input: {str(e)}")
        return None, None

# ==================== MAIN APP ====================
st.markdown('<div class="header-title">💰 SF Employee Compensation Predictor</div>', unsafe_allow_html=True)
st.write("Predict total employee compensation based on job roles and organizational attributes")
st.divider()

# Sidebar for navigation
page = st.sidebar.radio(
    "Navigation",
    ["🔮 Prediction", "📊 Data Analysis", "📈 Model Performance", "ℹ️ About"]
)

# ==================== PAGE 1: PREDICTION ====================
if page == "🔮 Prediction":
    st.header("Predict Employee Compensation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Job Details")
        
        # Get unique values from data
        years = sorted(df_raw['year'].unique())
        year_types = df_raw['year_type'].unique()
        organizations = sorted(df_raw['organization_group'].unique())
        departments = sorted(df_raw['department'].unique())
        job_families = sorted(df_raw['job_family'].unique())
        jobs = sorted(df_raw['job'].unique())
        unions = sorted(df_raw['union'].unique())
        
        # Create input form
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                year_type = st.selectbox("Year Type", year_types)
                year = st.selectbox("Year", years)
                organization_group = st.selectbox("Organization Group", organizations)
                department = st.selectbox("Department", departments)
            
            with col_b:
                job_family = st.selectbox("Job Family", job_families)
                job = st.selectbox("Job Title", jobs)
                union = st.selectbox("Union", unions)
                
            st.subheader("Compensation Components")
            col_c, col_d = st.columns(2)
            
            with col_c:
                # Get median values for suggestions
                salary_median = df_raw[df_raw['job'] == job]['salaries'].median() if job in df_raw['job'].values else 50000
                overtime_median = df_raw[df_raw['job'] == job]['overtime'].median() if job in df_raw['job'].values else 2000
                other_salary_median = df_raw[df_raw['job'] == job]['other_salaries'].median() if job in df_raw['job'].values else 1000
                
                salaries = st.number_input("Base Salaries ($)", value=int(salary_median), step=1000)
                overtime = st.number_input("Overtime ($)", value=int(overtime_median), step=100)
                other_salaries = st.number_input("Other Salaries ($)", value=int(other_salary_median), step=100)
            
            with col_d:
                retirement_median = df_raw[df_raw['job'] == job]['retirement'].median() if job in df_raw['job'].values else 5000
                health_median = df_raw[df_raw['job'] == job]['health_and_dental'].median() if job in df_raw['job'].values else 3000
                benefits_median = df_raw[df_raw['job'] == job]['other_benefits'].median() if job in df_raw['job'].values else 2000
                
                retirement = st.number_input("Retirement ($)", value=int(retirement_median), step=500)
                health_dental = st.number_input("Health & Dental ($)", value=int(health_median), step=500)
                other_benefits = st.number_input("Other Benefits ($)", value=int(benefits_median), step=500)
            
            # Prediction button
            predict_button = st.form_submit_button(
                "🎯 Predict Compensation",
                use_container_width=True,
                type="primary"
            )
    
    with col2:
        st.subheader("Quick Stats")
        
        # Get statistics for selected job
        if job in df_raw['job'].values:
            job_data = df_raw[df_raw['job'] == job]
            
            st.metric("Avg Compensation", f"${job_data['total_compensation'].mean():,.0f}")
            st.metric("Median Compensation", f"${job_data['total_compensation'].median():,.0f}")
            st.metric("Count", f"{len(job_data):,}")
    
    # ==================== MAKE PREDICTION ====================
    if predict_button:
        with st.spinner("🔄 Making prediction..."):
            input_processed, input_raw = prepare_prediction_input(
                year_type, year, organization_group, department, job_family,
                job, union, salaries, overtime, other_salaries,
                retirement, health_dental, other_benefits
            )
            
            if input_processed is not None:
                # Make prediction
                prediction = model.predict(input_processed)[0]
                
                # Calculate components
                total_salary = salaries + overtime + other_salaries
                total_benefits = retirement + health_dental + other_benefits
                
                # Create results columns
                st.divider()
                st.subheader("🎯 Prediction Results")
                
                result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                
                with result_col1:
                    st.metric(
                        "Predicted Total Compensation",
                        f"${prediction:,.0f}",
                        delta=f"${prediction - (total_salary + total_benefits):,.0f}",
                        delta_color="inverse"
                    )
                
                with result_col2:
                    st.metric("Total Salary Component", f"${total_salary:,.0f}")
                
                with result_col3:
                    st.metric("Total Benefits Component", f"${total_benefits:,.0f}")
                
                with result_col4:
                    benefits_pct = (total_benefits / (total_salary + total_benefits)) * 100 if (total_salary + total_benefits) > 0 else 0
                    st.metric("Benefits %", f"{benefits_pct:.1f}%")
                
                # Comparison with actual data
                if job in df_raw['job'].values:
                    job_data = df_raw[df_raw['job'] == job]
                    actual_mean = job_data['total_compensation'].mean()
                    diff_pct = ((prediction - actual_mean) / actual_mean) * 100
                    
                    st.divider()
                    st.subheader("📊 Comparison with Historical Data")
                    col_compare1, col_compare2, col_compare3 = st.columns(3)
                    
                    with col_compare1:
                        st.metric("Predicted Compensation", f"${prediction:,.0f}")
                    
                    with col_compare2:
                        st.metric("Job Historical Average", f"${actual_mean:,.0f}")
                    
                    with col_compare3:
                        st.metric("Difference", f"{diff_pct:+.1f}%")

# ==================== PAGE 2: DATA ANALYSIS ====================
elif page == "📊 Data Analysis":
    st.header("Data Analysis & Insights")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Jobs", "Departments", "Organization", "Compensation"])
    
    with tab1:
        st.subheader("Top Jobs by Average Compensation")
        top_jobs = df_raw.groupby('job')['total_compensation'].agg(['mean', 'count']).reset_index()
        top_jobs.columns = ['Job', 'Avg Compensation', 'Count']
        top_jobs = top_jobs[top_jobs['Count'] >= 10].sort_values('Avg Compensation', ascending=False).head(15)
        
        fig = px.bar(top_jobs, x='Avg Compensation', y='Job',
                    orientation='h',
                    title='Top 15 Jobs by Average Compensation',
                    labels={'Avg Compensation': 'Average Total Compensation ($)'})
        fig.update_xaxes(tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Average Compensation by Department")
        dept_comp = df_raw.groupby('department')['total_compensation'].agg(['mean', 'count']).reset_index()
        dept_comp.columns = ['Department', 'Avg Compensation', 'Count']
        dept_comp = dept_comp.sort_values('Avg Compensation', ascending=False).head(12)
        
        fig = px.bar(dept_comp, x='Avg Compensation', y='Department',
                    orientation='h',
                    title='Top Departments by Average Compensation',
                    color='Count')
        fig.update_xaxes(tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Average Compensation by Organization Group")
        org_comp = df_raw.groupby('organization_group')['total_compensation'].agg(['mean', 'count']).reset_index()
        org_comp.columns = ['Organization Group', 'Avg Compensation', 'Count']
        org_comp = org_comp.sort_values('Avg Compensation', ascending=True)
        
        fig = px.bar(org_comp, y='Organization Group', x='Avg Compensation',
                    color='Count',
                    title='Average Compensation by Organization Group')
        fig.update_xaxes(tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Compensation Distribution")
        
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            fig = px.histogram(df_raw, x='total_compensation', nbins=50,
                             title='Distribution of Total Compensation')
            fig.update_xaxes(tickformat="$,.0f")
            st.plotly_chart(fig, use_container_width=True)
        
        with col_dist2:
            # Compensation components pie chart
            salary_cols = ['salaries', 'overtime', 'other_salaries', 'retirement', 'health_and_dental', 'other_benefits']
            avg_components = df_raw[salary_cols].mean()
            
            fig = px.pie(values=avg_components.values, names=salary_cols,
                        title='Average Compensation Components')
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE 3: MODEL PERFORMANCE ====================
elif page == "📈 Model Performance":
    st.header("📈 Model Performance & How It Works")
    st.write("A simple breakdown of how this prediction model was built, tested, and why it works so well.")

    # ── Parse the summary report ──────────────────────────────────────────────
    report_data = {}
    try:
        with open('model_summary_report.txt', 'r') as f:
            raw_report = f.read()

        for line in raw_report.splitlines():
            line = line.strip()
            if "Total Records:" in line:
                report_data['total_records'] = line.split(":")[-1].strip().replace(",", "")
            elif "Training Records:" in line:
                report_data['train_records'] = line.split(":")[-1].strip().replace(",", "")
            elif "Test Records:" in line:
                report_data['test_records'] = line.split(":")[-1].strip().replace(",", "")
            elif "Number of Features:" in line:
                report_data['n_features'] = line.split(":")[-1].strip()
            elif "Mean Compensation:" in line:
                report_data['mean_comp'] = line.split(":")[-1].strip()
            elif "Median Compensation:" in line:
                report_data['median_comp'] = line.split(":")[-1].strip()
            elif "Min Compensation:" in line:
                report_data['min_comp'] = line.split(":")[-1].strip()
            elif "Max Compensation:" in line:
                report_data['max_comp'] = line.split(":")[-1].strip()
            elif "Std Deviation:" in line:
                report_data['std_comp'] = line.split(":")[-1].strip()
            elif "Test R² Score:" in line:
                report_data['r2'] = line.split(":")[-1].strip()
            elif "Test RMSE:" in line:
                report_data['rmse'] = line.split(":")[-1].strip()
            elif "Test MAE:" in line:
                report_data['mae'] = line.split(":")[-1].strip()
            elif "Cross-Validation R²" in line:
                report_data['cv'] = line.split(":")[-1].strip()

        # Feature importance from report
        feature_lines = []
        in_features = False
        for line in raw_report.splitlines():
            if "TOP 10 IMPORTANT FEATURES" in line:
                in_features = True
                continue
            if in_features and line.strip().startswith("==="):
                break
            if in_features and ":" in line and line.strip():
                parts = line.strip().split(":")
                if len(parts) == 2:
                    name = parts[0].split(".")[-1].strip()
                    try:
                        score = float(parts[1].strip())
                        feature_lines.append((name, score))
                    except ValueError:
                        pass

    except FileNotFoundError:
        st.warning("⚠️ Model summary report not found. Please run the Jupyter notebook first.")
        feature_lines = []

    # ── Section 1: Dataset at a Glance ───────────────────────────────────────
    st.subheader("1. The Dataset at a Glance")
    with st.expander("💡 What does this mean?", expanded=False):
        st.markdown("""
        Before training any model, we need data, lots of it. Think of data as the textbook the model studies
        from. The more examples it sees, the smarter it gets.

        We split our data into two groups:
        - **Training set** — the textbook the model learns from (80% of all data)
        - **Test set** — a surprise exam with data the model has *never* seen before (20%)

        This split is critical: it tells us whether the model actually *learned* to predict compensation,
        or just memorised the textbook.
        """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{int(report_data.get('total_records', 799232)):,}")
    c2.metric("Training Records", f"{int(report_data.get('train_records', 639385)):,}")
    c3.metric("Test Records", f"{int(report_data.get('test_records', 159847)):,}")
    c4.metric("Features Used", report_data.get('n_features', '19'))

    st.divider()

    # ── Section 2: What We Were Predicting ───────────────────────────────────
    st.subheader("2. What We Were Predicting")
    with st.expander("💡 What does this mean?", expanded=False):
        st.markdown("""
        The goal of this model is to predict **total compensation** ; the full dollar amount an SF city
        employee earns, including base salary, overtime, benefits, retirement, and more.

        Before building the model, it helps to understand the *shape* of what we're predicting:
        - **Mean vs Median**: If the mean and median are close, pay is fairly evenly spread.
          Here they're nearly identical, meaning most employees cluster around ~$113K.
        - **Standard Deviation (Std Dev)**: How spread out the values are. A $75K std dev means
          salaries vary a lot — some employees earn very little, others earn much more.
        - **Range**: The gap between the lowest and highest earners shows us the full spectrum.
        """)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean Compensation", report_data.get('mean_comp', '$113,427'))
    c2.metric("Median Compensation", report_data.get('median_comp', '$113,122'))
    c3.metric("Minimum", report_data.get('min_comp', '$0.01'))
    c4.metric("Maximum", report_data.get('max_comp', '$807,625'))
    c5.metric("Std Deviation", report_data.get('std_comp', '$75,068'))

    st.divider()

    # ── Section 3: Models We Tested ──────────────────────────────────────────
    st.subheader("3. Models We Tested — The Tournament")
    with st.expander("💡 What does this mean?", expanded=False):
        st.markdown("""
        We didn't just pick one model and hope for the best. We ran a **tournament** — testing four
        different algorithms and comparing how accurately each one predicted salaries on data it had
        never seen before.

        Here's a quick plain-English description of each contestant:

        | Model | What it does |
        |---|---|
        | **Random Forest** | Builds hundreds of decision trees and combines their answers (like getting opinions from 200 advisors and taking the average). Very powerful for complex data. |
        | **Gradient Boosting** | Also uses decision trees, but builds them one at a time — each tree learns from the mistakes of the previous one. Like a student who reviews every wrong answer before the next test. |
        | **Ridge Regression** | A smarter version of drawing a straight line through data. Fast and simple, but struggles when relationships aren't linear. |
        | **Voting Ensemble** | A combination of Random Forest and Gradient Boosting voting together. Two heads are better than one — but is it better than either alone? |

        **Key metrics explained:**
        - **R² (R-squared)**: How much of the variation in salary does the model explain? 1.0 = perfect, 0 = no better than guessing the average. Anything above 0.99 is excellent.
        - **RMSE (Root Mean Squared Error)**: The average dollar amount the model is off by, with larger errors penalised more. Lower = better.
        - **MAE (Mean Absolute Error)**: The average dollar amount the model is off by, treating all errors equally. Lower = better.
        - **CV R² (Cross-Validation)**: We tested R² across 5 different random splits of the data to make sure results are consistent, not lucky.
        """)

    # Model comparison data (from notebook run)
    models_data = {
        'Model': ['Random Forest', 'Gradient Boosting', 'Voting Ensemble', 'Ridge Regression'],
        'Train R²': [0.9999, 0.9992, 0.9997, 0.9697],
        'Test R²':  [0.9995, 0.9991, 0.9994, 0.9701],
        'Test RMSE ($)': [1702.91, 2277.42, 1776.34, 12940.82],
        'Test MAE ($)': [470.75, 1084.80, 732.70, 6984.96],
        'CV R²': [0.9995, 0.9991, 0.9995, 0.9696],
    }
    df_models = pd.DataFrame(models_data)

    # R² comparison chart
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig_r2 = go.Figure()
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
        colors_light = ['rgba(46,204,113,0.4)', 'rgba(52,152,219,0.4)', 'rgba(155,89,182,0.4)', 'rgba(231,76,60,0.4)']
        fig_r2.add_trace(go.Bar(
            x=df_models['Model'], y=df_models['Train R²'],
            name='Train R²', marker_color=colors_light,
            text=[f"{v:.4f}" for v in df_models['Train R²']], textposition='outside'
        ))
        fig_r2.add_trace(go.Bar(
            x=df_models['Model'], y=df_models['Test R²'],
            name='Test R²', marker_color=colors,
            text=[f"{v:.4f}" for v in df_models['Test R²']], textposition='outside'
        ))
        fig_r2.update_layout(
            title='R² Score by Model (higher = better)',
            barmode='group', yaxis=dict(range=[0.93, 1.002], title='R² Score'),
            legend=dict(orientation='h', y=-0.2)
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_chart2:
        fig_rmse = go.Figure()
        fig_rmse.add_trace(go.Bar(
            x=df_models['Model'], y=df_models['Test RMSE ($)'],
            marker_color=colors,
            text=[f"${v:,.0f}" for v in df_models['Test RMSE ($)']],
            textposition='outside'
        ))
        fig_rmse.update_layout(
            title='Test RMSE by Model (lower = better)',
            yaxis=dict(title='RMSE ($)', tickformat='$,.0f'),
            showlegend=False
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    # MAE comparison
    fig_mae = go.Figure()
    fig_mae.add_trace(go.Bar(
        x=df_models['Model'], y=df_models['Test MAE ($)'],
        marker_color=colors,
        text=[f"${v:,.0f}" for v in df_models['Test MAE ($)']],
        textposition='outside'
    ))
    fig_mae.update_layout(
        title='Test MAE by Model — Average Prediction Error in Dollars (lower = better)',
        yaxis=dict(title='MAE ($)', tickformat='$,.0f'),
        showlegend=False
    )
    st.plotly_chart(fig_mae, use_container_width=True)

    # Summary table
    st.markdown("**Full Results Table**")
    df_display = df_models.copy()
    df_display['Test RMSE ($)'] = df_display['Test RMSE ($)'].apply(lambda x: f"${x:,.2f}")
    df_display['Test MAE ($)'] = df_display['Test MAE ($)'].apply(lambda x: f"${x:,.2f}")
    df_display['Train R²'] = df_display['Train R²'].apply(lambda x: f"{x:.4f}")
    df_display['Test R²'] = df_display['Test R²'].apply(lambda x: f"{x:.4f}")
    df_display['CV R²'] = df_display['CV R²'].apply(lambda x: f"{x:.4f}")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.divider()

    # ── Section 4: Why Random Forest Won ─────────────────────────────────────
    st.subheader("4. Why Random Forest Won")
    with st.expander("💡 Detailed explanation", expanded=True):
        st.markdown("""
        After comparing all four models, **Random Forest** came out on top across every metric. Here's why:

        **vs Ridge Regression:**
        Ridge draws a single straight line (or hyperplane) through the data. But the relationship
        between job type, seniority, department, and compensation is far from linear.
        Ridge's RMSE of ~$12,941 was almost **8× worse** than Random Forest's ~$1,703.
        It simply couldn't capture the complex patterns in the data.

        **vs Gradient Boosting:**
        Gradient Boosting is a strong model and came in second. However, on a dataset of 800K records
        with 19 features, it was slower to train and still had a higher error ($2,277 RMSE vs $1,703).
        Random Forest parallelises its trees, making it faster and equally accurate.

        **vs Voting Ensemble:**
        The Voting Ensemble combined Random Forest and Gradient Boosting, hoping the two would
        complement each other. In practice, the Gradient Boosting component slightly dragged down
        the ensemble (RMSE $1,776 vs $1,703 for Random Forest alone). Sometimes simpler wins.

        **Random Forest's winning stats:**
        - **Test R² = 0.9995** — the model explains 99.95% of all salary variation
        - **Test RMSE = $1,703** — on average, predictions are off by only ~$1,703 (on salaries averaging $113K)
        - **Test MAE = $471** — the *typical* prediction error is just $471
        - **CV R² = 0.9995 ± 0.0000** — rock-steady across all 5 validation folds; no signs of overfitting
        - **Train vs Test gap is tiny** — which confirms the model generalises, not memorises
        """)

    # Gap chart: train vs test R²
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Scatter(
        x=df_models['Model'],
        y=(df_models['Train R²'] - df_models['Test R²']).round(6),
        mode='markers+lines+text',
        marker=dict(size=14, color=colors),
        text=[(f"{v:.6f}") for v in (df_models['Train R²'] - df_models['Test R²'])],
        textposition='top center',
        name='Train-Test R² Gap'
    ))
    fig_gap.update_layout(
        title='Train vs Test R² Gap (closer to 0 = less overfitting)',
        yaxis=dict(title='R² Gap (Train − Test)', tickformat='.6f'),
        showlegend=False
    )
    st.plotly_chart(fig_gap, use_container_width=True)
    st.caption("A tiny gap between training and test performance means the model learned *general* patterns, not just the training data. Random Forest shows almost zero gap.")

    st.divider()

    # ── Section 5: Winning Model Deep-Dive ───────────────────────────────────
    st.subheader("5. The Winning Model — Random Forest Deep Dive")

    col_arch, col_perf = st.columns(2)

    with col_arch:
        st.markdown("##### Architecture Settings")
        with st.expander("💡 What do these settings mean?", expanded=False):
            st.markdown("""
            - **Estimators (200)**: The number of decision trees in the forest. More trees = more stable predictions.
            - **Max Depth (20)**: How many questions each tree is allowed to ask before making a prediction. Deeper = more detail, but risks memorising noise.
            - **Min Samples Split (5)**: A branch only splits if there are at least 5 data points. Prevents overly specific tiny branches.
            - **Min Samples Leaf (2)**: Each leaf (final answer node) must represent at least 2 records. Avoids one-off fringe cases dominating predictions.
            """)
        st.markdown("""
        | Setting | Value |
        |---|---|
        | Algorithm | Random Forest Regressor |
        | Number of Trees | 200 |
        | Max Tree Depth | 20 |
        | Min Samples to Split | 5 |
        | Min Samples per Leaf | 2 |
        | Training Data Size | 639,385 records |
        | Validation Strategy | 5-Fold Cross-Validation |
        """)

    with col_perf:
        st.markdown("##### Final Performance on Unseen Test Data")
        with st.expander("💡 What do these numbers mean?", expanded=False):
            st.markdown("""
            - **R² = 0.9995**: If you picked a random SF employee, the model can explain 99.95% of why their pay is what it is. Only 0.05% of the variation is unexplained.
            - **RMSE = $1,706**: Think of this as a "typical worst-case error" — most predictions land within ~$1,700 of the real figure.
            - **MAE = $470**: Half the time the model is off by less than $470 on a salary that averages $113,000 — that's under 0.5% error.
            - **Cross-Validation R² = 0.9995 ± 0.0000**: Tested on 5 different random data splits. The ± 0.0000 means results were *identical* across all splits — extremely stable.
            """)
        r2_val   = float(report_data.get('r2', '0.9995'))
        rmse_val = report_data.get('rmse', '$1,705.99')
        mae_val  = report_data.get('mae', '$469.92')
        cv_val   = report_data.get('cv', '0.9995 ± 0.0000')

        st.metric("R² Score", r2_val, help="Closer to 1.0 is better. Explains % of salary variation.")
        st.metric("Test RMSE", rmse_val, help="Average prediction error (larger errors penalised more)")
        st.metric("Test MAE", mae_val, help="Typical prediction error in dollars")
        st.metric("Cross-Validation R²", cv_val, help="Consistency across 5 data splits")

    st.divider()

    # ── Section 6: What Drives the Predictions ───────────────────────────────
    st.subheader("6. What Drives the Predictions? — Feature Importance")
    with st.expander("💡 What does this mean?", expanded=False):
        st.markdown("""
        After training, the Random Forest can tell us which pieces of information it relied on
        *most* when making predictions. This is called **feature importance**.

        A score of 1.0 means that feature alone explains everything. A score of 0.0 means the
        model ignored it completely.

        Think of it like asking a recruiter: *"When you figure out someone's salary, what matters most?"*
        Their answer here is **salary components** (the sum of base pay + overtime + other salaries)
        — which makes intuitive sense, since total compensation is built directly from those parts.
        """)

    if feature_lines:
        feat_df = pd.DataFrame(feature_lines, columns=['Feature', 'Importance'])
        feat_df = feat_df.sort_values('Importance', ascending=True)

        # Clean feature names for display
        name_map = {
            'salary_component': 'Salary Component (base+OT+other)',
            'other_benefits': 'Other Benefits',
            'benefits_component': 'Benefits Component (retirement+health+other)',
            'salaries': 'Base Salaries',
            'retirement': 'Retirement',
            'health_and_dental': 'Health & Dental',
            'other_salaries': 'Other Salaries',
            'benefits_salary_ratio': 'Benefits-to-Salary Ratio',
            'overtime_salary_ratio': 'Overtime-to-Salary Ratio',
            'overtime': 'Overtime',
        }
        feat_df['Feature'] = feat_df['Feature'].map(name_map).fillna(feat_df['Feature'])

        fig_feat = px.bar(
            feat_df, x='Importance', y='Feature',
            orientation='h',
            title='Feature Importance — What the Model Relies On Most',
            color='Importance',
            color_continuous_scale='Blues',
            text=feat_df['Importance'].apply(lambda x: f"{x:.4f}")
        )
        fig_feat.update_traces(textposition='outside')
        fig_feat.update_layout(coloraxis_showscale=False, xaxis_title='Importance Score (0–1)')
        st.plotly_chart(fig_feat, use_container_width=True)

        # Breakdown insight
        top_feat = feat_df.sort_values('Importance', ascending=False).iloc[0]
        st.info(f"**Top driver:** *{top_feat['Feature']}* accounts for **{top_feat['Importance']:.1%}** of the model's decision-making. This is the computed total of base salary + overtime + other salary components — the single biggest predictor of what an employee takes home.")

# ==================== PAGE 4: ABOUT ====================
elif page == "ℹ️ About":
    st.header("About This Application")
    
    st.markdown("""
    ### San Francisco Employee Compensation Prediction
    
    This application predicts total employee compensation for San Francisco city employees 
    based on their job role and organizational attributes.
    
    #### Features
    - **Real-time Predictions**: Enter job details and get instant compensation estimates
    - **Data-Driven Insights**: Visualizations of compensation across different dimensions
    - **Model Transparency**: View model performance and feature importance
    - **Historical Comparison**: Compare predictions with historical compensation data
    
    #### Data Source
    The model is trained on cleaned San Francisco employee compensation data including:
    - Base salaries
    - Overtime pay
    - Retirement contributions
    - Health and dental benefits
    - Other benefits
    
    #### How It Works
    1. **Data Preprocessing**: Categorical encoding and feature normalization
    2. **Feature Engineering**: Creation of derived features like compensation ratios
    3. **Model Training**: Random Forest algorithm trained on historical data
    4. **Prediction**: User inputs are processed and fed to the trained model
    
    #### Model Performance
    - **R² Score**: Explains the variance in compensation predictions
    - **RMSE**: Measures average prediction error
    - **Cross-Validation**: Ensures model generalization
    
    #### Technology Stack
    - **Python 3.9+**
    - **Streamlit**: Web framework
    - **Scikit-learn**: Machine learning
    - **Pandas**: Data manipulation
    - **Plotly**: Interactive visualizations
    
    #### Files Generated
    - `best_model.pkl`: Trained Random Forest model
    - `scaler.pkl`: Feature scaler
    - `label_encoders.pkl`: Categorical encoders
    - `feature_names.pkl`: Feature names used in training
    - `model_summary_report.txt`: Detailed model report
    
    ---
    **Version**: 2.0 (Improved version)
    **Last Updated**: March 2026
    """)
    
    st.divider()
    
    st.subheader("Data Statistics")
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("Total Records", f"{len(df_raw):,}")
    
    with col_stats2:
        st.metric("Avg Compensation", f"${df_raw['total_compensation'].mean():,.0f}")
    
    with col_stats3:
        st.metric("Unique Jobs", df_raw['job'].nunique())
    
    with col_stats4:
        st.metric("Unique Departments", df_raw['department'].nunique())

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    San Francisco Employee Compensation Prediction Tool | Powered by Random Forest ML model, done by Mary Kinya!
</div>
""", unsafe_allow_html=True)
