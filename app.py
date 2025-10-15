"""
Streamlit app: Alloy Property Predictor & Alloy Type Helper
Single-file Streamlit app updated to:
 - Ask for CSV upload
 - Let the user preview and choose features + targets (numeric and categorical)
 - Provide separate tabs for Heatmap, Pairplots, Distributions, Model, Predict, and Alloy Type Helper
 - Train models (RandomForest) from the CSV in-session (no saving/loading of models)
 - Store trained models in Streamlit session_state so Predict uses the latest in-session model
 - No joblib or pickle used; retrain from CSV each session

Notes:
 - Save as streamlit_alloy_app.py and run with:
     streamlit run streamlit_alloy_app.py
 - Install dependencies: streamlit pandas scikit-learn matplotlib seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import json

st.set_page_config(page_title="Alloy Property Predictor", layout="wide")

# ---------- Helper functions ----------

def suggest_alloy_type(composition: dict):
    comp = {k.strip().lower(): float(v) for k, v in composition.items() if v not in [None, "", "-"]}
    def p(el):
        return comp.get(el, 0.0)

    fe = p('fe')
    al = p('al')
    cu = p('cu')
    ni = p('ni')
    ti = p('ti')
    cr = p('cr')
    mn = p('mn')
    si = p('si')
    c = p('c')
    zn = p('zn')
    sn = p('sn')

    if fe > 40 and cr >= 10.5:
        if ni >= 8:
            return 'Austenitic stainless steel (e.g., 300-series)', 'High Fe with Cr >= 10.5% and Ni present (austenitic stainless likely)'
        else:
            return 'Stainless steel (ferritic/martensitic)', 'High Fe with Cr >= 10.5% (stainless behavior expected)'
    if fe > 40 and c > 0.05:
        return 'Carbon/alloy steel', f'High Fe ({fe}%) with C {c}% -> carbon/alloy steel family'
    if fe > 40 and (mn > 0 or si > 0 or cr > 0 or ni > 0):
        return 'Alloy steel', 'Fe-dominant with alloying elements (Mn/Si/Cr/Ni present)'
    if al > 40:
        return 'Aluminium alloy', f'Aluminium {al}% dominant -> aluminum alloy family'
    if cu > 40 and zn > 0:
        return 'Brass (Cu-Zn)', f'Cu {cu}% with Zn {zn}% suggests brass'
    if cu > 50 and sn > 0:
        return 'Bronze (Cu-Sn)', f'Cu {cu}% with Sn {sn}% suggests bronze'
    if ti > 40:
        return 'Titanium alloy', f'Ti {ti}% dominant -> titanium alloy family'
    if ni > 40:
        return 'Nickel alloy (e.g., Inconel family)', f'Ni {ni}% dominant -> nickel-based superalloy family'

    if comp:
        max_el = max(comp.items(), key=lambda x: x[1])
        el, pct = max_el
        if pct >= 30:
            return f'{el.capitalize()}-rich alloy (approx {pct}%)', f'Major element {el.upper()} at {pct}%'

    return 'Unknown / mixed composition', 'No clear dominant element or rule matched'


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    plt.tight_layout()
    return fig


def plot_pairplot_sample(df, cols):
    # sample to limit size
    sample = df[cols].dropna()
    if len(sample) > 800:
        sample = sample.sample(800, random_state=0)
    g = sns.pairplot(sample)
    return g


def plot_actual_vs_pred(actual, predicted):
    fig, ax = plt.subplots()
    ax.scatter(actual, predicted, alpha=0.6)
    mn = min(min(actual), min(predicted))
    mx = max(max(actual), max(predicted))
    ax.plot([mn, mx], [mn, mx], '--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')
    plt.tight_layout()
    return fig

# ---------- Session state init ----------
if 'models' not in st.session_state:
    st.session_state['models'] = {}  # will contain 'regression' and/or 'classification' entries

# ---------- UI ----------
st.title('🛠️ Alloy Property Predictor — No Model Saving (CSV-only)')
st.markdown('Upload a CSV with element percentages and target columns. Models are trained in-session from the CSV; no model files are saved or loaded.')

uploaded_file = st.file_uploader('Upload alloy CSV (headers in first row). Example columns: Fe, C, Cr, Ni, Al, Cu, Ti, Zn, Sn, tensile_strength, Alloy_Type', type=['csv', 'txt'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success('CSV loaded')
    except Exception as e:
        st.error('Failed to read CSV: ' + str(e))
        df = None
else:
    df = None

# Tabs for visualizations and actions
tabs = st.tabs(['Data', 'Heatmap', 'Pairplots', 'Distributions', 'Model', 'Predict', 'Alloy Type Helper'])

# Data tab
with tabs[0]:
    st.header('Data')
    if df is None:
        st.info('Please upload a CSV to start.')
    else:
        st.subheader('Preview (first 200 rows)')
        st.dataframe(df.head(200))
        st.markdown('---')
        st.subheader('Column info & basic stats')
        st.write(df.describe(include='all'))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col1, col2 = st.columns(2)
        with col1:
            features_selected = st.multiselect('Select feature columns (element percentages)', numeric_cols, default=[c for c in numeric_cols if c.lower() in ['fe','c','cr','ni','al','cu','ti','zn','sn','mn','si']])
        with col2:
            reg_target = st.selectbox('Numeric target (regression, optional)', options=[None] + numeric_cols)
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            class_target = st.selectbox('Categorical target (classification, optional)', options=[None] + cat_cols)

# Heatmap tab
with tabs[1]:
    st.header('Correlation Heatmap')
    if df is None:
        st.info('Upload CSV to view heatmap')
    else:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            st.warning('Need at least two numeric columns for a correlation heatmap')
        else:
            fig = plot_correlation_heatmap(num_df)
            st.pyplot(fig)

# Pairplots tab
with tabs[2]:
    st.header('Pairplots')
    if df is None:
        st.info('Upload CSV to view pairplots')
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_for_pair = st.multiselect('Select numeric columns for pairplot (max 6 recommended)', numeric_cols, default=numeric_cols[:5])
        if len(cols_for_pair) >= 2:
            with st.spinner('Generating pairplot (may take time)...'):
                g = plot_pairplot_sample(df, cols_for_pair)
                st.pyplot(g.fig)

# Distributions tab
with tabs[3]:
    st.header('Distributions')
    if df is None:
        st.info('Upload CSV to view distributions')
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        chosen = st.multiselect('Choose numeric columns to plot distributions', numeric_cols, default=numeric_cols[:4])
        for col in chosen:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True)
            ax.set_title(f'Distribution — {col}')
            st.pyplot(fig)

# Model tab
with tabs[4]:
    st.header('Model Training (in-session)')
    if df is None:
        st.info('Upload CSV and select features/targets in the Data tab')
    else:
        if not features_selected:
            st.warning('Choose feature columns in Data tab first')
        else:
            task_choice = st.selectbox('Train which models?', options=['Regression only', 'Classification only', 'Both'], index=0)
            test_size = st.slider('Test set proportion', 0.1, 0.5, 0.2)
            random_state = 42

            if st.button('Train model(s) now'):
                trained = {}
                # Regression
                if task_choice in ['Regression only', 'Both']:
                    if reg_target is None:
                        st.error('No numeric target selected for regression (choose in Data tab)')
                    else:
                        data_reg = df[features_selected + [reg_target]].dropna()
                        X = data_reg[features_selected]
                        y = data_reg[reg_target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                        rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
                        rf.fit(X_train, y_train)
                        preds = rf.predict(X_test)
                        metrics = {streamlit 
                            'r2': r2_score(y_test, preds),
                            'mse': mean_squared_error(y_test, preds),
                            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
                            'mae': mean_absolute_error(y_test, preds)
                        }
                        fi = pd.Series(rf.feature_importances_, index=features_selected).sort_values(ascending=False)
                        trained['regression'] = {'model': rf, 'features': features_selected, 'target': reg_target, 'metrics': metrics, 'X_test': X_test, 'y_test': y_test, 'preds': preds, 'feature_importances': fi}

                # Classification
                if task_choice in ['Classification only', 'Both']:
                    if class_target is None:
                        st.error('No categorical target selected for classification (choose in Data tab)')
                    else:
                        data_clf = df[features_selected + [class_target]].dropna()
                        Xc = data_clf[features_selected]
                        yc = data_clf[class_target].astype(str)
                        Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=test_size, random_state=random_state)
                        rfc = RandomForestClassifier(n_estimators=100, random_state=random_state)
                        rfc.fit(Xc_train, yc_train)
                        preds_c = rfc.predict(Xc_test)
                        metrics_c = {'accuracy': accuracy_score(yc_test, preds_c)}
                        fi_c = pd.Series(rfc.feature_importances_, index=features_selected).sort_values(ascending=False)
                        trained['classification'] = {'model': rfc, 'features': features_selected, 'target': class_target, 'metrics': metrics_c, 'X_test': Xc_test, 'y_test': yc_test, 'preds': preds_c, 'feature_importances': fi_c}

                # store in session_state
                st.session_state['models'] = trained
                st.success('Training complete — models stored in session')

                # Display results
                if 'regression' in trained:
                    out = trained['regression']
                    st.subheader('Regression results')
                    st.metric('R²', f"{out['metrics']['r2']:.4f}")
                    st.metric('RMSE', f"{out['metrics']['rmse']:.4f}")
                    st.metric('MAE', f"{out['metrics']['mae']:.4f}")
                    st.write('Sample testset predictions:')
                    sample_df = out['X_test'].copy()
                    sample_df['actual'] = out['y_test'].values
                    sample_df['predicted'] = out['preds']
                    st.dataframe(sample_df.head(50))
                    st.pyplot(plot_actual_vs_pred(out['y_test'].values, out['preds']))
                    st.subheader('Feature importances (regression)')
                    st.bar_chart(out['feature_importances'])

                if 'classification' in trained:
                    outc = trained['classification']
                    st.subheader('Classification results')
                    st.metric('Accuracy', f"{outc['metrics']['accuracy']:.4f}")
                    st.write('Classification report (test set):')
                    st.text(classification_report(outc['y_test'], outc['preds']))
                    st.subheader('Feature importances (classification)')
                    st.bar_chart(outc['feature_importances'])

# Predict tab
with tabs[5]:
    st.header('Predict (uses in-session trained models)')
    if not st.session_state.get('models'):
        st.info('No trained model in session. Train a model in the Model tab first (models are not saved to disk).')
    else:
        models = st.session_state['models']
        # choose which model to use
        keys = list(models.keys())
        sel = st.selectbox('Choose model for prediction', options=keys)
        meta = models[sel]
        features_needed = meta['features']
        st.subheader(f'Enter values for features (model expects: {features_needed})')
        user_vals = {}
        cols = st.columns(3)
        for i, f in enumerate(features_needed):
            with cols[i % 3]:
                user_vals[f] = st.text_input(f, '')

        if st.button('Predict now'):
            try:
                row = [float(user_vals.get(f, '') or 0.0) for f in features_needed]
                X_new = pd.DataFrame([row], columns=features_needed)
                model_obj = meta['model']
                # wrap in pipeline with scaler for safety
                pipeline = Pipeline([('scaler', StandardScaler()), ('model', model_obj)])
                pred = pipeline.predict(X_new)
                if sel == 'regression':
                    st.metric('Predicted value', f"{pred[0]:.4f}")
                else:
                    st.metric('Predicted class', str(pred[0]))
            except Exception as e:
                st.error('Prediction failed: ' + str(e))

# Alloy Type Helper
with tabs[6]:
    st.header('Alloy Type Helper')
    st.markdown('Paste composition like `Fe:72, C:0.5, Cr:12, Ni:8` or enter values below')
    text = st.text_area('Enter composition (comma-separated key:value pairs or JSON)', height=140)
    input_cols = ['Fe','C','Cr','Ni','Al','Cu','Ti','Zn','Sn','Mn','Si']
    st.subheader('Or enter element percentages manually')
    user_input = {}
    cols = st.columns(3)
    for i, el in enumerate(input_cols):
        with cols[i % 3]:
            user_input[el] = st.text_input(el, '')

    if st.button('Analyze composition'):
        try:
            comp = {}
            if text and '{' in text and '}' in text:
                comp = json.loads(text)
            elif text:
                parts = [p.strip() for p in text.replace(';', ',').split(',') if p.strip()]
                for p in parts:
                    if ':' in p:
                        k, v = p.split(':', 1)
                        comp[k.strip()] = float(v.strip())
            # merge manual inputs (manual overrides text)
            for k, v in user_input.items():
                if v:
                    try:
                        comp[k] = float(v)
                    except:
                        pass
            alloy, reason = suggest_alloy_type(comp)
            st.success(alloy)
            st.write(reason)
        except Exception as e:
            st.error('Failed to parse composition: ' + str(e))

st.caption('Models are trained in-session only and will be lost if you refresh or close the app. Always validate ML outputs with lab tests before use.')

