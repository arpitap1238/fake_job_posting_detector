import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap

st.title("🔍 Fake Job/Internship Posting Detector")
st.write("Paste a job or internship posting below to check if it looks suspicious.")

@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('background.pkl', 'rb') as f:
        background = pickle.load(f)
    return model, vectorizer, background

rf_model, tfidf_final, background = load_model()

feature_names = ['has_company_logo', 'has_salary_range', 'has_department',
                  'has_company_profile', 'has_benefits', 'has_requirements',
                  'description_length', 'mentions_work_from_home',
                  'mentions_no_experience', 'mentions_immediate_start',
                  'telecommuting', 'has_questions'] + list(tfidf_final.get_feature_names_out())

user_text = st.text_area("Paste the job posting text here:", height=200)

if st.button("Analyze Posting"):
    if user_text.strip() == "":
        st.warning("Please paste some text first.")
    else:
        text_lower = user_text.lower()
        has_salary = int(any(w in text_lower for w in ['salary', '$', '₹', 'per annum', 'per month', 'ctc']))
        has_company_profile = int(any(w in text_lower for w in ['about us', 'about the company', 'our company', 'founded in']))
        mentions_wfh = int('work from home' in text_lower)
        mentions_no_exp = int('no experience' in text_lower)
        mentions_immediate = int('immediate start' in text_lower)
        desc_length = len(user_text)

        structured = np.array([[0, has_salary, 0, has_company_profile, 0, 0, desc_length,
                                 mentions_wfh, mentions_no_exp, mentions_immediate, 0, 0]])
        text_vec = tfidf_final.transform([user_text]).toarray()
        final_input = np.hstack([structured, text_vec])

        prob_fake = rf_model.predict_proba(final_input)[0][1]
        threshold = 0.4
        prediction = "FAKE" if prob_fake >= threshold else "REAL"

        st.subheader(f"Prediction: {'⚠️ ' + prediction if prediction == 'FAKE' else '✅ ' + prediction}")
        st.write(f"**Confidence (fake):** {prob_fake:.1%}")

        st.subheader("🚩 Detected Signals")
        col1, col2, col3 = st.columns(3)
        col1.metric("Salary Mentioned", "Yes" if has_salary else "No")
        col2.metric("Company Info", "Yes" if has_company_profile else "No")
        col3.metric("Desc Length", f"{desc_length} chars")

        flags = []
        if mentions_wfh: flags.append("Mentions 'work from home'")
        if mentions_no_exp: flags.append("Mentions 'no experience needed'")
        if mentions_immediate: flags.append("Mentions 'immediate start'")
        if flags:
            st.write("**Suspicious phrases found:**")
            for f in flags: st.write(f"- {f}")
        else:
            st.write("No suspicious phrases detected.")

        st.subheader("🧠 Why This Prediction? (Top Factors)")
        explainer = shap.TreeExplainer(rf_model, data=background, model_output='probability')
        shap_values = explainer.shap_values(final_input, check_additivity=False)
        values = shap_values[0][:, 1] if isinstance(shap_values, list) or shap_values.ndim == 3 else shap_values[0]
        shap_df = pd.DataFrame({'feature': feature_names, 'shap_value': values})
        shap_df['abs_value'] = shap_df['shap_value'].abs()
        top = shap_df.sort_values('abs_value', ascending=False).head(5)
        for _, row in top.iterrows():
            direction = "🔴 toward FAKE" if row['shap_value'] > 0 else "🟢 toward REAL"
            st.write(f"- **{row['feature']}**: {direction} (impact: {row['shap_value']:.3f})")
