import streamlit as st
import joblib
import os
import time
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

MODEL_PATH = './knn_model.joblib'

LABEL_MAP = {
    0: "NOT GUILTY",
    1: "GUILTY"
}

ISSUE_AREA_OPTIONS = [
    "nan (Unknown)",
    "Civil Rights",
    "Due Process",
    "First Amendment",
    "Criminal Procedure",
    "Privacy",
    "Federal Taxation",
    "Economic Activity",
    "Judicial Power",
    "Unions",
    "Federalism",
    "Attorneys",
    "Miscellaneous",
    "Interstate Relations",
    "Private Action"
]

@st.cache_resource
def load_nlp_model(path):
    if not os.path.exists(path):
        st.error(f"Error: Model file not found at location: {path}")
        st.stop()
    try:
        with st.spinner("Loading NLP model from .joblib..."):
            time.sleep(1)
            model = joblib.load(path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load the model. Ensure the .joblib file is valid and correct. Error: {e}")
        st.stop()

def get_feature_importance_explanation(pipeline, predicted_class, top_n=10):
    vectorizer = None
    classifier = None
    classifier_is_linear = False

    for step_name, step_obj in pipeline.steps:
        if 'vectorizer' in step_name and hasattr(step_obj, 'get_feature_names_out'):
            vectorizer = step_obj
        if 'classifier' in step_name:
            classifier = step_obj
            if hasattr(step_obj, 'coef_') and hasattr(step_obj.coef_, '__getitem__'):
                classifier_is_linear = True
            break

    if not classifier_is_linear or vectorizer is None:
        return "The model does not support coefficient-based explanations (it's not a linear model/KNN). Consider LIME/SHAP for more complex explanations or use other interpretations for non-linear/KNN models.", None

    feature_names = vectorizer.get_feature_names_out()
    coef = classifier.coef_
    if coef.ndim > 1:
        coef = coef[0]

    feature_importance = pd.DataFrame({'feature': feature_names, 'coefficient': coef})

    if predicted_class == 1:
        top_features = feature_importance.sort_values(by='coefficient', ascending=False).head(top_n)
        return "Words that contributed most to the **GUILTY** prediction:", top_features
    else:
        top_features = feature_importance.sort_values(by='coefficient', ascending=True).head(top_n)
        return "Words that contributed most to the **NOT GUILTY** prediction:", top_features


nlp_model = load_nlp_model(MODEL_PATH)




st.title("Supreme Court Judgment Prediction")
st.markdown("---")

st.sidebar.header("ℹ️ About")
st.sidebar.info("Over 60% of Supreme Court cases relate to criminal procedure and civil rights.")
st.sidebar.markdown("""
This tool predicts Supreme Court judgment outcomes based on textual case facts using an NLP model.

-  **Model**: KNN Classifier
-  **File**: `knn_model.joblib`
""")

st.write("""
  This model combines party names, issue area, and factual description, then vectorizes the text using NLP techniques to classify whether a situation indicates <span style='color: red;'>"**GUILTY**"</span> or <span style='color: green;'>"**NOT GUILTY**"</span> based on the text case you provide.
""", unsafe_allow_html=True)

def create_dummy_data_frame():
    data = {
        "first_party": [
            "United States",
            "Miranda",
            "Brown",
            "Roe",
            "Tinker",
            "New York Times Co."
        ],
        "second_party": [
            "Miller",
            "Arizona",
            "Board of Education",
            "Wade",
            "Des Moines Independent Community School District",
            "Sullivan"
        ],
        "issue_area": [
            "Criminal Procedure",
            "Due Process",
            "Civil Rights",
            "Privacy",
            "First Amendment",
            "First Amendment"
        ],
        "facts": [
            "The case concerned the constitutionality of the National Firearms Act of 1934, specifically its provisions related to sawed-off shotguns. The defendants argued that the act violated their Second Amendment rights.",
            "Ernesto Miranda was arrested for kidnapping and rape. He confessed after interrogation but was not informed of his right to an attorney or to remain silent.",
            "Segregation of students in public schools violates the Equal Protection Clause of the Fourteenth Amendment, even if the segregated schools are otherwise equal in quality.",
            "A pregnant single woman challenged a Texas law that prohibited abortions, arguing that it violated her constitutional right to privacy.",
            "Students wore black armbands to protest the Vietnam War and were suspended. They argued their First Amendment right to freedom of speech.",
            "An advertisement was published in The New York Times soliciting funds for the civil rights movement, containing minor factual inaccuracies regarding an Alabama city commissioner, L.B. Sullivan. Sullivan sued for libel."
        ]
    }
    lengths = {key: len(value) for key, value in data.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"All arrays in 'data' must be of the same length. Current lengths: {lengths}")

    return pd.DataFrame(data)

dummy_cases_df = create_dummy_data_frame()



DEFAULT_DUMMY_CASE = dummy_cases_df.iloc[0]

try:
    default_issue_area_index = ISSUE_AREA_OPTIONS.index(DEFAULT_DUMMY_CASE['issue_area'])
except ValueError:
    default_issue_area_index = 0


st.header("Case Information")

col1, col2 = st.columns(2)
with col1:
    first_party_input = st.text_input(
        label="Primary party",
            value=DEFAULT_DUMMY_CASE['first_party'],
            placeholder="e.g., Furina, John Doe",
            help="The primary party involved in the case.",
            key="first_party_text_input",
            label_visibility="visible"
    )
with col2:
    second_party_input = st.text_input(
        "Second Party:",
        value=DEFAULT_DUMMY_CASE['second_party'],
        placeholder="Example: Jeremy",
        key="second_party_text_input"
    )

issue_area_input = st.selectbox(
    "Issue Area:",
    options=ISSUE_AREA_OPTIONS,
    index=default_issue_area_index,
    key="issue_area_selectbox"
)

facts_input = st.text_area(
    "Case Facts:",
    value=DEFAULT_DUMMY_CASE['facts'],
    placeholder="Example: An anonymous report indicated suspicious activity at location X at Y. Three people were seen quickly leaving the area.",
    height=150,
    key="facts_text_input"
)

user_input_main = ""

if st.button("Classify"):
    is_any_text_input_filled = (first_party_input or second_party_input or facts_input)
    is_issue_area_meaningful = (issue_area_input != "nan (Unknown)")

    if not (is_any_text_input_filled or is_issue_area_meaningful):
        st.warning("Please enter at least one text input or select a specific Issue Area for analysis.")
        st.stop()
    

    with st.spinner("Analyzing text..."):
        try:
            processed_first_party = f"first party: {first_party_input}" if first_party_input else ""
            processed_second_party = f"second party: {second_party_input}" if second_party_input else ""
            processed_facts_input = f"facts: {facts_input}" if facts_input else ""
            processed_issue_area_input = f"issue area: {issue_area_input}" if issue_area_input and issue_area_input != "nan (Unknown)" else ""

            combined_text_for_model = " ".join(filter(None, [
                processed_first_party,
                processed_second_party,
                processed_issue_area_input,
                processed_facts_input
            ]))

            if not combined_text_for_model.strip():
                st.warning("Combined input is empty. Please provide enough information for analysis.")
                st.stop()

            prediction_raw = nlp_model.predict([combined_text_for_model])
            predicted_class = prediction_raw[0]

            confidence_score = None
            if hasattr(nlp_model, 'predict_proba'):
                prediction_proba = nlp_model.predict_proba([combined_text_for_model])
                confidence_score = prediction_proba[0][1]

            predicted_label = LABEL_MAP.get(predicted_class, "Unknown Label")

            st.markdown("## Prediction Result")
            st.markdown("---")

            st.subheader("Case Classification:")
            if predicted_label == "GUILTY":
                st.error(f"**{predicted_label}**")
                if confidence_score is not None:
                    st.write(f"Model Confidence (GUILTY): **{confidence_score:.2%}**")
            elif predicted_label == "NOT GUILTY":
                st.success(f"**{predicted_label}**")
                if confidence_score is not None:
                    st.write(f"Model Confidence (NOT GUILTY): **{1 - confidence_score:.2%}**")
            else:
                st.info(f"**{predicted_label}**")
                if confidence_score is not None:
                    st.write(f"Model Confidence: **{confidence_score:.2%}**")

            st.markdown("---")


            st.subheader("Reason for Model's Prediction:")
            explanation_message, top_features_df = get_feature_importance_explanation(nlp_model, predicted_class)



            if top_features_df is None:
                st.info(explanation_message)
            else:
                st.write(explanation_message)
                st.dataframe(top_features_df.style.format({'coefficient': "{:.4f}"}))
                st.caption("Note: Positive coefficients indicate support for the 'GUILTY' class, while negative coefficients indicate support for the 'NOT GUILTY' class.")


            st.markdown("---")
            st.markdown("### Combined Input Used by Model")
            st.code(combined_text_for_model)
            st.markdown("---")
            st.write("Note: This classification is the result of an AI model and should be used as an assistive tool, not as a substitute for professional legal judgment.")

        except Exception as e:
            st.error(f"An error occurred during prediction. Error: {e}")
            st.write("Ensure the input combination format matches your model's expectation and the model file is valid.")
            st.write(f"Error details: {e}")


clear_clicked = st.button("Clear Input Fields")

if clear_clicked:
    st.info("masi kosong")
    # st.session_state['first_party_text_input'] = ""
    # st.session_state['second_party_text_input'] = ""
    # st.session_state['issue_area_selectbox'] = 0
    # st.session_state['facts_text_input'] = ""
  
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #000000;
            color: #6c757d;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            z-index: 9999;
        }
        .main > div {
            padding-bottom: 60px; /* Prevent content from being hidden behind footer */
        }
    </style>

    <div class="footer">
        © 2025 SupremeCourt AI Tool
    </div>
""", unsafe_allow_html=True)
