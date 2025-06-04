import streamlit as st
import joblib
import os
import time
import pandas as pd
# from wordcloud import WordCloud
import logging
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from openai import OpenAI

logger = logging.getLogger(__name__)


MODEL_PATH = './naive_bayes.joblib' 

LABEL_MAP = {
    True: "First Party Win",
    False: "Second Party Win"
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
st.sidebar.info("On average, the Supreme Court receives about 7,000 petitions for writs of certiorari each year, but only grants about 80.")
st.sidebar.markdown("""
This tool predicts Supreme Court judgment outcomes based on textual case facts using an NLP model.

-  **Model**: Gaussian Naive Bayes
-  **File**: `naive_bayes.joblib`
-  **Source**: [Github Repository](https://github.com/dheovanwa/Supreme-court-judgement-classification)
-  **Prediction reasoning**: meta-llama/llama-3.3-8b
""")

st.write("""
  This model combines issue area, and factual description, then vectorizes the text using NLP techniques to classify who won <span style='color: blue;'>"**First Party**"</span> or <span style='color: blue;'>"**Second Party**"</span> based on the text case you provide.
""", unsafe_allow_html=True)

def processed_facts(facts):
    if not facts:
        return ""
    
    facts = facts.lower()
    
    facts = BeautifulSoup(facts, "html.parser").get_text()
    facts = re.sub(r'\s+', ' ', facts)
    facts = re.sub(r'[^\w\s]', '', facts) 

    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    fact_tokenized = nltk.word_tokenize(facts)

    lemmatizer = WordNetLemmatizer()

    words = fact_tokenized
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    fact_lemmatized = ' '.join(words)

    return fact_lemmatized

def summarize_results(facts, issue_area, prediction, first_party, second_party):
    prompt = f"""
        Analyze the following legal case scenario:

        Case Facts:
        '{facts}'

        Issue Area:
        '{issue_area}'

        First Party:
        '{first_party}'

        Second Party:
        '{second_party}'

        Predicted Outcome for the First Party:
        '{'Win' if prediction else 'Lose'}'

        Task:
        Based *only* on the information provided above (Case Facts, Issue Area, and Predicted Outcome), please provide a brief summary outlining 2-3 potential key factors or lines of reasoning that could support the predicted outcome. Focus on aspects within the case facts or the nature of the issue area that might logically lead to this prediction. Don't incorporate the first party or second party into the reason leadin to this prediction, but you may state their name on the summary.

        Please structure your response as follows:
        Potential supporting factors for the prediction that the '[Restate Predicted Outcome for the First Party, e.g., First Party Wins]':
        1. [Factor 1]
        2. [Factor 2]
        3. [Factor 3 (optional)]


        Important Considerations for your analysis:
        - Your explanation should be based on logical inference from the provided text.
        - Do not introduce external knowledge or legal precedents not mentioned in the provided facts.
        - Use cautious and analytical language (e.g., 'might suggest,' 'could be due to,' 'a possible factor is').
        - The goal is to identify potential reasoning within the given context, not to provide definitive legal advice or a comprehensive case analysis.
    """

    logger.error(f"\niya")
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["API_KEY"]

    )

    completion = client.chat.completions.create(
    model="meta-llama/llama-3.3-8b-instruct:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    return completion.choices[0].message.content

if st.button("Clear Inputs"):
    st.session_state.first_party_text_input = ""
    st.session_state.second_party_text_input = ""
    st.session_state.issue_area_selectbox = ISSUE_AREA_OPTIONS[0]
    st.session_state.facts_text_input = ""

try:
    default_issue_area_index = ISSUE_AREA_OPTIONS.index("nan (unknown)")
except ValueError:
    default_issue_area_index = 0


st.header("Case Information")

col1, col2 = st.columns(2)
with col1:
    first_party_input = st.text_input(
        label="First party",
            placeholder="Example: John Doe",
            key="first_party_text_input",
            label_visibility="visible"
    )
with col2:
    second_party_input = st.text_input(
        "Second Party:",
        placeholder="Example: United States",
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
            processed_first_party = f"{first_party_input}" if first_party_input else ""
            processed_second_party = f"{second_party_input}" if second_party_input else ""
            processed_facts_input = f"{facts_input}" if facts_input else ""
            processed_issue_area_input = f"{issue_area_input}" if issue_area_input and issue_area_input != "nan (Unknown)" else ""

            combined_text_for_model = " ".join(filter(None, [
                processed_first_party,
                processed_second_party,
                processed_issue_area_input,
                processed_facts_input
            ]))

            data_to_be_predicted = pd.DataFrame({
                'tokenized_facts': [processed_facts(processed_facts_input)],
                'issue_area': [processed_issue_area_input]
            })

            if not combined_text_for_model.strip():
                st.warning("Combined input is empty. Please provide enough information for analysis.")
                st.stop()

            predicted_class = nlp_model.predict(data_to_be_predicted)[0]
            # predicted_class = prediction_raw[0]

            confidence_score = None
            if hasattr(nlp_model, 'predict_proba'):
                prediction_proba = nlp_model.predict_proba(data_to_be_predicted)
                confidence_score = prediction_proba[0][1]

            predicted_label = LABEL_MAP.get(predicted_class, "Unknown Label")

            st.markdown("## Prediction Result")
            st.markdown("---")

            st.subheader("Case Classification:")
            if predicted_label == "First Party Win":
                st.error(f"**{predicted_label}**")
                if confidence_score is not None:
                    st.write(f"Model Confidence (GUILTY): **{confidence_score:.2%}**")
            elif predicted_label == "Second Party Win":
                st.success(f"**{predicted_label}**")
                if confidence_score is not None:
                    st.write(f"Model Confidence (NOT GUILTY): **{1 - confidence_score:.2%}**")
            else:
                st.info(f"**{predicted_label}**")
                if confidence_score is not None:
                    st.write(f"Model Confidence: **{confidence_score:.2%}**")

            st.markdown("---")


            st.subheader("Reason for Model's Prediction:")
            # explanation_message, top_features_df = get_feature_importance_explanation(nlp_model, predicted_class)
            st.markdown(summarize_results(processed_facts_input, processed_issue_area_input, predicted_class, processed_first_party, processed_second_party))

        except Exception as e:
            st.error(f"An error occurred during prediction. Error: {e}")
            st.write("Ensure the input combination format matches your model's expectation and the model file is valid.")
            st.write(f"Error details: {e}")

  
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
        © 2025 Supreme Court Judgement Classification
    </div>
""", unsafe_allow_html=True)
