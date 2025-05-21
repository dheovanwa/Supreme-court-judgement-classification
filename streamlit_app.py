import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to select a column
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.column_name]

@st.cache_data
def load_data(path='processed_facts.csv'):
    df = pd.read_csv(path)
    df = df.dropna(subset=['tokenized_facts', 'issue_area', 'first_party_winner'])
    df['first_party_winner'] = df['first_party_winner'].astype(bool)
    return df

@st.cache_resource
def build_pipeline(df):
    # Split data
    X = df[['tokenized_facts', 'issue_area']]
    y = df['first_party_winner']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=766, stratify=y)

    # Text pipeline
    text_pipeline = Pipeline([
        ('selector', ColumnSelector('tokenized_facts')),
        ('vectorizer', TfidfVectorizer(tokenizer=lambda x: x.split(), preprocessor=None)),
        ('svd', TruncatedSVD(n_components=200, random_state=766)),
    ])

    # Issue area pipeline
    cat_pipeline = Pipeline([
        ('selector', ColumnSelector('issue_area')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    # Combine features
    full_pipeline = FeatureUnion([
        ('text', text_pipeline),
        ('cat', cat_pipeline)
    ])

    # Final pipeline with classifier
    model_pipeline = Pipeline([
        ('features', full_pipeline),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=766))
    ])

    # Train
    model_pipeline.fit(X_train, y_train)
    return model_pipeline

# Load data and build model
df = load_data()
model = build_pipeline(df)

st.title("Supreme Court Judgment Outcome Prediction")

# User inputs
st.sidebar.header("Input Case Details")
facts_input = st.sidebar.text_area("Case Facts", "Enter the facts of the case here...")
issue_area_input = st.sidebar.selectbox(
    "Issue Area", sorted(df['issue_area'].unique())
)

if st.sidebar.button("Predict Outcome"):
    if not facts_input:
        st.error("Please enter case facts to predict.")
    else:
        # Preprocess input
        # Using same preprocessing: lowercase and remove special chars
        import re
        from bs4 import BeautifulSoup
        def clean_text(text):
            text = BeautifulSoup(text, "html.parser").get_text()
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"[^\w\s]", "", text)
            return text.strip().lower()
        clean = clean_text(facts_input)
        tokens = " ".join(clean.split())
        input_df = pd.DataFrame({
            'tokenized_facts': [tokens],
            'issue_area': [issue_area_input]
        })
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        label = "First Party Wins" if prediction else "First Party Loses"
        st.subheader("Prediction")
        st.write(f"**Outcome:** {label}")
        st.subheader("Probability")
        st.write(f"First Party Wins: {proba[1]:.2f}")
        st.write(f"First Party Loses: {proba[0]:.2f}")

st.markdown("---")
st.write("This app uses a Random Forest classifier trained on Supreme Court cases to predict whether the first party will win based on the facts and issue area.")
