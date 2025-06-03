# Final Project: Supreme Court Judgement Classification and Summarization

## Project Understanding
The project aims to assist legal institutions and lawtech platforms in classifying and summarizing U.S. Supreme Court judgements. Legal professionals often face challenges in reviewing large volumes of court documents. Automating the classification of judgement topics and summarizing lengthy facts can significantly improve efficiency and decision-making within legal practices.
### Problems
- Time-consuming legal document analysis, which reduces productivity in law firms and courts.
- Lack of automatic tagging or classification, making it difficult to retrieve related case laws efficiently.
- Insufficient tools for summarizing complex legal text, leading to cognitive overload among legal professionals.

### Project Scope
This project focuses on the following areas:
- Text preprocessing and feature engineering to clean and prepare the Supreme Court case dataset, including processing of case facts and encoding of issue areas.
- Model development for predicting the case outcome (first party winner) based on the factual description and legal issue area.
- Evaluation and comparison of multiple Machine Learning classifiers (e.g., Logistic Regression, KNN, Random Forest) using metrics such as accuracy, precision, and recall.
- Deployment interface using Streamlit.

### Preparation
Data source: [U.S. Supreme Court Decisions Dataset](https://github.com/dheovanwa/Supreme-court-judgement-classification/blob/f2008480d343bbc4bc882f5cdc125306919f8111/justice.csv)
- Key columns used: facts, issue_area, and first_party_winner.
Environtment Setup:
```
# To install packages that are necessary for the model to run
pip install -r requirements.txt

# To run the model via streamlit
streamlit run app.py
```

### Running the Machine Learning System
In order to run the Machine Learning prototype, you must follow these steps carefully:
1. Clone this repository
2. Install the required dependencies using ```pip install -r requirements.txt ```
3. Run the Streamlit app using ```streamlit run app.py```
4. The app will allow you to input the First Party, Second Party, Issue Area, and Case Facts
5. Press the "Classify" button to get the result of what you inputted

```
# Clone the repository
git clone <the-repository-url>

# Install the necessary packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
You can also access the deployed Streamlit application here: [Supreme Court Judgment Prediction](https://supreme-court-judgement-classification.streamlit.app/)

### Conclusion
The project succesfully developed a Machine Learning model to predict the Judgement of the Supreme Court with a quite reasonable accuracy.

# Machine Learning Flow
## Preprocessing
1. Converting the facts string to lower case
2. Removing unnecessary characters like html tags, whitespaces (whitespaces more than one will be changed to only one whitespace/space), and special characters
3. Tokenization <br>
   The initial step where raw text is converted into smaller, meaningful units called tokens (typically words, punctuation, or numbers). This process breaks down continuous text into discrete elements, making it understandable and processable for further analysis.
4. Lemmatization <br>
   Applied lemmatization to reduce words to their base or dictionary form. For example, "running", "ran", and "runs" are all reduces to "run". This helps standardize words and improves the model's understanding of semantic similarity.
5. Word Embedding <br>
   Each preprocessed token (word) is converted into a dense numerical vector. This vector representation captures the semantic relationships between words, words with similar meanings wil have similar vector representations.
6. One Hot Encoding <br>
   For categorical features like "First Party", "Second Party", or "Issue Area" as an alternative representation for words in some scenario. This converts categorical values into a binary vector, where each unique category becomes a distinct column.
7. Oversampling <br>
   We use RandomOverSampler oversampling technique. This involves generating synthetic samples for the minority class to balance the dataset, which helps prevent the model from becoming biased towards the majority class.
8. Dimensionality Reduction <br>
   To manage the complexity and improve efficiency, especially if our feature vectors are very large, we apply dimensionality reduction technique. This process reduces the number of features while trying to preserve as much of the important information as possible.

## Feature Selection
We select specific columns from the DataFrame that are relevant for the classification model. This helps us focus our analysis and prepare the data for subsequent preprocessing and modeling.
The selected columns are:
- ```facts```, The textual description of the case facts.
- ```issue_area```, The categorical classification of the legal issue involved.
- ```first_party_winner```, The column that represents the outcome we are trying to predict. It indicates whether the first party involved in the case won (```True```) or lost (```False```). This is the label
that our classification model will leartn to predict based on the input features.
## Exploratory Data Analysis (EDA)
To understand the distribution of the target variable (```first_party_winner```), bar plot is used to visualize the distribution of the data, we grouped the data by (```first_party_winner```) and calculated the size of each group. 
![image](https://github.com/user-attachments/assets/7d20115c-7151-4115-ba77-79584237fa59)

This was then visualized using a bar plot for easy comparison. This plot helps us understand the distribution in the target variable, which is important for model training.
