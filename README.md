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
3. Fetch your own api key from openrouter and then make a folder named .streamlit
4. Create file named secrets.toml inside of .streamlit and paste the api key into a variable below
```
API_KEY = <YOUR API KEY>
```
5. Run the Streamlit app using ```streamlit run app.py```
6. The app will allow you to input the First Party, Second Party, Issue Area, and Case Facts
7. Press the "Classify" button to get the result of what you inputted

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

## Dataset Source
Our dataset source came from the Kaggle platform from this link below:
[Supreme Court Judgment Prediction
](https://www.kaggle.com/datasets/deepcontractor/supreme-court-judgment-prediction)

## Feature Selection
We select specific columns from the DataFrame that are relevant for the classification model. This helps us focus our analysis and prepare the data for subsequent preprocessing and modeling.
The selected columns are:
- ```facts```, The textual description of the case facts.
- ```issue_area```, The categorical classification of the legal issue involved.
- ```first_party_winner```, The column that represents the outcome we are trying to predict. It indicates whether the first party involved in the case won (```True```) or lost (```False```). This is the label
that our classification model will leartn to predict based on the input features.

## Exploratory Data Analysis (EDA)
1. To understand the distribution of the target variable (```first_party_winner```), bar plot is used to visualize the distribution of the data, we grouped  the data by (```first_party_winner```) and calculated the size of each group. 
![image](https://github.com/user-attachments/assets/7d20115c-7151-4115-ba77-79584237fa59)

This was then visualized using a bar plot for easy comparison. This plot helps us understand the distribution in the target variable, which is important for model training.
The bar for ```True``` is significantly longer, indicating a much higher number of cases. On the other hand, the bar for ```False``` is notably shorter, indicating a lower number of cases.
Implication of this distribution:
- Imbalanced Dataset, the most apparent observation is the significant class imbalance. The number of cases where ```first_party_winner``` is ```True``` is nearly double that of cases where ```first_party_winner``` is ```False```
- Potential Issues for the Machine Learning model, this class imbalance is a critical factor to consider in machine Learning. If left unaddressed, the classification model might become biased towards predicting the majority class (```True```) because the model will "learn" that predicting ```True``` more often leads to higher overall accuracy, even if it means neglecting or misclassifyng the minority class (```False```)

2. We analyzed the `facts` column, which contains the textual description of each case, to identify the most frequently occurring words. This visualization provides insight into the common terminology and themes present in the legal case descriptions.
  ![image](https://hackmd.io/_uploads/r1RnGETflx.png)
The plot above shows the top 10 most frequent words. As observed, terms such as 'court', 'district', 'state', 'appeals', 'federal', and 'supreme' are highly prevalent, indicating the strong judicial and governmental context of the dataset. This analysis helps us understand the vocabulary and nature of the textual data, informing subsequent text preprocessing and feature engineering steps.

3. To gain a qualitative and immediate understanding of the most prominent terms in the `facts` column, we generated a word cloud. In this visualization, the size of each word is proportional to its frequency within the text data.
![image](https://hackmd.io/_uploads/HJ3KXV6fle.png) <br>
The word cloud vividly highlights keywords such as 'court', 'district', 'appeal', 'case', 'filed', 'state', 'federal', and 'supreme'. These terms consistently appear large, reinforcing the judicial and legal nature of the dataset. This visual representation quickly provides insights into the core vocabulary and main subjects discussed across the various case facts.

4. To understand the characteristics of our textual data, we analyzed the distribution of text lengths (e.g., number of characters or words) within the facts column. This histogram, overlaid with a Kernel Density Estimate (KDE) curve, shows the frequency of different text lengths.
![image](https://hackmd.io/_uploads/rkEBN4aMlx.png)
The plot reveals that the majority of case facts have lengths concentrated around 1000 to 1200 units, with the distribution being right-skewed, meaning there are a few much longer texts that trail off to the right. This analysis is crucial for tasks such as defining appropriate sequence lengths for neural network models or understanding the general verbosity of the case descriptions.

5. A boxplot reveals the distribution of text lengths in the `facts` column. The box shows the IQR (around 700-1500 units), with a median near 1000-1100. Notably, the plot displays numerous outliers to the right, indicating the presence of exceptionally long case descriptions. These outliers are important to consider for subsequent text processing steps.
![image](https://hackmd.io/_uploads/HyP5NEpfgx.png)

6. This bar chart visualizes the distribution of different legal `issue_area` categories within the dataset. It helps us understand which types of cases are most prevalent.
![image](https://hackmd.io/_uploads/r1sA446flg.png)
The plot clearly shows that 'Criminal Procedure' is the most dominant issue area, with significantly higher occurrences (over 800 cases). Other prominent categories include 'Civil Rights' and 'Economic Activity'. Conversely, several issue areas like 'Interstate Relations', 'Private Action', and 'Miscellaneous' have very low frequencies. This analysis highlights the thematic composition of the dataset and reveals potential class imbalance if Issue Area were to be used as a classification target itself.

## Preprocessing
1. Handled the missing value of `issue_area` by filling it with 'Unknown' and dropping the `first party winner` then changing it to a boolean type 0 and 1
2. Converting the facts string to lower case
3. Removing unnecessary characters like html tags, whitespaces (whitespaces more than one will be changed to only one whitespace/space), and special characters
4. Tokenization <br>
   The initial step where raw text is converted into smaller, meaningful units called tokens (typically words, punctuation, or numbers). This process breaks down continuous text into discrete elements, making it understandable and processable for further analysis.
5. Lemmatization <br>
   Applied lemmatization to reduce words to their base or dictionary form. For example, "running", "ran", and "runs" are all reduces to "run". This helps standardize words and improves the model's understanding of semantic similarity.
6. Removal of Extraneous Characters
  Prior to generating the word cloud and other text-based features, the raw text data underwent a cleaning process. This involved the explicit removal of various extraneous or anomalous characters, such as isolated single letters like 'p' and 'n', as well as apostrophes ('). This crucial step ensures that our analyses and visualizations, including the word cloud, are not skewed by irrelevant artifacts and instead accurately represent the meaningful terms within the dataset.
7. Splitting dataset into train and test set <br>
    We splitted the dataset into train and test set with 80% went into train set and the rest of 20% went into test set. We also stratified the label class to ensure the proportion are the same for every split. 
8. Word Embedding <br>
   We used TF-IDF because it considered word importance in a text. Computationally, TF-IDF is generally faster and requires fewer resources than training complex neural network-based embeddings. This makes it effective for information retrieval, text summarization, and keyword extraction tasks. Furthermore, TF-IDF does not require massive training datasets like deep learning embeddings to produce useful word representations for certain tasks and treats each word independently.
9. One Hot Encoding <br>
   For categorical features like "Issue Area" as an alternative representation for words in some scenario. This converts categorical values into a binary vector, where each unique category becomes a distinct column.
10. Oversampling <br>
   Given the imbalance in the `first_party_winner` distribution, we address this by oversampling the minority class using SMOTE (Synthetic Minority Over-sampling Technique). SMOTE synthesizes new samples for the minority class (`False`) to balance the dataset, ensuring our model learns effectively from both outcomes and improves its classification performance.
11. Dimensionality Reduction <br>
   To manage the complexity and improve efficiency, especially if our feature vectors are very large, we apply dimensionality reduction technique. This process reduces the number of features while trying to preserve as much of the important information as possible. We used truncatedSVD because it works greatly on sparse matrix which resulted by TF-IDF.

## Modelling and Evaluation
We compared various models for the classification task and the results of each model can be seen below.


|       Model Name        | Accuracy | Precision | Recall  |
|-------------------------|----------|-----------|---------|
| Logistic Regression     | 59.27%   | **70.62%**| 64.02%  |
| KNN                     | 54.56%   | 65.39%    | 64.02%  |
| Support Vector Classifier| 59.42%   | 67.77%    | 71.73%  |
| Gaussian Naive Bayes    | **63.22%**| 69.21%    | **78.27%**|
| Decision Tree Classifier| 52.58%   | 63.81%    | 62.62%  |
| Random Forest Classifier| 59.27%   | 66.06%    | 76.87%  |

As can be seen from the table above, the model with the best average metrics is Gaussian Naive Bayes with the prediction results of each label class seen below.
```
              precision    recall  f1-score   support

       False       0.47      0.35      0.40       230
        True       0.69      0.78      0.73       428

    accuracy                           0.63       658
   macro avg       0.58      0.57      0.57       658
weighted avg       0.61      0.63      0.62       658
```
