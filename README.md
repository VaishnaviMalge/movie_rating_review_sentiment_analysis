# Movie Review Sentiment Analysis

## Objective
The main motive of this project is to analyze movie review sentiments, classify them as positive and negative. It includes data preprocessing, feature extraction using NLP(Natural Language Processing), classification model for sentiment prediction and a UI for easy interaction with user

## Tech Stack
- Language: Python
- Libraries: NumPy, Pandas, Scikit-learn, NLTK, Matplotlib, Seaborn, Pickle, re, Streamlit, Wordcloud
- Environment: VS Code

## Project Workflow

### Data Collection
Dataset: IMDB movie review (kaggle dataset)

### Text Preprocessing
EDA, HTML tag removal, Special Character removal, lowercasing, Tokenization, Stopword removal, Stemming

### Feature Extraction
CountVectorizer: converted text to a numerical matrix to feed for model training

### Model Training 
Trained multiple models and analysed their performance using accuracy and precision to select appropriate model

### Evaluation 
Voting classifier used to improve overall performance

## Models Used

Logistic Regression, Naive Bayes (Gaussian, Bernoulli, Multinomial), K-Nearest Neighbors, Decision Tree, Random Forest, SVM, AdaBoost, Bagging, Extra Trees, Gradient Boosting, and XGBoost, SVC(Support Vector Classifier), Voting classifier (Ensemble Learning)


## How to run
- Download and open the files in vs code or any environment
- In vs code terminal install pip install wordcloud, xgboost, nltk==3.8.1 and other requirement (if not installed yet)
- Run the .ipynb file it will generate a pickle files for Multinomial model, voting classifier, and CountVectorizer
- Now open the .py file and save in same folder.(keep all file in same folder)
- In terminal run code: streamlit run movie_analysis.py  UI will open in a browser. 

## Improvement

### Code Optimization
Code can be optimized for faster work. Ex. First algorithms are assigned to a variable and then used that variable in the dictionary, instead they can directly be used as values of the dictionary.
Cons of the improvement: Increased complexity of code 
