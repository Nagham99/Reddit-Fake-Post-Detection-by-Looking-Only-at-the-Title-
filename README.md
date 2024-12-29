# Data Mining course at Queen’s University Master’s program.
comptition link : https://www.kaggle.com/competitions/cisc-873-dm-w24-a3
# Reddit-Fake-Post-Detection-by-Looking-Only-at-the-Title-
This project aims to predict if a Reddit post is fake news or not by analyzing its title. The dataset is raw and contains various forms of words, requiring effective text preprocessing techniques to handle. The goal is to develop a model for detecting fake posts using only title information.

### Problem Formulation:

#### Input:
- The input data is a tabular dataset.It consists of Text column: This column contains the text data representing the titles of Reddit posts.

#### Output:
- The output is a binary classification Probability (0 to 1) indicating whether the Reddit post is fake news or not.

#### Data Mining Function:
- Classification algorithm to predict the binary label (fake or not fake) based on the input title.

#### Challenges:
| Challenge                    | Description                                                                                                                                               |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ambiguous Labeling           | The presence of "label 2" with unclear meaning necessitates investigation to determine its relevance or potential removal, consuming analysis time.     |
| Data Structure Inconsistencies | Identifying irregularities such as variations in data formats (tabs vs. spaces) requires thorough examination and correction, prolonging preprocessing time. |
| Text Data Cleaning           | Discovery of noise elements like URLs, timestamps, and special characters demands meticulous cleaning efforts to maintain data integrity, increasing cleaning time. |
| Duplicate Detection          | Detection of duplicate entries within the dataset demands careful scrutiny and removal to ensure data quality, consuming additional processing time.     |
| Optimizing Hyperparameters   | Experimenting with hyperparameters to optimize model performance and generalization involves iterative testing and tuning, leading to extended optimization time. |


#### Preprocessing Steps:

| Step                                    | Description                                                                                                                                                                     |
|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Convert text to lowercase              | All text is converted to lowercase letters to ensure uniformity in the text data.                                                                                               |
| Remove HTML tags                       | Any HTML tags present in the text are removed using a regular expression to eliminate irrelevant markup.                                                                         |
| Replace special characters with spaces | Special characters, such as punctuation marks and symbols, are replaced with spaces to simplify the text and remove noise.                                                      |
| Tokenize the text                      | The text is tokenized into individual words using the `word_tokenize` function from the NLTK library, splitting it into a list of tokens.                                      |
| Remove stopwords and single-letter words | Stopwords, which are common words like "the," "is," and "and" that often do not carry significant meaning, are removed from the tokenized text.                                |
| Lemmatization                         | Each word in the tokenized text is lemmatized using the WordNetLemmatizer from NLTK. Lemmatization reduces words to their base or dictionary form, aiding in standardization. |
| Snowball stemming                     | Snowball stemming is applied to further reduce words to their root or stem form. This aggressive technique simplifies the vocabulary by removing suffixes and prefixes.       |
| Join tokens back into a single string | Finally, the preprocessed tokens are joined back together into a single string, with spaces separating the words. This processed text is returned by the function.           |

#### Experimental Protocol

1. **Data Loading**: Load the training and test datasets into the environment for analysis and model building. Ensure that the data contains features (title text) and labels (fake or not) for classification.

2. **Data Cleaning**: Perform initial data exploration and identify any missing values or duplicates

3. **Preprocessing**: Apply preprocessing steps to  preprocess the text data by converting to lowercase, removing HTML tags, special characters, and stopwords, and applying stemming or lemmatization.

4. **Model Selection**: Experiment with various classification models such as Logistic Regression, Random Forest, Gradient Boosting, XGBoost. Choose models that have shown promising results in previous experiments, such as those mentioned in the ideal solution.

5. **Hyperparameter Tuning**: Utilize different hyperparameter tuning techniques including Grid Search, Random Search, and to optimize model performance. Tune hyperparameters for each selected model to achieve the best possible performance.

6. **Model Evaluation**: Evaluate each model's performance using AUROC as the metric on the predicted probability. Compare the AUROC scores of different models to select the best-performing one.

7. **Submission Preparation**: Prepare the submission file using the best-performing model to predict the probability of a fake post in the test dataset. Save the results in a CSV file and submit it to the competition platform.


#### Impact:
- The detection of fake news can help prevent the spread of misinformation and promote informed decision-making.
- Enhances trust and credibility in online information sources.
- Provides insights into the prevalence and characteristics of fake news on social media platforms.

#### Ideal Solution:
- Our ideal solution includes the top-performing models identified in our analysis: Random Forest with RandomizedSearchCV, Logistic Regression with Stemming and CountVectorizer, and SVM with TfidfVectorizer.

1. Random Forest with RandomizedSearchCV (Trail 7): Achieved the highest score on the Kaggle leaderboard, indicating its effectiveness in predicting fake news. With a ROC AUC score of 0.8562, this model provides a strong baseline. We would further optimize this model by expanding the hyperparameter search space to potentially improve performance.

2. Logistic Regression with Stemming and CountVectorizer (Trail 8): Also performed well with a ROC AUC score of 0.8854, demonstrating effectiveness in predicting fake news. We would continue to fine-tune hyperparameters to maximize performance.

3. SVM with TfidfVectorizer (Trail 13): Showed promising results with a ROC AUC score of 0.8847. We would further optimize this model by fine-tuning hyperparameters and exploring ensemble methods for improved accuracy.


### Model Tuning and Documentation:
| Trail | Model | Reason | Expected Outcome | Observations |
|-------|-------|--------|------------------|--------------|
| 0 | Naive Bayes with Lemmatization | Baseline model for text classification | Moderate AUROC Score | Achieved AUROC Score: 0.7926 |
| 1 | Naive Bayes with Stemming | Evaluate the effect of stemming on model performance | Improved AUROC Score compared to Lemmatization | Achieved AUROC Score: 0.7985 |
| 2 | Naive Bayes with Stemming and Word-Level TF-IDF | Assess the impact of different vectorization techniques | Slightly improved AUROC Score compared to Trail 1 | Achieved AUROC Score: Slightly higher |
| 3 | Naive Bayes with Stemming and Character-Level TF-IDF | Explore character-level vectorization | Lower AUROC Score compared to word-level TF-IDF | Achieved AUROC Score: Lower |
| 4 | XGBoost with Stemming and Word-Level TF-IDF | Apply a different model with word-level TF-IDF | Improved AUROC Score compared to Naive Bayes | Achieved AUROC Score: 0.7411 |
| 5 | Logistic Regression with Stemming and Word-Level TF-IDF | Compare with XGBoost | Higher AUROC Score compared to XGBoost | Achieved AUROC Score: 0.8021 |
| 6 | Random Forest with Stemming and CountVectorizer | Evaluate ensemble learning | Higher AUROC Score compared to Logistic Regression | Achieved AUROC Score: 0.8563 |
| 7 | Random Forest with Stemming and TfidfVectorizer (RandomizedSearchCV) | Tune hyperparameters for Random Forest | Higher AUROC Score compared to Trail 6 | Achieved AUROC Score: 0.8562 |
| 8 | Logistic Regression with Stemming and CountVectorizer (RandomizedSearchCV) | Tune hyperparameters for Logistic Regression | Higher AUROC Score compared to Trail 7 | Achieved AUROC Score: 0.8854 |
| 9 | LSTM with Stemming and CountVectorizer | Experiment with deep learning approach | Poor AUROC Score indicating random predictions | Achieved AUROC Score: 0.5 (Poor performance) |
| 10 | GradientBoostingClassifier with CountVectorizer (Grid Search) | Explore gradient boosting with grid search | Higher AUROC Score compared to Random Forest | Achieved AUROC Score: 0.8588 |
| 11 | XGBoost with TfidfVectorizer (RandomizedSearchCV) | Experiment with different vectorization and model | Higher AUROC Score compared to Trail 4 | Achieved AUROC Score: 0.8517 |
| 12 | LightGBM with TfidfVectorizer | Experiment with LightGBM classifier | Similar or higher AUROC Score compared to XGBoost | Achieved AUROC Score: 0.8585 |
| 13 | SVM with TfidfVectorizer | Experiment with Support Vector Machine classifier | Higher AUROC Score compared to other models | Achieved AUROC Score: 0.8847 |

### Model Evaluation and Analysis
| Trail | Model | ROC AUC Score | Best Parameters | Observation |
|------|-------|---------------|-----------------|-------------|
| 0 | Naive Bayes with Lemmatization | 0.7926 | - | Baseline model with Lemmatization. Moderate AUROC score. |
| 1 | Naive Bayes with Stemming | 0.7985 | - | Improved AUROC score compared to Lemmatization. Stemming performed better. |
| 2 | Naive Bayes with Stemming and Word-Level TF-IDF | Slightly higher | - | Word-Level TF-IDF vectorization slightly improved AUROC score compared to Trail 1. |
| 3 | Naive Bayes with Stemming and Character-Level TF-IDF | Lower | - | Character-Level TF-IDF vectorization resulted in lower AUROC score compared to word-level TF-IDF. |
| 4 | XGBoost with Stemming and Word-Level TF-IDF | 0.7411 | - | XGBoost model performed decently with word-level TF-IDF vectorization, but lower compared to other models. |
| 5 | Logistic Regression with Stemming and Word-Level TF-IDF | 0.8021 | - | Logistic Regression outperformed XGBoost with word-level TF-IDF. |
| 6 | Random Forest with Stemming and CountVectorizer | 0.8563 | {'vectorizer__ngram_range': (1, 2), 'vectorizer__max_features': 10000, 'classifier__n_estimators': 200, 'classifier__max_depth': None} | Ensemble method with random search performed well, but slightly lower than other models. Stemming and CountVectorizer preprocessing provided good feature representation. |
| 7 | Random Forest with Stemming and TfidfVectorizer (RandomizedSearchCV) | 0.8562 | {'vectorizer__ngram_range': (1, 2), 'classifier__n_estimators': 300, 'classifier__max_depth': None} | RandomizedSearchCV improved the AUROC score compared to Trail 6. |
| 8 | Logistic Regression with Stemming and CountVectorizer (RandomizedSearchCV) | 0.8854 | {'classifier__C': 0.7809054218843381} | Achieved the highest AUROC score among all models. Stemming and CountVectorizer preprocessing yielded optimal results. |
| 9 | LSTM with Stemming and CountVectorizer | 0.5 | - | Poor performance indicated by AUROC score of 0.5, suggesting random predictions. |
| 10 | GradientBoostingClassifier with CountVectorizer (Grid Search) | 0.8588 | {'classifier__learning_rate': 0.2, 'classifier__max_depth': 5, 'classifier__n_estimators': 300, 'vectorizer__max_features': 25000, 'vectorizer__ngram_range': (1, 2)} | Ensemble method with grid search achieved competitive performance. CountVectorizer with gradient boosting provided effective feature representation. |
| 11 | XGBoost with TfidfVectorizer (RandomizedSearchCV) | 0.8517 | {'vectorizer__ngram_range': (1, 2), 'vectorizer__max_features': 5000, 'classifier__n_estimators': 200, 'classifier__max_depth': 5} | XGBoost showed competitive performance, but slightly lower than other models. TfidfVectorizer effectively represented features for the model. |
| 12 | LightGBM with TfidfVectorizer | 0.8585 | - | LightGBM showed comparable performance to GradientBoostingClassifier. TfidfVectorizer effectively represented features for the model. |
| 13 | SVM with TfidfVectorizer | 0.8847 | - | Close second to Logistic Regression in terms of AUROC score. TfidfVectorizer performed well in capturing important features. |
| 14 | SVM Classifier with TfidfVectorizer | 0.8847 | - | Achieved the highest score in Kaggle submission, validating the model's effectiveness. |

These trails involve a variety of preprocessing techniques, vectorization methods, and machine learning models to explore their impact on predicting fake news in Reddit posts. The observations provide insights into the effectiveness of different approaches and guide the selection of the best-performing model for the task

**Conclusion:**
we explored various preprocessing techniques, vectorization methods, and machine learning models to predict fake news in Reddit posts based on their titles. Through 14 different trails, we evaluated the performance of different combinations of preprocessing and modeling techniques.

Our analysis revealed several key findings:

- **Preprocessing Techniques**: Stemming and Lemmatization were both effective in improving model performance compared to the raw text. Lemmatization slightly outperformed stemming in terms of AUROC score.

- **Vectorization Methods**: TF-IDF vectorization showed better performance compared to CountVectorizer, especially when combined with advanced models like XGBoost and LightGBM.

- **Model Performance**: Logistic Regression with Stemming and CountVectorizer achieved the highest AUROC score of 0.8854, followed closely by SVM with TfidfVectorizer (AUROC: 0.8847). These models demonstrated robust performance in distinguishing between fake and non-fake Reddit posts.

- **Advanced Models**: GradientBoostingClassifier and LightGBM also performed well with AUROC scores of 0.8588 and 0.8585, respectively. These models showcased the effectiveness of ensemble learning techniques in this classification task.

- Additionally, our Random Forest model with RandomizedSearchCV achieved the highest score in Kaggle submission, further validating the effectiveness of the approach.

