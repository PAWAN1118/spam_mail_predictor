# spam_mail_predictor
1. Import Necessary Libraries
The script uses pandas for data handling, nltk for text preprocessing, sklearn for machine learning, and matplotlib/seaborn for visualization.

2. Load and Preprocess Data
Reads a CSV file (spam_dataset.csv) containing emails labeled as spam or ham (not spam).

Converts labels to numerical values (ham = 0, spam = 1).

Cleans email text:

Converts text to lowercase.

Removes special characters.

Splits text into words (tokens).

Removes stopwords (common words like "the", "and", etc.).

3. Feature Extraction (TF-IDF Vectorization)
The TfidfVectorizer converts email text into numerical form so that the machine learning model can process it.

4. Train the Naïve Bayes Model
Splits data into training (80%) and testing (20%) sets.

Uses Multinomial Naïve Bayes, which is great for text classification.

Trains the model on email data.

5. Make Predictions and Evaluate the Model
Predicts spam/ham for test data.

Calculates accuracy and displays a classification report.

6. Plot Confusion Matrix
Uses seaborn to visualize model performance in a confusion matrix.

Final Output
Accuracy Score (how well the model predicts spam/ham).

Classification Report (Precision, Recall, F1-score).

Confusion Matrix (visual representation of correct vs. incorrect predictions).
