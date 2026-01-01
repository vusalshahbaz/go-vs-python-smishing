import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

# Load phishing email dataset
def load_phishing_email_data():
    df = pd.read_csv("../datasets/spam-phishing-emails.csv", usecols=["subject", "body", "label"])
    # Ignore subject column, use only body
    df = df[["body", "label"]].dropna()
    df["label"] = df["label"].astype(int)  # Convert label to int
    return df

# Simple preprocessing (lowercase only)
def preprocess(text):
    return text.lower().strip()

print("Loading phishing email dataset...")
df = load_phishing_email_data()
df['body'] = df['body'].apply(preprocess)

print(f"Total samples: {len(df)}")

# Convert text to TF-IDF features
print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['body']).toarray()
print(f"vectorizer.Transform() - total: {len(df)}, processed: {len(df)}")
y = df['label'].values

print(f"Features shape: {X.shape}")

# Note: We don't standardize here because MultinomialNB requires non-negative values
# TF-IDF values are already normalized and work well for all three models
# Split data (80% train, 20% test)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Define base models
nb = MultinomialNB()
lr = LogisticRegression(max_iter=500)
dt = DecisionTreeClassifier(max_depth=25)

# Ensemble model using hard voting
print("Initializing PhishingEmailDetector ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('nb', nb),
        ('lr', lr),
        ('dt', dt),
    ],
    voting='hard'
)

print("phishingEmailDetector.Fit() - total: {}, starting training...".format(len(X_train)))

time_start = time.time()

# Train the ensemble
print("phishingEmailDetector.Fit() - training LogisticRegression model...")
print("phishingEmailDetector.Fit() - training MultinomialNaiveBayes model...")
print("phishingEmailDetector.Fit() - training DecisionTreeClassifier model...")
ensemble.fit(X_train, y_train)

elapsed = time.time() - time_start
print(f"Time taken to fit the model: {elapsed} seconds")
print("phishingEmailDetector.Fit() - LogisticRegression model completed")
print("phishingEmailDetector.Fit() - MultinomialNaiveBayes model completed")
print("phishingEmailDetector.Fit() - DecisionTreeClassifier model completed")
print("phishingEmailDetector.Fit() - all models trained, processed: {}".format(len(X_train)))

time_start = time.time()
print("Predicting...")
ensemble_predictions = ensemble.predict(X_test)
total = len(X_test)
print(f"phishingEmailDetector.Predict() - total: {total}, processed: {total}")

elapsed = time.time() - time_start
print(f"Time taken to predict: {elapsed} seconds")

# Calculate metrics
accuracy = accuracy_score(y_test, ensemble_predictions)
precision = precision_score(y_test, ensemble_predictions)
recall = recall_score(y_test, ensemble_predictions)

# Macro-averaged metrics
macro_precision = precision_score(y_test, ensemble_predictions, average='macro')
macro_recall = recall_score(y_test, ensemble_predictions, average='macro')
macro_f1 = f1_score(y_test, ensemble_predictions, average='macro')

# Confusion matrix
cm = confusion_matrix(y_test, ensemble_predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Macro Precision: {macro_precision}")
print(f"Macro Recall: {macro_recall}")
print(f"Macro F1: {macro_f1}")
print("Confusion Matrix:")
print(f"  Actual\\Predicted   0      1")
print(f"  0                  {cm[0][0]}     {cm[0][1]}")
print(f"  1                  {cm[1][0]}     {cm[1][1]}")

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    message = message.lower().strip()

    X_input = vectorizer.transform([message]).toarray()
    pred = ensemble.predict(X_input)[0]
    label = 'phishing' if pred == 1 else 'legitimate'

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)

