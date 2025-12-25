import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
# Load dataset
def load_data():
    df1 = pd.read_csv("./spam.csv", usecols=["label", "message"])
    df1["label"] = df1["label"].map({'ham': 0, 'spam': 1})
    df1 = df1[["message", "label"]].dropna()

    return df1

# Simple preprocessing (lowercase only)
def preprocess(text):
    return text.lower().strip()

df = load_data()
df['message'] = df['message'].apply(preprocess)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['message']).toarray()
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define base models
nb = MultinomialNB()
lr = LogisticRegression(max_iter=500)
dt = DecisionTreeClassifier(max_depth=25)

# Ensemble model using hard voting
ensemble = VotingClassifier(
    estimators=[
        ('nb', nb),
        ('lr', lr),
        ('dt', dt),
    ],
    voting='hard'
)

time_start = time.time()

# Train the ensemble
ensemble.fit(X_train, y_train)

elapsed = time.time() - time_start
print(f"Time taken to fit the model: {elapsed} seconds")

time_start = time.time()
ensemble_predictions = ensemble.predict(X_test)

elapsed = time.time() - time_start
print(f"Time taken to predict: {elapsed} seconds")

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
    label = 'spam' if pred == 1 else 'ham'

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)