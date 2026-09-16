import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

# Encode labels
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42
)

# TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate
preds = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Test predictions
sample = [
    "Congratulations! You've won a free cruise. Call now!",
    "Hey Joseph, are we still meeting tomorrow?"
]

sample_tfidf = tfidf.transform(sample)
print("Sample predictions:", model.predict(sample_tfidf))

