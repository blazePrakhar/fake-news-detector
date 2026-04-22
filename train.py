import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

# Labels
fake["label"] = 0   # Fake
real["label"] = 1   # Real

# Combine dataset
df = pd.concat([fake, real], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title + text (IMPORTANT FIX)
df["content"] = df["title"] + " " + df["text"]

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Apply cleaning
df["content"] = df["content"].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["content"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))