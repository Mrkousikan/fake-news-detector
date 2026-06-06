import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load cleaned data
df = pd.read_csv('cleaned_data.csv')
df = df.dropna()

# Features and labels
X = df['cleaned']
y = df['label']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorization
print("⚙️ Applying TF-IDF...")
tfidf = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_accuracy = 0
best_name = ""

print("\n📊 Model Results:\n")
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc*100:.2f}% accuracy")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

print(f"\n🏆 Best Model: {best_name} ({best_accuracy*100:.2f}%)")

# Save best model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("\n✅ Model and vectorizer saved!")