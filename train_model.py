
import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/job_skills.csv")

X = data["skills"]
y = data["job_role"]

# Encode job roles
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.35,
    random_state=42,
    stratify=y_encoded
)

# Define three ML algorithms
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=10
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

os.makedirs("models", exist_ok=True)

results = {}

# Train and evaluate each model
for name, classifier in models.items():

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    joblib.dump(pipeline, f"models/{name.lower().replace(' ', '_')}.pkl")

    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

# Save label encoder
joblib.dump(encoder, "models/label_encoder.pkl")

# Select best model
best_model = max(results, key=results.get)

print("\nBest Algorithm:", best_model)
print("Accuracy:", round(results[best_model] * 100, 2), "%")

joblib.dump(best_model, "models/best_model_name.pkl")

print("\nAll models trained and saved successfully!")
