# ===============================
# PHISHING EMAIL DETECTION MODEL
# ===============================

import pandas as pd
import re
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# ===============================
# 1. LOAD DATASET
# ===============================

# Replace with your dataset path
df = pd.read_csv(r"C:\Users\varsh\Downloads\phishing_email_detection\emails.csv")

# Expected columns: 'text', 'label'
print(df.head())

# Convert labels: safe=0, phishing=1
df['label'] = df['label'].map({'safe': 0, 'phishing': 1})

# ===============================
# 2. FEATURE ENGINEERING
# ===============================

def extract_url_features(text):
    urls = re.findall(r'(https?://\S+)', str(text))

    return [
        len(urls),  # URL count
        int(any(re.search(r'\d+\.\d+\.\d+\.\d+', url) for url in urls)),  # IP-based URL
        int(any(word in text.lower() for word in ['login', 'verify', 'bank', 'urgent', 'click']))  # suspicious words
    ]

# Apply URL features
url_features = df['text'].apply(extract_url_features)
url_df = pd.DataFrame(url_features.tolist(), columns=[
    'url_count', 'has_ip', 'suspicious_words'
])

# ===============================
# 3. TEXT FEATURES (TF-IDF)
# ===============================

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_text = tfidf.fit_transform(df['text'])

# Combine features
X = hstack([X_text, url_df.values])
y = df['label']

# ===============================
# 4. TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5. TRAIN MODEL
# ===============================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===============================
# 6. EVALUATION
# ===============================

y_pred = model.predict(X_test)

print("\n=== RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# 7. CONFUSION MATRIX PLOT
# ===============================

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

# ===============================
# 8. PREDICTION FUNCTION
# ===============================

def predict_email(email_text):
    url_feat = extract_url_features(email_text)
    url_feat_df = pd.DataFrame([url_feat])

    text_feat = tfidf.transform([email_text])
    combined = hstack([text_feat, url_feat_df.values])

    pred = model.predict(combined)[0]

    return "Phishing" if pred == 1 else "Safe"

# ===============================
# 9. TEST YOUR MODEL
# ===============================

test_email = "URGENT! Verify your bank account now at http://fakebank.com"

result = predict_email(test_email)
print("\nTest Email Prediction:", result)