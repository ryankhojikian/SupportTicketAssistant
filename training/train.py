import sys
from pathlib import Path

import pandas as pd
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import time
import joblib
import os

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from be.config import URGENT_PATTERNS

# Load your data (update path as needed)
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_amazonhelp.csv')
df_tickets = pd.read_csv(DATA_PATH)
df_tickets = df_tickets[df_tickets['text'].notna() & (df_tickets['text'].astype(str).str.strip() != '')].reset_index(drop=True)

def clean_text(text):
    # Remove @mentions (e.g., @AmazonHelp)
    text = re.sub(r'@\w+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Clean whitespace
    text = ' '.join(text.split())
    return text
df_tickets['original_text'] = df_tickets['text']
df_tickets['text'] = df_tickets['original_text'].apply(clean_text)

# 1. Pre-calculate all component features for transparency (keywords from .env via be.config)
df_tickets['sentiment_polarity'] = df_tickets['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df_tickets['has_all_caps_word'] = df_tickets['text'].apply(lambda x: 1 if any(w.isupper() for w in str(x).split() if len(w) > 3) else 0)
df_tickets['excl_count'] = df_tickets['text'].str.count('!')
df_tickets['question_count'] = df_tickets['text'].str.count('\\?')
df_tickets['is_long_text'] = (df_tickets['text'].str.len() > 20).astype(int)
df_tickets['text_len'] = df_tickets['text'].str.len()
df_tickets['word_count'] = df_tickets['text'].apply(lambda x: len(str(x).split()))
df_tickets['urgent_keyword_count'] = df_tickets['text'].apply(
    lambda x: sum(1 for word in URGENT_PATTERNS if word.lower() in str(x).lower())
)

def labeling_function(row):
    sentiment = row['sentiment_polarity']
    has_caps = row['has_all_caps_word']
    excl = row['excl_count']
    ques = row['question_count']
    is_long = row['is_long_text']
    keywords = row['urgent_keyword_count']

    # Logic: Start with keyword weight (2 points per keyword found)
    score = keywords * 2

    # Use the pre-calculated features
    score += 1 if (excl > 1 or ques > 1) else 0
    score += 1 if has_caps > 1 else 0
    score += 1 if is_long == 1 else 0
    score += 1 if sentiment < -0.3 else 0

    return 1 if score >= 3 else 0

# 2. Apply Labeling Function
df_tickets['priority'] = df_tickets.apply(labeling_function, axis=1)

print(f"Percentage of Urgent Tickets: {df_tickets['priority'].mean()*100:.2f}%")


df_tickets['negativity_sentiment_polarity'] = df_tickets['sentiment_polarity'].apply(lambda x: -x)

# 1. Text Features (TF-IDF) - Reduced to 500 most important words
# min_df=5 ignores words that appear in fewer than 5 tickets
tfidf = TfidfVectorizer(max_features=1000, stop_words='english', min_df=5)
X_text = tfidf.fit_transform(df_tickets['text'])

# 2. Metadata Features (Excluding 'priority' from features since it is our target label)
scaler = StandardScaler()
# Note: We removed 'priority' from this list to prevent data leakage!
meta_cols = ['negativity_sentiment_polarity', 'word_count', 'has_all_caps_word', 'excl_count', 'urgent_keyword_count']
X_meta = df_tickets[meta_cols].values
X_meta_scaled = scaler.fit_transform(X_meta)

# 3. Combine them
# X_text.toarray() is fine for 500 features; for 100k it would have crashed memory
X_combined = np.hstack((X_text.toarray(), X_meta_scaled))
y = df_tickets['priority'].values

print(f"New Optimized Feature matrix shape: {X_combined.shape}")
print(f"We are now using {X_text.shape[1]} text patterns and {X_meta_scaled.shape[1]} metadata signals.")

# Debugging: Print shapes just before split to ensure full dataset is being used
print(f"Shape of X_combined before split: {X_combined.shape}")
print(f"Length of y before split: {len(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# LABEL-LEAKAGE ACKNOWLEDGEMENT
# ─────────────────────────────────────────────────────────────────────────────
# urgent_keyword_count is computed from URGENT_PATTERNS and is also the dominant
# driver of the labeling function (keywords * 2 inside labeling_function).
# Any model trained with urgent_keyword_count as a feature will partly reproduce
# the labeling regex rather than learning genuine urgency from text.
# High accuracy numbers therefore reflect this overlap, not pure generalisation.
# The "no-keyword" variant below trains RF without urgent_keyword_count so you
# can see how much of the signal is truly learned vs regex-reproduced.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("LABEL-LEAKAGE NOTE")
print("="*70)
print(
    "urgent_keyword_count is used both to create labels (keywords * 2 in the\n"
    "labeling function) AND as a model feature. The classifier can score very\n"
    "high simply by reproducing the regex — not by learning urgency.\n"
    "Accuracy numbers reflect that. See the no-keyword variant below for a\n"
    "fairer picture of what TF-IDF + soft signals can do on their own."
)
print("="*70 + "\n")

# ── VARIANT A: full feature set (includes urgent_keyword_count) ──────────────
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model, vectorizer, and scaler for later use (production model)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf_model, os.path.join(MODEL_DIR, 'random_forest_model.joblib'))
joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

print("\n--- Variant A: RF with urgent_keyword_count (FULL feature set) ---")
start_time = time.time()
rf_preds = rf_model.predict(X_test)
end_time = time.time()
latency_a = (end_time - start_time) / len(X_test) * 1000
acc_a = accuracy_score(y_test, rf_preds)
report_a = classification_report(y_test, rf_preds)
print(f"Accuracy: {acc_a:.4f}")
print(f"Latency (ms per ticket): {latency_a:.4f}")
print("Classification Report:\n" + report_a)

# ── VARIANT B: drop urgent_keyword_count — TF-IDF + soft signals only ────────
# This variant isolates how much the model genuinely learns from language vs
# reproducing the keyword-based labeling rule.
meta_cols_no_kw = ['negativity_sentiment_polarity', 'word_count', 'has_all_caps_word', 'excl_count']
scaler_no_kw = StandardScaler()
X_meta_no_kw = scaler_no_kw.fit_transform(df_tickets[meta_cols_no_kw].values)
X_combined_no_kw = np.hstack((X_text.toarray(), X_meta_no_kw))

X_train_nk, X_test_nk, y_train_nk, y_test_nk = train_test_split(
    X_combined_no_kw, y, test_size=0.2, random_state=42, stratify=y
)

rf_no_kw = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_no_kw.fit(X_train_nk, y_train_nk)

print("\n--- Variant B: RF WITHOUT urgent_keyword_count (TF-IDF + soft signals only) ---")
print("    [This is the honest baseline — the model cannot reproduce the labeling regex]")
start_time = time.time()
rf_preds_nk = rf_no_kw.predict(X_test_nk)
end_time = time.time()
latency_b = (end_time - start_time) / len(X_test_nk) * 1000
acc_b = accuracy_score(y_test_nk, rf_preds_nk)
report_b = classification_report(y_test_nk, rf_preds_nk)
print(f"Accuracy: {acc_b:.4f}")
print(f"Latency (ms per ticket): {latency_b:.4f}")
print("Classification Report:\n" + report_b)

# ── Side-by-side summary ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("VARIANT COMPARISON SUMMARY")
print("="*70)
print(f"  Variant A (with urgent_keyword_count):    accuracy = {acc_a:.4f}")
print(f"  Variant B (without urgent_keyword_count): accuracy = {acc_b:.4f}")
gap = acc_a - acc_b
print(f"  Accuracy gap (A − B): {gap:+.4f}")
if gap > 0.05:
    print(
        "  → Large gap: the keyword feature is carrying significant weight.\n"
        "    Part of the accuracy is the model reproducing the labeling regex."
    )
elif gap > 0.01:
    print(
        "  → Moderate gap: the keyword feature helps but TF-IDF + soft signals\n"
        "    still capture meaningful urgency signal on their own."
    )
else:
    print(
        "  → Small gap: TF-IDF + soft signals account for most of the accuracy.\n"
        "    The model is learning genuine urgency patterns beyond the regex."
    )
print("="*70)

