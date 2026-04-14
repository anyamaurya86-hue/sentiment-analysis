import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

print("Loading dataset...")
df = pd.read_csv('./dataset/emobank.csv')
df = df.dropna(subset=['text']) 

X = df['text']
y = df[['V', 'A']] 

print("Vectorizing text...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

print("Training model...")
base_model = Ridge(alpha=1.0)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae_v = mean_absolute_error(y_test['V'], predictions[:, 0])
mae_a = mean_absolute_error(y_test['A'], predictions[:, 1])

print(f"\nValence MAE: {mae_v:.3f} (Lower is better)")
print(f"Arousal MAE: {mae_a:.3f} (Lower is better)")

# Save the new models
joblib.dump(model, './emotion_model.pkl')
joblib.dump(vectorizer, './tfidf_vectorizer.pkl')
print("\nModels saved successfully.")