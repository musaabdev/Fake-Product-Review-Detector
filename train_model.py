import joblib
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
df=pd.read_csv("dataset/fake reviews dataset.csv")
print(df.columns)
print("First 5 rows")
print(df.head())
print("\nColumns:")
print(df.columns)

print("\nShape")
print(df.shape)

print("\nLabel Counts:")
print(df["label"].value_counts())

print("n\Missing Values:")
print(df.isnull().sum())

X=df["text_"]
y=df["label"]
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
      random_state=42
)
vectorizer=TfidfVectorizer()
X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)
joblib.dump(model,"models/fake_review_model.pkl")
joblib.dump(vectorizer,"models/vectorizer.pkl")
print("Model saved successfully")