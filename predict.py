import joblib
vectorizer=joblib.load("models/vectorizer.pkl")
model=joblib.load("models/fake_review.pkl")
review=input("Enter a review: ")
review_vector=vectorizer.transform([review])
prediction=model.predict(review_vector)
print(prediction)
