import streamlit as st
import joblib
vectorizer=joblib.load("models/vectorizer.pkl")
model=joblib.load("models/fake_review_model.pkl")
st.set_page_config(
    page_title="Fake Product Review Detector",
    page_icon="🤖",
    layout="centered"
)
# ===================== SIDEBAR =====================
st.sidebar.title("🤖 Model Dashboard")
st.sidebar.write("**Algorithm:** Logistic Regression")
st.sidebar.write("**Vectorizer:** TF-IDF")
st.sidebar.write("**Accuracy:** 89.92%")
st.sidebar.divider()
st.sidebar.caption("👨‍💻 Developed by")
st.sidebar.write("**Mohammed Musaab**")
st.title("🤖 Fake Product Review Detector")
st.caption("AI-Powered Machine Learning System")
st.divider()
review=st.text_area(
    "Paste or type a product review below:",
    height=180,
    placeholder="Example: This phone has an excellent battery life and a great camera..."
)
if st.button ("🚀 Analyze Review", use_container_width=True):

 if  review.strip()=="":
    st.warning("⚠️Please Enter a Review first.")
 else:
    review_vector=vectorizer.transform([review])
    prediction=model.predict(review_vector)
    probability=model.predict_proba(review_vector)
    confidence=max(probability[0])*100
    probability=probability[0]
    fake_prob=0
    genuine_prob=0
    for label,prob in zip(model.classes_,probability):
        if label =="CG":
            fake_prob=prob*100
        else:
            genuine_prob=prob*100
    st.divider()
    st.subheader("📊 Analysis Result")
    col1,col2=st.columns(2)
    with col1:
        if prediction[0] == "CG":
            st.error("❌ Fake Review Detected")
        else:
            st.success("✅ Genuine Review")

    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")

    st.progress(confidence / 100)

    if confidence >= 90:
        st.success("🎯 High Confidence Prediction")
    elif confidence >= 70:
        st.info("👍 Moderate Confidence Prediction")
    else:
        st.warning("⚠️ Low Confidence Prediction")
    st.divider()
    st.subheader("📈 Probability Breakdown")

    st.write(f"❌ **Fake Review:** {fake_prob:.2f}%")
    st.progress(fake_prob/100)
    st.write(f"✅ **Genuine Review:** {genuine_prob:.2f}%")
    st.progress(genuine_prob/100)
st.divider()
st.caption("🚀 Built using Streamlit • Python • Scikit-Learn • TF-IDF")
