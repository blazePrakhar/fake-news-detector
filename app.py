import streamlit as st
import pickle
import re
import requests
from sklearn.metrics.pairwise import cosine_similarity

 
# LOAD MODEL
 
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

 
# TEXT CLEANING
 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

 
# FETCH NEWS (WITH ERROR HANDLING)
 
def fetch_news(query):
    url = "https://newsapi.org/v2/everything"
    
    params = {
        "q": query,
        "apiKey": st.secrets["NEWS_API_KEY"],  #  API
        "pageSize": 5,
        "language": "en",
        "sortBy": "publishedAt"
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Handle API errors
    if data.get("status") != "ok":
        st.error(f"API Error: {data.get('message')}")
        return []

    return data.get("articles", [])

 
# PREDICTION
 
def predict_news(text):
    text = clean_text(text)
    text = text + " " + text  # match training pattern

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]

    return prediction, probability

 
# UI CONFIG
 
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection System")
st.markdown("Check whether a news statement is **likely real or fake** using AI + real-time verification.")

input_text = st.text_area(" Enter news text:")
st.caption("Tip: Use short, keyword-rich headlines for better results.")

 
# BUTTON ACTION
 
 
if st.button("Analyze"):
    if input_text.strip() == "":
        st.warning("Please enter some text")
    else:
        st.divider()

        # Short input warning
        if len(input_text.split()) < 8:
            st.warning("   Short factual statement detected. Results may be less reliable.")

        # Prediction
        result, proba = predict_news(input_text)
        confidence = max(proba)

        # Confidence UI
        st.subheader(" Model Confidence")
        st.progress(int(confidence * 100))
        st.write(f"{round(confidence * 100, 2)}%")

         
        # FETCH NEWS (CONDITIONAL)
         
        query = input_text

        if confidence < 0.8:
            articles = fetch_news(query)
        else:
            st.info("High confidence prediction – skipping external verification.")
            articles = []

         
        # DISPLAY NEWS
         
        st.subheader(" Related News")

        if len(articles) == 0:
            st.warning("   No relevant news found. Try simplifying your input.")
            max_similarity = 0
        else:
            news_texts = []

            for article in articles:
                st.markdown(f"**{article['title']}**")
                st.caption(article['source']['name'])
                st.write(article['url'])
                st.write("---")

                combined_text = article["title"] + " " + (article.get("description") or "")
                news_texts.append(clean_text(combined_text))

             
            # SIMILARITY CHECK
             
            input_vec = vectorizer.transform([clean_text(input_text)])
            news_vec = vectorizer.transform(news_texts)

            similarity_scores = cosine_similarity(input_vec, news_vec)[0]
            max_similarity = max(similarity_scores)

        st.subheader(" Similarity Score")
        st.write(f"{round(max_similarity * 100, 2)}% match with current news")

         
        # FINAL DECISION
         
        st.subheader(" Final Verdict")

        if max_similarity > 0.15:
            st.success("   Likely Real (matches real-world news)")

        elif confidence > 0.75:
            st.success("   Likely Real (based on strong language patterns)")

        elif max_similarity == 0:
            st.warning("   No supporting news found. Cannot verify this claim.")

        elif result == 0 and confidence > 0.7:
            st.error(" Likely Fake (based on language patterns)")

        else:
            st.warning("   Uncertain – cannot verify confidently")