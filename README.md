# 📰 Fake News Detector

This is a small project I built to check whether a news headline is likely real or fake using machine learning + some real-time verification.

The idea is simple: instead of just trusting a model, I also try to match the input with current news articles to make the result more reliable.

---
## 🌐 Live Demo
👉 https://fake-news-detector46.streamlit.app

---

## 🎥 Demo
![Demo](assets/demo.gif)

---

## 📸 Screenshots

### 🏠 Home Screen
![Home](assets/screenshots/home.png)

### 📊 Analysis Result
![Result](assets/screenshots/result.png)

### 📰 News Verification
![News](assets/screenshots/news.png)

## 🚀 What it does

* Takes a news headline or short text
* Predicts if it’s fake or real using an ML model
* Shows confidence score
* If unsure → fetches real news using NewsAPI
* Compares similarity with latest articles
* Gives a final verdict

---

## 🧠 How it works (basic idea)

* Clean the input text
* Convert it using a vectorizer
* Run it through a trained model
* If confidence is low, fetch news articles
* Use cosine similarity to compare
* Combine everything for final result

Not perfect, but works decently for most cases.

---

## 🛠️ Tech used

* Python
* Streamlit (for UI)
* scikit-learn
* requests
* some basic NLP

---

## 📂 Project structure

```
fake-news-detector/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── assets/
│   └── bg.png
```

---

## ⚙️ How to run

Clone the repo:

```
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

Create virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Add your NewsAPI key:

Create:

```
.streamlit/secrets.toml
```

Inside it:

```
NEWS_API_KEY = "your_api_key"
```

Then run:

```
streamlit run app.py
```

---

## ⚠️ Notes

* You need `model.pkl` and `vectorizer.pkl` or it won’t run
* API has rate limits (free version especially)
* Results are not 100% accurate (depends on training data)

---

## 💡 Things I might improve later

* Better model (maybe deep learning)
* Cleaner UI
* More accurate verification logic
* Deploy it online

---

## 👨‍💻 About

Made this while learning ML + building real projects.
Still improving it.

---

That’s it :)
