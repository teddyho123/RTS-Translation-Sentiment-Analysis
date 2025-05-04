import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from gtts import gTTS
import tempfile
import os
import langdetect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup

# Load transformers pipeline for translation (English only for simplicity)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
summarizer = pipeline("summarization")
sentiment_analyzer = SentimentIntensityAnalyzer()

def record_and_transcribe():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now...")
        audio = recognizer.listen(source, timeout=5)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "[Error: Speech not recognized]"
        except sr.RequestError:
            return "[Error: Could not request results]"

def analyze_sentiment(text):
    score = sentiment_analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        sentiment = "Positive"
    elif score['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, score

def extract_sentiment_keywords(text, threshold=0.2):
    words = text.split()
    keyword_scores = []
    for word in words:
        score = sentiment_analyzer.polarity_scores(word)
        if abs(score['compound']) > threshold:
            keyword_scores.append((word, score['compound']))
    keyword_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return keyword_scores[:10]  # top 10 impactful words

def speak_text(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        os.system(f"start {fp.name}")

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return article_text[:3000]  # limit to 3000 characters for processing
    except Exception as e:
        return f"[Error fetching article: {e}]"

# Streamlit UI
st.title("Real-Time Translation and Sentiment Analysis")

if st.button("Record & Translate"):
    user_text = record_and_transcribe()
    st.write("**You said:**", user_text)

    if "[Error" not in user_text:
        detected_lang = langdetect.detect(user_text)
        st.write("**Detected Language:**", detected_lang)

        sentiment, _ = analyze_sentiment(user_text)
        st.write("**Sentiment:**", sentiment)

        if detected_lang != "en":
            translation = translator(user_text)[0]['translation_text']
            st.write("**Translated Text:**", translation)
        else:
            translation = user_text

        if st.button("Record Again"):
            st.rerun()

st.markdown("---")
st.subheader("Type Your Own Text for Sentiment Analysis")
text_input = st.text_area("Enter your text:")
if st.button("Analyze Sentiment"):
    if text_input.strip():
        sentiment, score = analyze_sentiment(text_input)
        st.write("**Sentiment:**", sentiment)
        st.write("**VADER Scores:**", score)

        keywords = extract_sentiment_keywords(text_input)
        if keywords:
            st.write("**Top Sentiment Keywords:**")
            for word, s in keywords:
                st.write(f"- {word}: {s:.2f}")
        else:
            st.write("No strong sentiment words detected.")
    else:
        st.warning("Please enter some text before analyzing.")

st.markdown("---")
st.subheader("Analyze Sentiment of a Web Article")
url_input = st.text_input("Paste a news/article URL:")
if st.button("Analyze Article"):
    if url_input.strip():
        article_text = extract_text_from_url(url_input)
        sentiment, _ = analyze_sentiment(article_text)
        keywords = extract_sentiment_keywords(article_text)

        st.write("**Sentiment of Article:**", sentiment)
        if keywords:
            st.write("**Top Sentiment Keywords:**")
            for word, score in keywords:
                st.write(f"- {word}: {score:.2f}")
        else:
            st.write("No strong sentiment words detected.")
    else:
        st.warning("Please paste a valid URL.")

st.caption("This is a simple demonstration of ASR, language detection, sentiment analysis, translation, and TTS.")