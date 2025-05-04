import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from gtts import gTTS
import tempfile
import os
import langdetect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def speak_text(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        os.system(f"start {fp.name}")

# Streamlit UI
st.title("Real-Time Translation and Sentiment Analysis")

if st.button("Record & Translate"):
    user_text = record_and_transcribe()
    st.write("**You said:**", user_text)

    if "[Error" not in user_text:
        detected_lang = langdetect.detect(user_text)
        st.write("**Detected Language:**", detected_lang)

        sentiment = analyze_sentiment(user_text)
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
        sentiment = analyze_sentiment(text_input)
        st.write("**Sentiment:**", sentiment)
    else:
        st.warning("Please enter some text before analyzing.")
st.caption("This is a simple demonstration of ASR, language detection, sentiment analysis, translation, and TTS.")
