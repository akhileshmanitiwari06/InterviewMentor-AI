# =====================================
# 1. Install Dependencies
# =====================================
# pip install streamlit speechrecognition pyttsx3 transformers sentence-transformers librosa scikit-learn matplotlib pandas sounddevice

import streamlit as st
import speech_recognition as sr
import pyttsx3
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import pandas as pd

# =====================================
# 2. Initialize Models
# =====================================
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
engine = pyttsx3.init()

ideal_answers = {
    "Tell me about yourself.": "I am a computer science student skilled in Python, ML, and data analysis. I have worked on projects like Vocal AI and Caption Crafter.",
    "What are your strengths and weaknesses?": "My strengths include problem solving and quick learning. My weakness is I sometimes over-focus on details.",
    "Why should we hire you?": "Because I have the technical skills, project experience, and adaptability required for this role.",
    "Explain polymorphism in Python.": "Polymorphism allows functions or methods to process objects differently depending on their type or class.",
    "What is overfitting in Machine Learning?": "Overfitting happens when a model learns the training data too well but performs poorly on new data."
}
questions = list(ideal_answers.keys())
results = []

# =====================================
# 3. Core Functions
# =====================================
def listen_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return "Sorry, could not recognize your voice."

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def evaluate_answer(question, user_answer):
    if question not in ideal_answers:
        return 0.0
    emb1 = similarity_model.encode(user_answer, convert_to_tensor=True)
    emb2 = similarity_model.encode(ideal_answers[question], convert_to_tensor=True)
    return round(util.cos_sim(emb1, emb2).item(), 2)

def check_filler_words(text):
    fillers = ["um", "uh", "like", "you know", "actually"]
    return sum(word in fillers for word in text.lower().split())

# =====================================
# 4. Streamlit UI
# =====================================
st.set_page_config(page_title="AI Interview Coach", layout="wide")
st.title("🎤 AI Interview Coach")
st.write("Practice interviews with real-time feedback on **content, clarity, and filler words**.")

if "current_q" not in st.session_state:
    st.session_state.current_q = 0
    st.session_state.results = []

if st.session_state.current_q < len(questions):
    q = questions[st.session_state.current_q]
    st.subheader(f"🤖 Interviewer: {q}")
    
    if st.button("🎙️ Answer with Voice"):
        answer = listen_voice()
        st.write(f"📝 Your Answer: {answer}")

        score = evaluate_answer(q, answer)
        fillers = check_filler_words(answer)

        feedback = "✅ Good job!" if score > 0.6 else "⚠️ Add more details."
        if fillers > 2:
            feedback += " Reduce filler words."
        
        st.success(f"Content Score: {score*100:.1f}%")
        st.warning(f"Filler Words: {fillers}")
        st.info(f"💡 Feedback: {feedback}")

        speak_text(feedback)

        # Save result
        st.session_state.results.append({
            "question": q,
            "answer": answer,
            "score": score*100,
            "fillers": fillers
        })
        st.session_state.current_q += 1

else:
    st.subheader("📊 Final Interview Summary")
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df)

    avg_score = df['score'].mean()
    avg_fillers = df['fillers'].mean()
    st.metric("Average Score", f"{avg_score:.1f}%")
    st.metric("Avg. Filler Words", f"{avg_fillers:.1f} per answer")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df.set_index("question")["score"])
    with col2:
        st.bar_chart(df.set_index("question")["fillers"])

    # Overall Rating
    if avg_score > 80 and avg_fillers < 2:
        rating = "🌟 Excellent"
    elif avg_score > 60:
        rating = "👍 Good"
    else:
        rating = "📝 Needs Improvement"

    st.subheader(f"Overall Rating: {rating}")
