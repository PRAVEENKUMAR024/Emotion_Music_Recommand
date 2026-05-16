import streamlit as st
import numpy as np
from deepface import DeepFace
import urllib.parse
import webbrowser
import cv2

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Moodify",
    page_icon="🎵",
    layout="centered"
)

# -----------------------------------
# TITLE
# -----------------------------------
st.title("🎵 Moodify")
st.subheader("Music Recommendation System Based on Facial Emotion")

st.write("""
This application detects your facial emotion using AI
and recommends songs from YouTube based on your mood.
""")

# -----------------------------------
# EMOTION → MUSIC MAPPING
# -----------------------------------
def get_music_query(emotion):

    mapping = {
        "happy": "happy tamil songs",
        "sad": "sad melody tamil songs",
        "angry": "motivational rap songs",
        "fear": "calm relaxing music",
        "surprise": "party dance songs",
        "neutral": "relaxing instrumental music",
        "disgust": "rock music"
    }

    return mapping.get(emotion.lower(), "top tamil songs")


# -----------------------------------
# YOUTUBE SEARCH LINK
# -----------------------------------
def get_youtube_link(query):

    search_query = urllib.parse.quote(query)

    return f"https://www.youtube.com/results?search_query={search_query}"


# -----------------------------------
# EMOTION DETECTION
# -----------------------------------
def detect_emotion(image):

    try:

        result = DeepFace.analyze(
            image,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = result[0]['dominant_emotion']

        return emotion

    except Exception as e:

        st.error(f"Error: {e}")

        return "neutral"


# -----------------------------------
# CAMERA INPUT
# -----------------------------------
img_file = st.camera_input("📸 Capture Your Face")

# -----------------------------------
# PROCESS IMAGE
# -----------------------------------
if img_file is not None:

    # Convert image bytes to OpenCV format
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)

    frame = cv2.imdecode(file_bytes, 1)

    # Detect emotion
    emotion = detect_emotion(frame)

    st.success(f"Detected Emotion: {emotion.upper()}")

    # Music recommendation
    music_query = get_music_query(emotion)

    st.info(f"Recommended Music Type: {music_query}")

    # YouTube link
    youtube_link = get_youtube_link(music_query)

    st.markdown(
        f"[▶ Click Here to Play Music]({youtube_link})"
    )

    st.balloons()

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")

st.write("Developed using DeepFace, OpenCV, Streamlit, and YouTube")
