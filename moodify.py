import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import webbrowser
import urllib.parse
import tempfile

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
This application detects your facial emotion using Artificial Intelligence
and recommends songs from YouTube based on your mood.
""")

# -----------------------------------
# EMOTION → MUSIC MAPPING
# -----------------------------------
def get_music_query(emotion):

    emotion = emotion.lower()

    mapping = {

        "happy": "happy tamil songs",

        "sad": "sad melody tamil songs",

        "angry": "motivational rap songs",

        "fear": "calm relaxing music",

        "surprise": "party dance songs",

        "neutral": "relaxing instrumental music",

        "disgust": "rock music"
    }

    return mapping.get(emotion, "top tamil songs")


# -----------------------------------
# YOUTUBE SEARCH LINK
# -----------------------------------
def get_youtube_link(query):

    search_query = urllib.parse.quote(query)

    youtube_url = (
        f"https://www.youtube.com/results?"
        f"search_query={search_query}"
    )

    return youtube_url


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

    # Save temporary image
    tfile = tempfile.NamedTemporaryFile(delete=False)

    tfile.write(img_file.getvalue())

    # Read image
    frame = cv2.imread(tfile.name)

    # Detect emotion
    emotion = detect_emotion(frame)

    st.success(f"Detected Emotion: {emotion.upper()}")

    # Get music query
    music_query = get_music_query(emotion)

    st.info(f"Recommended Music Type: {music_query}")

    # Get YouTube link
    youtube_link = get_youtube_link(music_query)

    # Display link
    st.markdown(
        f"""
        ### 🎶 Open Recommended Songs
        
        [▶ Click Here to Play Music]({youtube_link})
        """
    )

    # Open automatically
    webbrowser.open(youtube_link)

    st.balloons()

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")

st.write("Developed using DeepFace, OpenCV, Streamlit, and YouTube")
