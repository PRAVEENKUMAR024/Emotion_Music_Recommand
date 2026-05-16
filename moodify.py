import streamlit as st
import numpy as np
import cv2
from deepface import DeepFace
import urllib.parse

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
            img_path=image,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = result[0]['dominant_emotion']

        return emotion

    except Exception as e:

        st.error(f"Error Detecting Emotion: {e}")

        return "neutral"


# -----------------------------------
# CAMERA INPUT
# -----------------------------------
img_file = st.camera_input("📸 Capture Your Face")


# -----------------------------------
# PROCESS IMAGE
# -----------------------------------
if img_file is not None:

    try:

        # Convert image bytes to numpy array
        file_bytes = np.asarray(
            bytearray(img_file.read()),
            dtype=np.uint8
        )

        # Decode image
        frame = cv2.imdecode(file_bytes, 1)

        # Show captured image
        st.image(frame, channels="BGR")

        # Detect emotion
        with st.spinner("Detecting Emotion..."):

            emotion = detect_emotion(frame)

        # Display emotion
        st.success(f"Detected Emotion: {emotion.upper()}")

        # Get music recommendation
        music_query = get_music_query(emotion)

        st.info(f"Recommended Music Type: {music_query}")

        # Generate YouTube link
        youtube_link = get_youtube_link(music_query)

        # Display clickable link
        st.markdown(
            f"""
            ### 🎶 Recommended Songs
            
            [▶ Click Here to Play Music]({youtube_link})
            """
        )

        st.balloons()

    except Exception as e:

        st.error(f"Application Error: {e}")


# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")

st.write("Developed using Streamlit, DeepFace, OpenCV, and YouTube")
