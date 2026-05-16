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
Capture your face and Moodify will detect your emotion
and recommend music from YouTube.
""")

# -----------------------------------
# EMOTION → MUSIC MAPPING
# -----------------------------------
def get_music_query(emotion):

    emotion_map = {

        "happy": "happy tamil songs",

        "sad": "sad melody tamil songs",

        "angry": "motivational rap songs",

        "fear": "calm relaxing music",

        "surprise": "party dance tamil songs",

        "neutral": "relaxing instrumental tamil music",

        "disgust": "rock music"
    }

    return emotion_map.get(
        emotion.lower(),
        "top tamil songs"
    )


# -----------------------------------
# YOUTUBE LINK
# -----------------------------------
def get_youtube_link(query):

    query = urllib.parse.quote(query)

    return (
        f"https://www.youtube.com/results?"
        f"search_query={query}"
    )


# -----------------------------------
# DETECT EMOTION
# -----------------------------------
def detect_emotion(image):

    try:

        result = DeepFace.analyze(
            img_path=image,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = result[0]["dominant_emotion"]

        return emotion

    except Exception as e:

        st.error(f"Emotion Detection Error: {e}")

        return "neutral"


# -----------------------------------
# CAMERA INPUT
# -----------------------------------
img_file = st.camera_input(
    "📸 Capture Your Face"
)

# -----------------------------------
# PROCESS IMAGE
# -----------------------------------
if img_file is not None:

    try:

        # Convert image to numpy array
        bytes_data = img_file.getvalue()

        np_array = np.frombuffer(
            bytes_data,
            np.uint8
        )

        # Decode image
        image = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        # Display image
        st.image(
            image,
            channels="BGR"
        )

        # Detect emotion
        with st.spinner(
            "Analyzing Emotion..."
        ):

            emotion = detect_emotion(image)

        # Show emotion
        st.success(
            f"Detected Emotion: "
            f"{emotion.upper()}"
        )

        # Get music recommendation
        music_query = get_music_query(
            emotion
        )

        st.info(
            f"Recommended Music: "
            f"{music_query}"
        )

        # Generate YouTube link
        youtube_link = get_youtube_link(
            music_query
        )

        # Show clickable link
        st.markdown(
            f"""
            ### 🎶 Play Recommended Songs

            [▶ Click Here to Open YouTube]({youtube_link})
            """
        )

        st.balloons()

    except Exception as e:

        st.error(
            f"Application Error: {e}"
        )

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")

st.write(
    "Developed using Streamlit, "
    "OpenCV, DeepFace, and YouTube"
)
