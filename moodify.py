import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from youtubesearchpython import VideosSearch
import webbrowser
import tempfile

# -------------------------------
# Load Trained Emotion Model
# -------------------------------
emotion_model = load_model("emotion_model.h5")

# Emotion Labels
emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]

# -------------------------------
# Emotion → YouTube Music Mapping
# -------------------------------
def get_song_query(emotion):

    mapping = {
        'Happy': 'happy tamil songs',
        'Sad': 'sad melody tamil songs',
        'Angry': 'motivational rap songs',
        'Surprise': 'party dance songs',
        'Neutral': 'relaxing instrumental music',
        'Fear': 'calm peaceful music',
        'Disgust': 'rock music'
    }

    return mapping.get(emotion, 'top tamil songs')


# -------------------------------
# YouTube Song Search
# -------------------------------
def get_youtube_song(query):

    videosSearch = VideosSearch(query, limit=5)

    results = videosSearch.result()

    songs = []

    for item in results['result']:

        songs.append({
            'title': item['title'],
            'channel': item['channel']['name'],
            'link': item['link']
        })

    return songs


# -------------------------------
# Emotion Detection Function
# -------------------------------
def detect_emotion_from_image(image):

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Process detected face
    for (x, y, w, h) in faces:

        roi = gray[y:y+h, x:x+w]

        roi = cv2.resize(roi, (48, 48))

        roi = roi.astype("float32") / 255.0

        roi = np.expand_dims(roi, axis=0)

        roi = np.expand_dims(roi, axis=-1)

        # Predict emotion
        prediction = emotion_model.predict(roi)

        max_index = int(np.argmax(prediction))

        detected_emotion = emotion_labels[max_index]

        return detected_emotion

    return "Neutral"


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Moodify",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Moodify")
st.subheader("Music Recommendation Based on Facial Emotion")

st.write("""
This application detects your facial emotion using Deep Learning
and recommends songs from YouTube based on your mood.
""")

# -------------------------------
# Camera Input
# -------------------------------
img_file = st.camera_input("Capture Your Face")

# -------------------------------
# Process Image
# -------------------------------
if img_file is not None:

    # Save image temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)

    tfile.write(img_file.getvalue())

    # Read image using OpenCV
    frame = cv2.imread(tfile.name)

    # Detect emotion
    emotion = detect_emotion_from_image(frame)

    st.success(f"Detected Emotion: {emotion}")

    # Get search query
    query = get_song_query(emotion)

    st.info(f"Recommended Music Type: {query}")

    # Fetch YouTube songs
    songs = get_youtube_song(query)

    st.subheader("🎶 Recommended Songs")

    # Display Songs
    for i, song in enumerate(songs):

        st.write(f"### {i+1}. {song['title']}")

        st.write(f"Channel: {song['channel']}")

        st.markdown(
            f"[▶ Watch on YouTube]({song['link']})"
        )

    # Auto Open First Song
    if len(songs) > 0:

        webbrowser.open(songs[0]['link'])

    st.success("Enjoy Your Music 🎧")


# -------------------------------
# Footer
# -------------------------------
st.markdown("---")

st.write("Developed using CNN, OpenCV, Streamlit, and YouTube Search API")
