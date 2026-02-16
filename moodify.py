import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import tempfile
import os

# -----------------------------
# Load Emotion Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")

emotion_model = load_model(MODEL_PATH)

emotion_labels = [
    'Angry', 'Disgust', 'Fear',
    'Happy', 'Sad', 'Surprise', 'Neutral'
]

# -----------------------------
# Spotify Authentication
# -----------------------------
client_id = st.secrets["SPOTIFY_CLIENT_ID"]
client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
)

# -----------------------------
# Emotion → Genre Mapping
# -----------------------------
def get_genre(emotion):
    mapping = {
        'Happy': 'pop',
        'Sad': 'acoustic',
        'Angry': 'rock',
        'Surprise': 'dance',
        'Neutral': 'chill',
        'Fear': 'ambient',
        'Disgust': 'metal'
    }
    return mapping.get(emotion, 'pop')

# -----------------------------
# Spotify Song Fetch
# -----------------------------
def get_tracks_by_genre(genre):
    results = sp.search(
        q=f'genre:{genre}',
        type='track',
        limit=5
    )

    tracks = []
    for item in results['tracks']['items']:
        tracks.append({
            "name": item['name'],
            "artist": item['artists'][0]['name'],
            "url": item['external_urls']['spotify']
        })

    return tracks

# -----------------------------
# Emotion Detection (FIXED)
# -----------------------------
def detect_emotion_from_image(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return "Neutral"

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]

        # Resize to model input
        roi = cv2.resize(roi, (48, 48))

        # Normalize
        roi = roi / 255.0

        # Reshape → (1, 48, 48, 1)
        roi = roi.reshape(1, 48, 48, 1)

        prediction = emotion_model.predict(roi, verbose=0)
        emotion_index = int(np.argmax(prediction))

        return emotion_labels[emotion_index]

    return "Neutral"

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Moodify 🎵", layout="centered")

    st.title("🎵 Music Recommendation from Facial Emotion")
    st.write("Capture your face and get music recommendations based on your mood!")

    img_file = st.camera_input("📸 Capture your face")

    if img_file is not None:
        # Save image temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(img_file.getvalue())
        temp_file.close()

        frame = cv2.imread(temp_file.name)

        emotion = detect_emotion_from_image(frame)
        genre = get_genre(emotion)
        tracks = get_tracks_by_genre(genre)

        st.subheader(f"😀 Detected Emotion: **{emotion}**")
        st.subheader(f"🎧 Recommended Genre: **{genre}**")

        if tracks:
            st.markdown("### 🎶 Recommended Songs")
            for i, track in enumerate(tracks, start=1):
                st.markdown(
                    f"**{i}. {track['name']}** by *{track['artist']}*  \n"
                    f"[🔗 Listen on Spotify]({track['url']})"
                )
        else:
            st.warning("No tracks found. Try again!")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
