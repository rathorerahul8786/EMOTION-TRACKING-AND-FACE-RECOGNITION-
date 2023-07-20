import cv2
import os
import streamlit as st
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from tensorflow import keras

from keras.models import load_model


# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access Webcam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = None

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time,Emotion')


# Get the total number of registered users
def total_reg():
    return len(os.listdir('static/faces'))


# Extract the face from an image
def extract_faces(img):
    if img is None or img.size == 0:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


# Identify face using ML model
def identify_face(face_array):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(face_array)


# Detect emotion using ML model
def detect_emotion(face):
    model = load_model('emotion_detection_model.h5')
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.astype("float") / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion_prediction = model.predict(face)[0]
    emotion = emotion_labels[np.argmax(emotion_prediction)]
    return emotion


# Train the model on all the faces available in the faces folder
def train_model():
    faces = []
    labels = []
    user_list = os.listdir('static/faces')
    for user in user_list:
        for img_name in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{img_name}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from today's attendance file in the attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    emotions = df['Emotion']
    l = len(df)
    return names, rolls, times, emotions, l


# Add attendance of a specific user
def add_attendance(name, emotion):
    username = name.split('_')[0]
    userid = name.split('_')[1] if len(name.split('_')) > 1 else ''
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if userid and int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},{emotion}')


# Clear the attendance file
def clear_attendance():
    global cap
    if cap is not None:
        cap.release()  # Release webcam capture resources
    cap = None
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    if os.path.isfile(attendance_file):
        os.remove(attendance_file)
        st.success("Attendance cleared successfully.")
    else:
        st.warning("No attendance file found.")


# Main page
def home():
    global cap
    names, rolls, times, emotions, l = extract_attendance()
    st.title("SMART ATTENDANCE AND EMOTION TRACKING SYSTEM USING FACIAL RECOGNITION TECHNOLOGY")
    st.image('https://emerj.com/wp-content/uploads/2018/04/facial-recognition-applications-security-retail-and-beyond.jpg',
             use_column_width=True)
    st.write(f"Date: {datetoday2}")
    st.write(f"Total Registered Users: {total_reg()}")

    if st.button("Take Attendance"):
        st.write("Taking attendance...")
        start()

    if st.button("Clear Attendance"):
        clear_attendance()

    st.write("Attendance:")
    attendance_df = pd.DataFrame({"Name": names, "Roll": rolls, "Time": times, "Emotion": emotions})
    st.write(attendance_df)


# Main function to run the Streamlit App
def main():
    st.set_page_config(page_title="Attendance Tracking System Using Facial Technology")
    menu = ["Home", "Add User", "View Registered Users"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        home()
    elif choice == "Add User":
        add()
    elif choice == "View Registered Users":
        select_user()


if __name__ == '__main__':
    main()
