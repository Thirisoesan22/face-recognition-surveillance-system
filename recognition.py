import os
import cv2
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras.models import load_model, Model
from sklearn.preprocessing import Normalizer, LabelEncoder
import pygame
from scipy.spatial.distance import cosine
import sqlite3
from datetime import datetime

current_file_dir = os.path.dirname(os.path.abspath(__file__))

class FaceDetector:
    def __init__(self, project_dirpath):
        self.project_dirpath = project_dirpath
        self.model_path = os.path.join(project_dirpath, "file/My_Model.h5")
        self.dense_model = load_model(os.path.join(project_dirpath, "file/classifier.h5"))
        self.data = np.load(os.path.join(project_dirpath, "file/ce_dataset.npz"))
        self.model = load_model(self.model_path)
        self.feature_extractor = Model(inputs=self.model.input, outputs=self.model.get_layer('dropout').output)
        self.detector = MTCNN()
        self.unknown_count = 0
        pygame.mixer.init()
        self.history = []
        self.history_max_size = 10
        self.threshold = 0.8
        self.cosine_threshold = 0.5
        self.state = None
        self.photo_dir = os.path.join(project_dirpath, "file", "photos")
        if not os.path.exists(self.photo_dir):
            os.makedirs(self.photo_dir)

        # Initialize database
        self.init_database()

        self.previous_unknown_embeddings = []
        self.current_unknown_id = self.get_highest_unknown_id() + 1
        self.current_unknown_embedding = None
        self.unknown_face_tracking = {}  # Dictionary to track unknown faces and their frame counts

    def init_database(self):
        self.conn = sqlite3.connect(os.path.join(self.project_dirpath, 'detectedFace.db'))
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS detectedFaces
                               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               label TEXT,
                               timestamp TEXT)''')
        self.conn.commit()
        
        self.conn2 = sqlite3.connect(os.path.join(self.project_dirpath, 'unknown.db'))
        self.cursor2 = self.conn2.cursor()
        self.cursor2.execute('''CREATE TABLE IF NOT EXISTS unknown
                               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               label TEXT,
                               timestamp TEXT,
                               photo TEXT)''')
        self.conn2.commit()

    def get_highest_unknown_id(self):
        self.cursor2.execute('SELECT MAX(id) FROM unknown')
        result = self.cursor2.fetchone()
        if result[0] is not None:
            return result[0]
        else:
            return 0

    def is_significant_time_gap(self, last_time, current_time):
        """Check if there is a significant time gap between detections."""
        last_time = datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S")
        current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
        time_gap = current_time - last_time
        return time_gap.total_seconds() > 10  # 10 seconds threshold for considering a new entry

    def get_next_unknown_id(self):
        self.current_unknown_id += 1
        return self.current_unknown_id

    def record_face(self, label):
        if label != "unknown":
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.cursor.execute('''INSERT INTO detectedFaces (label, timestamp)
                                   VALUES (?, ?)''', (label, timestamp))
            self.conn.commit()
            print(f"Recorded face: {label} at {timestamp}")

    def record_unknown(self, frame, unknown_faces):
        now_db = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp for database
        now_file = datetime.now().strftime("%Y-%m-%d %Hh %Mm %Ss")  # Timestamp for file name

        for (x1, y1, x2, y2, face_id) in unknown_faces:
            photo_path = os.path.join(self.photo_dir, f"{now_file}_unknown_{face_id}.jpg")
            face = frame[y1:y2, x1:x2]
            cv2.imwrite(photo_path, face)  # Save the face
            self.cursor2.execute("INSERT INTO unknown (id, label, time, photo) VALUES (?, ?, ?, ?)", 
                                 (face_id, "unknown", now_db, photo_path))
            self.conn2.commit()
            print(f"Recorded unknown person ID {face_id} at {now_db}, photo saved to {photo_path}")

    def track_unknown_face(self, new_embedding):
        min_distance = float('inf')
        min_index = -1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Include timestamps in the previous unknown embeddings
        for idx, (emb, last_seen) in enumerate(self.previous_unknown_embeddings):
            distance = cosine(new_embedding, emb)
            if distance < min_distance and distance < self.cosine_threshold:
                min_distance = distance
                min_index = idx

        if min_index != -1 and not self.is_significant_time_gap(self.previous_unknown_embeddings[min_index][1], current_time):
            self.previous_unknown_embeddings[min_index] = (new_embedding, current_time)
            return min_index + 1
        else:
            unknown_id = self.get_next_unknown_id()
            self.previous_unknown_embeddings.append((new_embedding, current_time))
            return unknown_id

    def face_mtcnn_extractor(self, frame):
        result = self.detector.detect_faces(frame)
        return result

    def face_localizer(self, person):
        bounding_box = person['box']
        x1, y1 = abs(bounding_box[0]), abs(bounding_box[1])
        width, height = bounding_box[2], bounding_box[3]
        x2, y2 = x1 + width, y1 + height
        return x1, y1, x2, y2, width, height

    def face_preprocessor(self, frame, x1, y1, x2, y2, required_size=(160, 160)):
        face = frame[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        face_pixels = face_array.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = np.expand_dims(face_pixels, axis=0)
        yhat = self.feature_extractor.predict(samples)
        face_embedded = yhat[0]
        in_encoder = Normalizer(norm='l2')
        X = in_encoder.transform(face_embedded.reshape(1, -1))
        return X[0]

    def compare_with_history(self, new_embedding):
        for (embedding, label) in self.history:
            distance = cosine(new_embedding, embedding)
            if distance < self.cosine_threshold:
                return label
        return None

    def face_dense_classifier(self, X):
        yhat = self.dense_model.predict(X.reshape(1, -1))
        predicted_class_index = np.argmax(yhat)
        probability = yhat[0][predicted_class_index]
        trainy = self.data['arr_1']
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        predicted_class_label = out_encoder.inverse_transform([predicted_class_index])
        label = predicted_class_label[0]
        if probability < self.threshold:
            return "unknown", probability
        return label, probability

    def update_history(self, embedding, label):
        self.history.append((embedding, label))
        if len(self.history) > self.history_max_size:
            self.history.pop(0)

    def play_alarm(self):
        pygame.mixer.music.load('alarm.mp3')
        pygame.mixer.music.play()

    def face_detector(self):
        cap = cv2.VideoCapture(0)
        while True:
            __, frame = cap.read()
            result = self.face_mtcnn_extractor(frame)
            unknown_faces = []

            if result:
                for person in result:
                    x1, y1, x2, y2, width, height = self.face_localizer(person)
                    new_embedding = self.face_preprocessor(frame, x1, y1, x2, y2)
                    label_from_history = self.compare_with_history(new_embedding)

                    if label_from_history is not None:
                        label = label_from_history
                        probability = "Tracked"
                    else:
                        label, probability = self.face_dense_classifier(new_embedding)
                        if label != "unknown" and float(probability) >= self.threshold:
                            self.update_history(new_embedding, label)
                        elif self.current_unknown_embedding is not None and cosine(new_embedding, self.current_unknown_embedding) < self.cosine_threshold:
                            label = "unknown"
                            unknown_id = self.current_unknown_id
                        else:
                            label = "unknown"
                            unknown_id = self.track_unknown_face(new_embedding)
                            self.current_unknown_embedding = new_embedding
                            self.current_unknown_id = unknown_id

                    if label == "unknown":
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        unknown_faces.append((x1, y1, x2, y2, unknown_id))
                        
                        # Track unknown faces by ID
                        if unknown_id in self.unknown_face_tracking:
                            self.unknown_face_tracking[unknown_id] += 1
                        else:
                            self.unknown_face_tracking[unknown_id] = 1

                        # If unknown face is detected for 3 consecutive frames
                        if self.unknown_face_tracking[unknown_id] >= 3:
                            self.record_unknown(frame, [(x1, y1, x2, y2, unknown_id)])
                            self.unknown_face_tracking.pop(unknown_id)  # Reset tracking for this ID
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        self.record_face(label)  # Record detected label

                    print(" Person : {} , Probability : {}".format(label, probability))
                    cv2.putText(frame, label + " " + str(probability), (x1, y1 - 10),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0),
                                lineType=cv2.LINE_AA)
                    
                    if unknown_faces:
                        self.unknown_count += 1
                    else:
                        self.unknown_count = 0

                    if self.unknown_count >= 3:
                        self.play_alarm()
                        self.unknown_count = 0

            cv2.imshow('Face Recognition System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.conn.close()
        self.conn2.close()

if __name__ == "__main__":
    facedetector = FaceDetector(current_file_dir)
    facedetector.face_detector()
