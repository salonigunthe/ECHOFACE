import face_recognition
import os, sys
import cv2
import numpy as np
import math
import pickle  # For storing encodings

faces_dir = "C:/Users/Saloni/Desktop/Face Recognition/faces"  # Update with your full path


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        # Load known faces from saved encodings, if they exist
        if os.path.exists("encodings.pkl"):
            with open("encodings.pkl", "rb") as f:
                self.known_face_encodings, self.known_face_names = pickle.load(f)
            print("Loaded encodings from file.")
        else:
            self.encode_faces()
            # Save encodings to file for faster load next time
            with open("encodings.pkl", "wb") as f:
                pickle.dump((self.known_face_encodings, self.known_face_names), f)
            print("Saved encodings to file.")

    def encode_faces(self, num_samples=50):
        for person in os.listdir(faces_dir):
            count = 0
            for image_file in os.listdir(f'{faces_dir}/{person}'):
                if count >= num_samples:
                    break
                face_image = face_recognition.load_image_file(f'{faces_dir}/{person}/{image_file}')
                face_encoding = face_recognition.face_encodings(face_image)

                if len(face_encoding) > 0:
                    self.known_face_encodings.append(face_encoding[0])
                    self.known_face_names.append(person)
                    count += 1
            print(f"Encoded {count} images for {person}")

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found....')

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all faces in the current frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

                        if best_match_index < len(matches) and matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])
                        else:
                            name = 'Unknown'
                            confidence = 'Unknown'

                        self.face_names.append(f'{name} ({confidence})')
                    else:
                        self.face_names.append("Unknown (No match)")

            self.process_current_frame = not self.process_current_frame

            # Display annotations
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
