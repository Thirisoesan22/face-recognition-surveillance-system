import cv2
import os
import sqlite3
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from mtcnn import MTCNN
from recognition import FaceDetector
import numpy as np
from scipy.spatial.distance import cosine
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import pandas as pd

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.create_layout()
        self.detector = MTCNN()
        project_dirpath = "E:\\Thesis\\burn\\myproject" #Change your directory accordingly
        self.project_dirpath = project_dirpath
        self.detector = FaceDetector(project_dirpath)
        self.cap = None
        self.is_running = False

        # Initialize tracking dictionary
        self.unknown_frame_count = {}

        # Calculate screen width and height
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # Calculate window width and height
        window_width = 1350
        window_height = 700

        # Calculate position for centering window
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)

        # Set window position
        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Set background image
        self.background_image = Image.open("ycc.jpg")
        self.background_image = self.background_image.resize((window_width, window_height), Image.ANTIALIAS)
        self.background_image = ImageTk.PhotoImage(self.background_image)
        self.background_label = tk.Label(self.window, image=self.background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.background_label.lower()  # Place the background label behind all other widgets

    def create_layout(self):
        label1 = tk.Label(self.window, text="University of Technology - Yatanarpon Cyber City", font=("Helvetica", 22), fg="green")
        label1.pack(pady=10)
        label2 = tk.Label(self.window, text="Department of Computer Engineering", font=("Helvetica", 20), fg="blue")
        label2.pack(pady=0)
        label3 = tk.Label(self.window, text="Smart Surveillance System Based On Face Recognition for CE Department", font=("Helvetica", 17), fg="red")
        label3.pack(pady=12)

        # Create a canvas to display the webcam feed
        self.canvas = tk.Canvas(self.window, width=630, height=440, bd=2, relief=tk.SOLID)
        self.canvas.pack(side=tk.TOP, padx=0, pady=0)

        left_label = tk.Label(self.window, text="Supervised by \n Daw Htar Ei Khine", font=("Helvetica", 15), bg="yellow", fg="purple")
        left_label.place(x=20, rely=0.97, anchor=tk.SW)

        # Add label at bottom right corner
        right_label = tk.Label(self.window, text="Presented by \n Ma Su Hlaing Oo", font=("Helvetica", 15), bg="yellow", fg="purple")
        right_label.place(relx=0.97, rely=0.97, anchor=tk.SE)

        # Add start button
        self.start_btn = tk.Button(self.window, text="Start", width=12, height=2, bg="indigo", fg="white", font=("Helvetica", 13, "bold"), command=self.start_video)
        self.start_btn.place(x=465, y=620)  # Adjust padding as needed

        # Add stop button
        self.stop_btn = tk.Button(self.window, text="Stop", width=12, height=2, bg="indigo", fg="white", font=("Helvetica", 13, "bold"), command=self.stop_video)
        self.stop_btn.place(x=605, y=620)  # Adjust padding as needed
        self.stop_btn.config(state=tk.DISABLED)

        # Add report button
        self.report_btn = tk.Button(self.window, text="Report", width=12, height=2, bg="red", fg="white", font=("Helvetica", 13, "bold"), command=self.send_report)
        self.report_btn.place(x=745, y=620)

        # Create a Label widget to act as a hyperlink
        link_label2 = tk.Label(self.window, text="Refresh", fg="cyan", bg="black", cursor="hand2", font=("Helvetica", 14, "underline"))
        link_label2.place(x=920, y=635)
        link_label2.bind("<Button-1>", self.confirm_clear_table)  # Bind to confirm_clear_table method

    def confirm_clear_table(self, event):
        answer = messagebox.askyesno("Confirmation", "Are you sure you want to clear the database?")
        if answer:
            self.clear_table()

    def clear_table(self):
        # Connect to the 'detectedFace.db' database and clear the table
        conn1 = sqlite3.connect(os.path.join(self.project_dirpath, 'detectedFace.db'))
        cursor1 = conn1.cursor()
        cursor1.execute("DELETE FROM detectedFaces")
        conn1.commit()

        # Attempt to reset the auto-increment value
        try:
            cursor1.execute("DELETE FROM sqlite_sequence WHERE name='detectedFaces'")
            conn1.commit()
        except sqlite3.OperationalError:
            print("sqlite_sequence table does not exist. Skipping reset.")

        conn1.close()
        print("Cleared all entries from detectedFaces table and attempted to reset IDs.")

        # Connect to the 'unknown.db' database and clear the table
        conn2 = sqlite3.connect(os.path.join(self.project_dirpath, 'unknown.db'))
        cursor2 = conn2.cursor()
        cursor2.execute("DELETE FROM unknown")
        conn2.commit()

        # Attempt to reset the auto-increment value
        try:
            cursor2.execute("DELETE FROM sqlite_sequence WHERE name='unknown'")
            conn2.commit()
        except sqlite3.OperationalError:
            print("sqlite_sequence table does not exist. Skipping reset.")

        conn2.close()
        print("Cleared all entries from unknown table and attempted to reset IDs.")

    def start_video(self):
        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        self.update_frame()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop_video(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.canvas.delete("all")

    def update_frame(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                result = self.detector.face_mtcnn_extractor(frame)
                unknown_faces = []  # Initialize an empty list to collect unknown faces

                if result:
                    for person in result:
                        x1, y1, x2, y2, width, height = self.detector.face_localizer(person)
                        new_embedding = self.detector.face_preprocessor(frame, x1, y1, x2, y2)
                        label_from_history = self.detector.compare_with_history(new_embedding)

                        if label_from_history is not None:
                            label = label_from_history
                            probability = "Tracked"
                        else:
                            label, probability = self.detector.face_dense_classifier(new_embedding)
                            if label != "unknown" and float(probability) >= self.detector.threshold:
                                self.detector.update_history(new_embedding, label)
                                # Reset unknown frame count for known labels
                                if label in self.unknown_frame_count:
                                    del self.unknown_frame_count[label]
                            elif label == "unknown":
                                # Initialize or update the frame count for unknown labels
                                if label not in self.unknown_frame_count:
                                    self.unknown_frame_count[label] = 0
                                self.unknown_frame_count[label] += 1
                                unknown_id = self.detector.track_unknown_face(new_embedding)
                                unknown_faces.append((x1, y1, x2, y2, unknown_id))  # Add unknown face to list with ID
                            else:
                                # Reset unknown frame count if not classified as unknown
                                if label in self.unknown_frame_count:
                                    del self.unknown_frame_count[label]

                        if label == "unknown":
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.95, (114, 250, 225), lineType=cv2.LINE_AA)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (114,250,225), lineType=cv2.LINE_AA)
                            self.detector.record_face(label)

                    if unknown_faces:
                        self.detector.unknown_count += 1
                    else:
                        self.detector.unknown_count = 0

                    if self.detector.unknown_count >= 3:
                        self.detector.record_unknown(frame, unknown_faces)  # Pass the list of unknown faces with IDs
                        self.detector.play_alarm()
                        self.detector.unknown_count = 0                           

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.imgtk = imgtk

            self.window.after(10, self.update_frame)

    def send_report(self):
        # Fetch and process data from detectedFace.db
        conn1 = sqlite3.connect(os.path.join(self.project_dirpath, 'detectedFace.db'))
        df1 = pd.read_sql_query("SELECT * FROM detectedFaces", conn1)
        df1['timestamp'] = pd.to_datetime(df1['timestamp'])
        df1.sort_values(by=['label', 'timestamp'], inplace=True)

        # Filter out duplicates within 3-minute intervals for the same label
        def filter_within_3_minutes(group):
            group = group.sort_values(by='timestamp')
            return group.loc[(group['timestamp'].diff() > pd.Timedelta(minutes=3)) | (group['timestamp'].diff().isnull())]

        df1 = df1.groupby('label', group_keys=False).apply(filter_within_3_minutes)
        
        # Sort the final dataframe by time
        df1.sort_values(by='timestamp', inplace=True)

        # Drop the 'id' column
        if 'id' in df1.columns:
            df1 = df1.drop(columns=['id'])

        # Add 'No.' column with sequential numbers
        df1.insert(0, 'No.', range(1, len(df1) + 1))

        # Save the processed data to a CSV file
        csv_path = os.path.join(self.project_dirpath, 'detected_faces_report.csv')
        df1.to_csv(csv_path, index=False)
        conn1.close()

        # Fetch and process data from unknown.db
        conn2 = sqlite3.connect(os.path.join(self.project_dirpath, 'unknown.db'))
        df2 = pd.read_sql_query("SELECT * FROM unknown", conn2)
        df2['time'] = pd.to_datetime(df2['time'])
        df2.sort_values(by=['time'], inplace=True)  # Sort by time instead of id
        df2 = df2.drop_duplicates(subset='id', keep='first')
        conn2.close()

        # Prepare email (add your desier sender, receiver email)
        sender_email = "sender@gmail.com"
        receiver_email = "receiver@gmail.com"
        password = "umxytmaqweniowu" #  generate an app password in the Google account

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = "Surveillance System Report"

        body = "Please find attached the report for detected faces.\n\n"
        msg.attach(MIMEText(body, 'plain'))

        html = """\
        <html>
        <body>
            <p>Below is the table for unknown faces:</p>
            <table border="1" cellpadding="5" cellspacing="0">
            <tr>
                <th>No.</th>
                <th>Label</th>
                <th>Time</th>
                <th>Photo</th>
            </tr>
        """

        for idx, (index, row) in enumerate(df2.iterrows(), start=1):
            img_path = row['photo']
            with open(img_path, 'rb') as img_file:
                img_data = img_file.read()
            img = MIMEImage(img_data, name=os.path.basename(img_path))

            img_id = f"image_{idx}"  # Unique ID for each image
            img.add_header('Content-ID', f"<{img_id}>")
            msg.attach(img)

            html += f"""\
            <tr>
            <td>{idx}</td>
            <td>{row['label']}</td>
            <td>{row['time']}</td>
            <td><img src="cid:{img_id}" style="width:120px;height:100px;"/></td>
            </tr>
            """

        html += """\
            </table>
        </body>
        </html>
        """
        msg.attach(MIMEText(html, 'html'))

        # Attach the CSV file
        with open(csv_path, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(csv_path)}')
            msg.attach(part)

        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()

        messagebox.showinfo("Report", "Report sent successfully!")

    
def main():
    root = tk.Tk()
    app = App(root, "Surveillance System")
    root.mainloop()

if __name__ == "__main__":
    main()
