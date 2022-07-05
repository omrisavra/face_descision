import datetime
import tkinter
from collections import Counter
from pathlib import Path
from typing import List

import PIL.Image
import PIL.ImageTk
import cv2
from deepface import DeepFace

from open_face import OPEN_FACE_OUT_IMG, analyze_img_open_face, BLACK_BACKGROUND_IMG
from utils import set_keyboard_bindings, get_screen_width_height, resize_image_with_aspect_ratio

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'  # getting a haarcascade xml file
face_cascade = cv2.CascadeClassifier()  # processing it for our project
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  # adding a fallback event
    print("Error loading xml file")

LOGGING = True

VIDEO_TIME = 3
DEFAULT_NO_EMOTION = "I don't see your pretty face"

VIDEO_WIDTH, VIDEO_HEIGHT = get_screen_width_height()

video = cv2.VideoCapture(0)  # requisting the input from the webcam or camera
video.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)

SUPPORT_BUTTON = False


def detect_face(frame):
    # changing the video to grayscale to make the face analisis work properly
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"got an exception in here {e}")

    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


def most_frequent(emotions: List):
    occurence_count = Counter(emotions)
    most_common_emotion = occurence_count.most_common(1)[0][0]
    if most_common_emotion is None:
        return DEFAULT_NO_EMOTION
    if LOGGING:
        print("most common emotion is", most_common_emotion)
    return most_common_emotion


def is_time_pass(start_time):
    return datetime.datetime.now() < start_time + datetime.timedelta(seconds=VIDEO_TIME)


def rectangle_face(frame) -> None:
    face = detect_face(frame)
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),
                      1)  # making a recentangle to show up and detect the face and setting it position and colour


def get_frame_emotion(frame):
    try:
        analyze = DeepFace.analyze(frame, actions=[
            'emotion'])  # same thing is happing here as the previous example, we are using the analyze class from deepface and using ‘frame’ as input
        print(analyze)
        return analyze['dominant_emotion']
    except Exception as e:
        if LOGGING:
            print("got exception with deepface analyze", e)
        return None


def show_image(frame):
    cv2.imshow('video', frame)
    # allow cv2 process the image
    cv2.waitKey(1)


def record_emotions(video):
    emotions = []
    start_time = datetime.datetime.now()
    frame = None
    last_frame = None
    while video.isOpened() and is_time_pass(start_time):
        _, frame = video.read()
        if frame is not None:
            last_frame = frame.copy()
        rectangle_face(frame)
        emotion = get_frame_emotion(frame)
        show_image(frame)
        emotions.append(emotion)
    cv2.destroyAllWindows()
    return [emotions, last_frame]


def init_black_image():
    img = PIL.Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0))
    img.save(BLACK_BACKGROUND_IMG)


class ScreenFeed(object):
    def __init__(self, window):
        self.canvas = tkinter.Canvas(window, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        try:
            with PIL.Image.open(OPEN_FACE_OUT_IMG) as im:
                self.photo = PIL.ImageTk.PhotoImage(image=im)
                self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        except FileNotFoundError as e:
            print("file not found", e)

        self.canvas.pack()
        if SUPPORT_BUTTON:
            btn_blur = tkinter.Button(window, text="Try it", width=50, command=self.handle_button)
            btn_blur.pack(anchor=tkinter.CENTER, expand=True)
        set_keyboard_bindings(window, lambda event: self.handle_button())

    def update_canvas_image(self, img_path: Path) -> None:
        with PIL.Image.open(img_path) as im:
            self.photo = PIL.ImageTk.PhotoImage(image=im)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        self.canvas.update_idletasks()

    def handle_button(self):
        self.update_canvas_image(BLACK_BACKGROUND_IMG)
        emotions, last_frame = record_emotions(video)
        font = cv2.FONT_HERSHEY_SIMPLEX
        resized_frame = resize_image_with_aspect_ratio(last_frame, VIDEO_WIDTH, VIDEO_HEIGHT)
        cv2.putText(resized_frame, most_frequent(emotions), (10, 450), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        if last_frame is not None:
            cv2.imwrite(str(OPEN_FACE_OUT_IMG), resized_frame)
        analyze_img_open_face(OPEN_FACE_OUT_IMG)
        self.update_canvas_image(OPEN_FACE_OUT_IMG)


def main():
    init_black_image()
    window = tkinter.Tk()
    try:
        ScreenFeed(window)
        window.mainloop()
        print("finish running")
    except KeyboardInterrupt:
        video.release()
        print("bye bye :)")


if __name__ == "__main__":
    main()
