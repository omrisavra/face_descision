import os
from pathlib import Path

OPEN_FACE_OUT_DIR = Path(r"C:\Users\omris\Documents\20200825_projects\code\emotions_helper\assets\images_ouptut")
OPEN_FACE_OUT_IMG = OPEN_FACE_OUT_DIR / Path("img.jpg")
BLACK_BACKGROUND_IMG = OPEN_FACE_OUT_DIR / Path("black_img.jpg")
OPEN_FACE_DIR = Path(r"C:\Users\omris\Documents\20200825_projects\code\OpenFace_2.2.0_win_x64")
IMG_PARSER = OPEN_FACE_DIR / Path("FaceLandmarkImg.exe")


def analyze_img_open_face(img_path: Path, destination=OPEN_FACE_OUT_DIR):
    """
    Pass the image to open face detection analyze run it and store it in destination file
    :param img_path: example raw image to process on
    :param destination: the destination to the out openface dir
    """
    openface_command = rf'{IMG_PARSER} -f {img_path} -out_dir {destination} -tracked'
    os.system(openface_command)
