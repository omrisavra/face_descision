import ctypes

import cv2


def set_keyboard_bindings(window, event_func):
    # see options in here https://anzeljg.github.io/rin2/book2/2405/docs/tkinter/key-names.html
    keyboards_options = ['<Return>', '<Left>', '<BackSpace>']
    for k in keyboards_options:
        window.bind(k, event_func)


def get_screen_width_height():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def resize_image_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        r = height / float(h)
        dim = (int(w * r), height)

    return cv2.resize(image, dim, interpolation=inter)
