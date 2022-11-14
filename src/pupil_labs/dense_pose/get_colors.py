import cv2
import numpy as np
import os

colors = []


def on_mouse_click(event, x, y, flags, frame):
    if event == cv2.EVENT_LBUTTONUP:
        colors.append(frame[y, x].tolist())


def main():
    frame = cv2.imread(
        os.path.join(os.path.dirname(__file__), "assets/body_shape_coloured.png")
    )
    while True:
        cv2.imshow("frame", frame)
        cv2.setMouseCallback("frame", on_mouse_click, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print(colors)


if __name__ == "__main__":
    main()
