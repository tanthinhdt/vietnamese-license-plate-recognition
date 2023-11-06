import cv2
import math
import numpy as np


def change_contrast_clahe(image: np.ndarray) -> np.ndarray:
    """
    Change the contrast of the image with CLAHE.
    :param image:   The image to enhance the contrast.
    :return:        The image with enhanced contrast.
    """
    # Splitting the image into L, A, B channels.
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab_image)

    # Applying CLAHE to the L channel.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))

    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the image.
    :param image:   The image to rotate.
    :param angle:   The angle to rotate.
    :return:        The rotated image.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(
        src=image, M=rot_mat, dsize=image.shape[1::-1], flags=cv2.INTER_LINEAR
    )


def compute_skewness(image: np.ndarray, center_threshold: int) -> float:
    """
    Compute the skewness of the image.
    :param image:           The image to compute the skewness.
    :param center_thres:    Whether to exclude the center of the image.
    :return:                The skewness of the image.
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    elif len(image.shape) == 2:
        height, width = image.shape
    else:
        raise Exception('Image type is not supported.')

    #
    edges = cv2.Canny(
        image=cv2.medianBlur(image, 3),   # Blur to reduce noise.
        threshold1=30,
        threshold2=100,
        apertureSize=3,
        L2gradient=True
    )

    #
    lines = cv2.HoughLinesP(
        image=edges,
        rho=1,
        theta=math.pi/180,
        threshold=30,
        minLineLength=width/1.5,
        maxLineGap=height/3.0
    )

    if lines is None:
        return 1

    min_line = 100
    min_line_pos = 0
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            center_point = [((x1+x2)/2), ((y1+y2)/2)]
            if center_threshold == 1:
                if center_point[1] < 7:
                    continue
            if center_point[1] < min_line:
                min_line = center_point[1]
                min_line_pos = i

    angle = 0.0
    cnt = 0
    for x1, y1, x2, y2 in lines[min_line_pos]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30:    # Excluding extreme rotations.
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt) * (180 / math.pi)


def deskew(image: np.ndarray, change_constant: int, center_threshold: int) -> np.ndarray:
    """
    Deskew the image.
    :param image:               The image to deskew.
    :param change_constant:     Whether to change the contrast of the image.
    :param center_threshold:    Whether to exclude the center of the image.
    :return:                    The deskewed image.
    """
    image = change_contrast_clahe(image) if change_constant == 1 else image
    return rotate_image(image, compute_skewness(image, center_threshold))


def get_frames(cap: cv2.VideoCapture) -> np.ndarray:
    """
    Get frames from the video.
    :param cap:     The video capture.
    :return:        The frame.
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break
