import cv2
import math
import numpy as np


def change_contrast_fuzzy(image: np.ndarray) -> np.ndarray:
    """
    Change the contrast of the image with fuzzy logic.
    :param image:   The image to enhance the contrast.
    :return:        The image with enhanced contrast.
    """
    # Gaussian Function:
    def Gaussian(x, mean, std):
        return np.exp(-0.5 * np.square((x-mean) / std))

    # Membership Functions:
    def ExtremelyDark(x, M):
        return Gaussian(x, -50, M/6)

    def VeryDark(x, M):
        return Gaussian(x, 0, M/6)

    def Dark(x, M):
        return Gaussian(x, M/2, M/6)

    def SlightlyDark(x, M):
        return Gaussian(x, 5*M/6, M/6)

    def SlightlyBright(x, M):
        return Gaussian(x, M+(255-M)/6, (255-M)/6)

    def Bright(x, M):
        return Gaussian(x, M+(255-M)/2, (255-M)/6)

    def VeryBright(x, M):
        return Gaussian(x, 255, (255-M)/6)

    def ExtremelyBright(x, M):
        return Gaussian(x, 305, (255-M)/6)

    def OutputFuzzySet(x, f, M, thres):
        x = np.array(x)
        result = f(x, M)
        result[result > thres] = thres
        return result

    def AggregateFuzzySets(fuzzy_sets):
        return np.max(np.stack(fuzzy_sets), axis=0)

    def Infer(i, M, get_fuzzy_set=False):
        # Calculate degree of membership for each class
        VD = VeryDark(i, M)
        Da = Dark(i, M)
        SD = SlightlyDark(i, M)
        SB = SlightlyBright(i, M)
        Br = Bright(i, M)
        VB = VeryBright(i, M)

        # Fuzzy Inference:
        x = np.arange(-50, 306)
        Inferences = (
            OutputFuzzySet(x, ExtremelyDark, M, VD),
            OutputFuzzySet(x, VeryDark, M, Da),
            OutputFuzzySet(x, Dark, M, SD),
            OutputFuzzySet(x, Bright, M, SB),
            OutputFuzzySet(x, VeryBright, M, Br),
            OutputFuzzySet(x, ExtremelyBright, M, VB)
        )

        # Calculate AggregatedFuzzySet:
        fuzzy_output = AggregateFuzzySets(Inferences)

        # Calculate crisp value of centroid
        if get_fuzzy_set:
            return np.average(x, weights=fuzzy_output), fuzzy_output
        return np.average(x, weights=fuzzy_output)

    # Convert RGB to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Get L channel
    l_channel = lab_image[:, :, 0]

    # Calculate M value
    M = np.mean(l_channel)
    if M < 128:
        M = 127 - (127 - M)/2
    else:
        M = 128 + M/2

    # Precompute the fuzzy transform
    x = list(range(-50, 306))
    FuzzyTransform = dict(zip(x, [Infer(np.array([i]), M) for i in x]))

    # Apply the transform to l channel
    u, inv = np.unique(l_channel, return_inverse=True)
    l_channel = np.array([FuzzyTransform[i] for i in u])[inv].reshape(l_channel.shape)

    # Min-max scale the output L channel to fit (0, 255):
    Min = np.min(l_channel)
    Max = np.max(l_channel)
    lab_image[:, :, 0] = (l_channel - Min)/(Max - Min) * 255

    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)


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


def sharpen_details(image: np.ndarray) -> np.ndarray:
    """
    Sharpen the details of the image.
    :param image:   The image to sharpen.
    :return:        The image with sharpened details.
    """
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8.5, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(image, -1, kernel)


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


def deskew(image: np.ndarray, enhancement: int, center_threshold: int) -> np.ndarray:
    """
    Deskew the image.
    :param image:               The image to deskew.
    :param enhancement:         Which enhance function to use.
    :param center_threshold:    Whether to exclude the center of the image.
    :return:                    The deskewed image.
    """
    enhance_functions = {
        0: lambda image: image,
        1: change_contrast_clahe,
        2: sharpen_details,
        3: lambda image: change_contrast_clahe(sharpen_details(image))
    }
    return rotate_image(
        enhance_functions[enhancement](image),
        compute_skewness(image, center_threshold)
    )


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
