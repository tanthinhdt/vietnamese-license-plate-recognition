import cv2
import numpy as np
from tqdm import tqdm
from moviepy import editor as mp
from tools import deskew, get_frames
from classes import BaseModel, LicensePlateDetector, LicensePlateOCR


class LicensePlateRecognition(BaseModel):
    def load_from_checkpoint(
        self, detector_checkpoint_path: str,
        ocr_checkpoint_path: str,
        confidence: float
    ) -> None:
        self.detector = LicensePlateDetector()
        self.detector.load_from_checkpoint(detector_checkpoint_path)
        self.ocr = LicensePlateOCR()
        self.ocr.load_from_checkpoint(ocr_checkpoint_path, confidence)

        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def inference(self, modality: str, src_path: str = None, dest_path: str = None) -> None:
        """
        Inference.
        :param modality:        The modality of the inference.
        :param src_path:        The source path.
        :param dest_path:       The destination path.
        """
        if modality == "image":
            self.inference_image(src_path, dest_path)
        elif modality == "video":
            self.inference_video(src_path, dest_path)
        elif modality == "realtime":
            self.inference_realtime()
        else:
            raise Exception("Modality is not supported.")

    def inference_image(self, image_path: str, output_path: str) -> None:
        """
        Inference the image.
        :param image_path:      The path to the image.
        :param output_path:     The path to the output image.
        """
        assert image_path is not None, "Image path is not provided."
        image = cv2.imread(image_path)
        cv2.imwrite(output_path, self.inference_image_array(image))

    def inference_video(self, video_path: str, output_path: str) -> None:
        """
        Inference the video.
        :param video_path:      The path to the video.
        :param output_path:     The path to the output video.
        """
        assert video_path is not None, "Video path is not provided."

        video = mp.VideoFileClip(video_path)
        duration = video.duration
        fps = video.fps
        video.close()

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        writer = cv2.VideoWriter(
            output_path,
            self.fourcc,
            fps,
            (frame_width, frame_height),
        )

        progress_bar = tqdm(
            get_frames(cap),
            desc="Process frames",
            unit="frame",
            total=int(duration * fps),
            leave=False,
        )
        for frame in progress_bar:
            writer.write(self.inference_image_array(frame))

        cap.release()
        writer.release()

    def inference_realtime(self) -> None:
        """
        Inference in realtime.
        """
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Traffic Light Detector", self.inference_image_array(frame))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        cv2.destroyAllWindows()

    def inference_image_array(self, image: np.ndarray) -> np.ndarray:
        """
        Inference the image array.
        :param image:       The image array.
        :return:            The image array with bounding boxe(s) and label(s).
        """
        license_plates = self.detector.inference(image)
        for license_plate in license_plates:
            x1, y1, x2, y2, _, _ = map(int, license_plate)
            cropped_image = image[y1:y2, x1:x2]
            cv2.rectangle(
                img=image,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(0, 0, 225),
                thickness=2,
            )

            license = ""
            flag = False
            for change_constant in range(2):
                for center_threshold in range(2):
                    license = self.ocr.inference(
                        deskew(cropped_image, change_constant, center_threshold)
                    )
                    if license is not None:
                        image = cv2.putText(
                            img=image,
                            text=license,
                            org=(x1, y1 - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.9,
                            color=(36, 255, 12),
                            thickness=2,
                        )
                        flag = True
                        break
                if flag:
                    break
        return image
