import numpy as np
from classes.base_model import BaseModel


class LicensePlateDetector(BaseModel):
    def inference(self, image: np.ndarray) -> list:
        """
        Detect the license plate(s) in the image.
        :param image:   The image of the car.
        :return:        The license plate(s).
        """
        return self.model(image).xyxy[0].tolist()
