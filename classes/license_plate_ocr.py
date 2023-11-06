import numpy as np
from classes.base_model import BaseModel
from tools import check_point_closeness


class LicensePlateOCR(BaseModel):
    def load_from_checkpoint(self, checkpoint_path: str, confidence: float) -> None:
        super().load_from_checkpoint(checkpoint_path)
        self.model.conf = confidence

    def inference(self, image: np.ndarray) -> str:
        """
        Extract numbers and letters from the license plate.
        :param license_plate_ocr:   The license plate OCR model.
        :param image:               The image of the license plate.
        :return:                    The license plate.
        """
        LP_type = 1
        results = self.model(image)
        bb_list = results.pandas().xyxy[0].values.tolist()
        if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
            return None

        center_list = []
        y_mean = 0
        y_sum = 0
        for bb in bb_list:
            center_x = (bb[0] + bb[2]) / 2
            center_y = (bb[1] + bb[3]) / 2
            y_sum += center_y
            center_list.append([center_x, center_y, bb[-1]])

        # Find max-left and max-right point to draw line.
        left_point = center_list[0]
        right_point = center_list[0]
        for center_point in center_list:
            if center_point[0] < left_point[0]:
                left_point = center_point
            if center_point[0] > right_point[0]:
                right_point = center_point

        if left_point[0] == right_point[0]:
            return None

        # Check if 1 line plate or 2 line plate.
        for center_point in center_list:
            is_close = check_point_closeness(
                center_point[0], center_point[1],
                left_point[0], left_point[1],
                right_point[0], right_point[1]
            )
            if not is_close:
                LP_type = 2
                break

        # Get characters from license plate.
        line_1 = []
        line_2 = []
        license_plate = ""
        if LP_type == 2:
            y_mean = int(int(y_sum) / len(bb_list))
            for center in center_list:
                if int(center[1]) > y_mean:
                    line_2.append(center)
                else:
                    line_1.append(center)
            for line in sorted(line_1, key=lambda x: x[0]):
                license_plate += str(line[2])
            license_plate += "-"
            for line in sorted(line_2, key=lambda x: x[0]):
                license_plate += str(line[2])
        else:
            for line in sorted(center_list, key=lambda x: x[0]):
                license_plate += str(line[2])
        return license_plate
