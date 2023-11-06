import os
import torch


class BaseModel:
    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        assert os.path.isfile(checkpoint_path), "Checkpoint does not exist."
        self.model = torch.hub.load(
            'yolov5', 'custom', path=checkpoint_path, force_reload=True, source='local'
        )

    def inference(self):
        raise NotImplementedError
