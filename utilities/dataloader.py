import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

class ImageData(Dataset):
    def __init__(self, image_list, model_name):
        self.image_list = image_list
        self.model_name = model_name

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        input_image = cv2.imread(image_path)

        if input_image is None:
            raise ValueError(f"Image not found or empty at path: {image_path}")

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        if self.model_name.startswith("YOLO"):
            new_image = np.zeros((352, 640, 3), dtype=np.uint8)
            height, width = input_image.shape[:2]
            new_image[0:height, 0:width] = input_image
            input_image = new_image

        input_image = input_image.astype(np.float32)
        input_image = input_image / 255.0
        input_image = input_image.transpose(2, 0, 1)
        return input_image, os.path.basename(image_path)


class CustomDataLoader(DataLoader):
    def __init__(self, image_list, model_name, batch_size=1, shuffle=False, pin_memory=True, num_workers=0):
        self.dataset = ImageData(image_list, model_name)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    def close(self):
        if hasattr(self.dataset, 'close'):
            self.dataset.close()
