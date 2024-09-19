import asyncio
import logging
import os
import time
import warnings
from pathlib import Path

import aiofiles
import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from ultralytics import YOLO
from utilities.dataloader import CustomDataLoader, ImageData

warnings.filterwarnings("ignore", category=UserWarning, message=".*_free_weak_ref")


class InferencePipeline:
    def __init__(
        self, dataset_path, yolo_model_path, reformed_image_path, ensemble_image_path
    ):
        self.dataset_path = dataset_path
        self.yolo_model_path = yolo_model_path
        self.reformed_image_path = reformed_image_path
        self.ensemble_image_path = ensemble_image_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.person_class_id = 15

    def load_model(self, model_name):
        if model_name == "DeepLabV3_ResNet50":
            model = deeplabv3_resnet50(pretrained=True).to(self.device).eval()
        elif model_name == "YOLOv8s-seg":
            model = YOLO(self.yolo_model_path, verbose=False).to(self.device)
        return model

    @torch.no_grad()
    def predict_and_measure_time(self, model, input_batch, model_name):
        with autocast():
            if model_name.startswith("YOLO"):
                results = model(input_batch, verbose=False)
                batch_size = input_batch.shape[0]
                output_results = []

                for i in range(batch_size):
                    masks = results[i].masks
                    if masks is not None:
                        masks = masks.data
                        classes = results[i].boxes.cls
                        combined_mask = torch.zeros(
                            (input_batch.shape[2], input_batch.shape[3]),
                            dtype=torch.uint8,
                            device=self.device,
                        )
                        person_masks = masks[classes == 0]
                        for mask in person_masks:
                            mask = mask.to(torch.uint8)
                            combined_mask = combined_mask | mask
                    else:
                        combined_mask = torch.zeros(
                            (input_batch.shape[2], input_batch.shape[3]),
                            dtype=torch.uint8,
                            device=self.device,
                        )
                    output_results.append(combined_mask)
                return torch.stack(output_results)

            else:
                output = model(input_batch)["out"]
                output_predictions = output.argmax(1).long()
                combined_mask = output_predictions == self.person_class_id
                return combined_mask

    def visualize_result(self, input_image, mask):
        input_image = input_image.cpu().numpy().transpose(0, 2, 3, 1)
        mask = mask.cpu().numpy().astype(np.float32)
        person_image = input_image * mask[..., None] * 255.0
        return person_image

    async def save_image_async(self, image, path):
        async with aiofiles.open(path, "wb") as f:
            await f.write(image)

    def ensemble(self, filename):
        dlab_path = f"{self.reformed_image_path}/DeepLabV3_ResNet50/{filename}"
        yolo_path = f"{self.reformed_image_path}/YOLOv8s-seg/{filename}"

        dlab = cv2.imread(dlab_path)
        yolo = cv2.imread(yolo_path)

        if dlab is None or yolo is None:
            raise ValueError(f"Images for ensemble not found for filename: {filename}")

        assert dlab.shape == yolo.shape

        out_img = np.zeros_like(dlab)
        out_img = np.maximum(dlab, yolo)

        cv2.imwrite(f"{self.ensemble_image_path}/{filename}", out_img)

    def process_model(self, model_name, image_list):
        os.makedirs(f"{self.reformed_image_path}/{model_name}", exist_ok=True)
        model = self.load_model(model_name)
        data_loader = CustomDataLoader(
            image_list, model_name, batch_size=64, num_workers=16, pin_memory=False
        )
        start_time = time.time()
        for batch, labels in data_loader:
            batch = batch.to(self.device)

            mask = self.predict_and_measure_time(model, batch, model_name)
            person_seg_image = self.visualize_result(batch, mask)
            for batch_idx, filename in enumerate(labels):
                save_img = cv2.cvtColor(person_seg_image[batch_idx], cv2.COLOR_RGB2BGR)
                if model_name.startswith("YOLO"):
                    save_img = save_img[0:345, 0:640]
                asyncio.run(
                    self.save_image_async(
                        cv2.imencode(".png", save_img)[1].tobytes(),
                        f"{self.reformed_image_path}/{model_name}/{filename}",
                    )
                )
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:.2f} seconds")
        print(f"model : {model_name}")
        data_loader.close()

    def run_ensemble(self):
        folder_path = Path(f"{self.dataset_path}")
        os.makedirs(self.ensemble_image_path, exist_ok=True)

        for filename in os.listdir(folder_path):
            try:
                self.ensemble(filename)
            except Exception as e:
                print(f"Error for filename {filename}: {e}")
                continue

    def run_inference(
        self,
        model_name,
        image_list,
    ):
        inference_pipeline = InferencePipeline(
            self.dataset_path,
            self.yolo_model_path,
            self.reformed_image_path,
            self.ensemble_image_path,
        )
        inference_pipeline.process_model(model_name, image_list)


# if __name__ == "__main__":
#     # 경로 변수 설정
#     import torch.multiprocessing as mp

#     mp.set_start_method("spawn")
#     start_time = time.time()
#     DATASET_PATH = (
#         "/home/shan/advanced_vision/dataset_forfinal/dataset/train_labeled/train"
#     )
#     YOLO_MODEL_PATH = "./yolo_seg_EF.pt"
#     REFORMED_IMAGE_PATH = (
#         "/home/shan/advanced_vision/dataset_forfinal/dataset/refactor_train"
#     )
#     ENSEMBLE_IMAGE_PATH = (
#         "/home/shan/advanced_vision/dataset_forfinal/dataset/ensemble_train"
#     )

#     # 파이프라인 실행
#     # inference_pipeline = InferencePipeline(
#     #     DATASET_PATH, YOLO_MODEL_PATH, REFORMED_IMAGE_PATH, ENSEMBLE_IMAGE_PATH
#     # )
#     # inference_pipeline.run()

#     folder_path = Path(f"{DATASET_PATH}")
#     image_list = sorted([str(folder_path / image) for image in os.listdir(folder_path)])
#     print(f"total datasetnum {len(image_list)}")

#     processes = []
#     for model_name in ["DeepLabV3_ResNet50", "YOLOv8s-seg"]:
#         p = mp.Process(
#             target=run_inference,
#             args=(
#                 DATASET_PATH,
#                 YOLO_MODEL_PATH,
#                 REFORMED_IMAGE_PATH,
#                 ENSEMBLE_IMAGE_PATH,
#                 model_name,
#                 image_list,
#             ),
#         )
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

#     end_time = time.time()

#     print("total_time :", end_time - start_time)
