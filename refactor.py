import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet101, fcn_resnet101
from ultralytics import YOLO


class ImageData(Dataset):
    def __init__(self, image_list, model_name):
        self.image_list = image_list
        self.model_name = model_name

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        input_image = cv2.imread(image_path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        if self.model_name.startswith("YOLO"):
            # zero padding
            new_image = np.zeros((352, 640, 3), dtype=np.uint8)
            height, width = input_image.shape[:2]
            new_image[0:height, 0:width] = input_image
            input_image = new_image

        input_image = input_image.astype(np.float32)
        input_image = input_image / 255.0
        input_image = input_image.transpose(2, 0, 1)
        # print(self.image_list[idx].replace("train/", "refactor/"))
        return input_image, self.image_list[idx].replace(
            f"/home/shan/advanced_vision/dataset/{TRAIN_TEST}/", ""
        )


class CustomDataLoader(DataLoader):
    def __init__(
        self,
        image_list,
        model_name,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    ):
        self.dataset = ImageData(image_list, model_name)
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

    def __len__(self):
        return len(self.dataset) // self.batch_size  # type: ignore


# 모델 예측 및 시간 측정 함수
@torch.no_grad()
def predict_and_measure_time(model, input_batch, model_name):
    if model_name.startswith("YOLO"):
        results = model(input_batch)  # B, 3, H, W
        batch_size = input_batch.shape[0]
        output_results = []

        for i in range(batch_size):
            masks = results[i].masks
            if masks is not None:
                masks = masks.data
                classes = results[i].boxes.cls  # 클래스 정보 추출
                combined_mask = torch.zeros(
                    (input_batch.shape[2], input_batch.shape[3]),
                    dtype=torch.uint8,
                    device=device,
                )
                person_masks = masks[classes == 0]  # 사람 클래스만 선택
                for mask in person_masks:
                    mask = mask.to(torch.uint8)  # uint8 타입으로 변환
                    combined_mask = combined_mask | mask
            else:
                combined_mask = torch.zeros(
                    (input_batch.shape[2], input_batch.shape[3]),
                    dtype=torch.uint8,
                    device=device,
                )
            output_results.append(combined_mask)
        return torch.stack(output_results)

    else:
        output = model(input_batch)["out"]
        output_predictions = output.argmax(1).long()
        combined_mask = output_predictions == person_class_id
        return combined_mask


def visualize_result(input_image, mask):
    input_image = input_image.cpu().numpy().transpose(0, 2, 3, 1)
    mask = mask.cpu().numpy().astype(np.float32)
    person_image = input_image * mask[..., None] * 255.0
    return person_image


TRAIN_TEST = "test"

# 환경 설정 및 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 사람 클래스 ID (COCO 데이터셋에서 사람 클래스 ID는 15, Mask R-CNN에서는 1)
person_class_id = 15
# 모델 리스트
models_dict = {
    "DeepLabV3_ResNet101": deeplabv3_resnet101(pretrained=True).to(device).eval(),
    "FCN_ResNet101": fcn_resnet101(pretrained=True).to(device).eval(),
}
# YOLOv8 모델 추가
yolov8_models = {"YOLOv8s-seg": "/home/shan/advanced_vision/path/to/yolov8s-seg.pt"}
for model_name, model_path in yolov8_models.items():
    yolov8_model = YOLO(model_path, verbose=False)
    yolov8_model.to(device)
    models_dict[model_name] = yolov8_model

folder_path = Path(f"/home/shan/advanced_vision/dataset/{TRAIN_TEST}")
image_list = os.listdir(folder_path)
for i, image in enumerate(image_list):
    image_list[i] = str(folder_path / image)
print("total datasetnum", len(image_list))

for model_name, model in models_dict.items():
    if not (model_name == "DeepLabV3_ResNet101"):
        continue
    data_loader = CustomDataLoader(
        image_list, model_name, batch_size=64, num_workers=4, pin_memory=True
    )

    start_time = time.time()
    for batch, labels in data_loader:
        batch = batch.to(device)

        mask = predict_and_measure_time(model, batch, model_name)
        person_seg_image = visualize_result(batch, mask)
        for batch_idx, image in enumerate(labels):
            save_img = cv2.cvtColor(person_seg_image[batch_idx], cv2.COLOR_RGB2BGR)
            # For grey scale
            # save_img = cv2.cvtColor(save_img, cv2.COLOR_BGR2GRAY)
            # save_img[save_img != 0] = 255
            # Save image
            if model_name.startswith("YOLO"):
                save_img = save_img[0:345, 0:640]
            cv2.imwrite(
                f"/home/shan/advanced_vision/dataset/refactor/{model_name}_{image}",
                save_img,
            )
            # print(f"/home/shan/advanced_vision/dataset/refactor/{model_name}_{image}")
            # print("image", image)
            # print("model", model_name)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    print("model :", model_name)

# # 결과 테스트
# plt.figure(figsize=(10, 10))
# plt.imshow(person_seg_image)
# plt.title(f"{image}")
# plt.axis("off")
# plt.show()
