import json
from pathlib import Path

import torch


class YOLOInference:
    def __init__(self, model_path, yolo_path, test_folder, output_folder):
        self.model_path = model_path
        self.yolo_path = yolo_path
        self.test_folder = Path(test_folder)
        self.output_folder = Path(output_folder)
        self.model = self.load_model()

    def load_model(self):
        return torch.hub.load(
            self.yolo_path,
            "custom",
            path=self.model_path,
            source="local",
            force_reload=True,
        )

    def run_inference(self):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        coco_results = []

        img_paths = list(self.test_folder.glob("*.png"))
        for img_path in img_paths:
            results = self.model(str(img_path))

            predictions = results.pandas().xyxy[0]

            for _, prediction in predictions.iterrows():
                xmin, ymin, xmax, ymax = (
                    prediction["xmin"],
                    prediction["ymin"],
                    prediction["xmax"],
                    prediction["ymax"],
                )
                width = xmax - xmin
                height = ymax - ymin

                coco_results.append(
                    {
                        "image_id": img_path.stem,
                        "category_id": int(prediction["class"]),
                        "bbox": [float(xmin), float(ymin), float(width), float(height)],
                        "score": float(prediction["confidence"]),
                    }
                )

        json_path = self.output_folder / "results.json"
        with open(json_path, "w") as json_file:
            json.dump(coco_results, json_file, indent=4)
