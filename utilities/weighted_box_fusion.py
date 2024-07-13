
import json
from PIL import Image
import os
from collections import defaultdict
from ensemble_boxes import weighted_boxes_fusion

class WeightedBoxFusion:
    def __init__(self, image_folder, predictions_file_A, predictions_file_B, output_file, iou_thr=0.5, skip_box_thr=0.0):
        self.image_folder = image_folder
        self.predictions_file_A = predictions_file_A
        self.predictions_file_B = predictions_file_B
        self.output_file = output_file
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.image_sizes = self.get_image_sizes()

    def load_predictions(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def get_image_sizes(self):
        image_sizes = {}
        for filename in os.listdir(self.image_folder):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_id = os.path.splitext(filename)[0]
                image_path = os.path.join(self.image_folder, filename)
                with Image.open(image_path) as img:
                    width, height = img.size
                    image_sizes[image_id] = (width, height)
        return image_sizes

    def transform_predictions(self, raw_data):
        predictions = defaultdict(lambda: {'boxes': [], 'scores': [], 'labels': []})
        for item in raw_data:
            image_id = str(item['image_id'])
            width, height = self.image_sizes[image_id]
            x_min, y_min, box_width, box_height = item['bbox']
            x_max = x_min + box_width
            y_max = y_min + box_height
            x_min /= width
            x_max /= width
            y_min /= height
            y_max /= height
            predictions[image_id]['boxes'].append([x_min, y_min, x_max, y_max])
            predictions[image_id]['scores'].append(item['score'])
            predictions[image_id]['labels'].append(item['category_id'])
        return predictions

    def combine_predictions(self, predictions_A, predictions_B):
        combined_predictions = defaultdict(lambda: {'boxes': [], 'scores': [], 'labels': []})
        all_image_ids = set(predictions_A.keys()).union(set(predictions_B.keys()))
        for image_id in all_image_ids:
            if image_id in predictions_A and image_id in predictions_B:
                boxes_list = [predictions_A[image_id]['boxes'], predictions_B[image_id]['boxes']]
                scores_list = [predictions_A[image_id]['scores'], predictions_B[image_id]['scores']]
                labels_list = [predictions_A[image_id]['labels'], predictions_B[image_id]['labels']]
                boxes, scores, labels = weighted_boxes_fusion(
                    boxes_list, scores_list, labels_list, 
                    iou_thr=self.iou_thr, skip_box_thr=self.skip_box_thr
                )
                combined_predictions[image_id]['boxes'] = boxes
                combined_predictions[image_id]['scores'] = scores
                combined_predictions[image_id]['labels'] = labels
            elif image_id in predictions_A:
                combined_predictions[image_id] = predictions_A[image_id]
            elif image_id in predictions_B:
                combined_predictions[image_id] = predictions_B[image_id]
        return combined_predictions

    def restore_boxes(self, combined_predictions):
        restored_predictions = defaultdict(lambda: {'boxes': [], 'scores': [], 'labels': [], 'image_ids': []})
        for image_id, preds in combined_predictions.items():
            width, height = self.image_sizes[image_id]
            restored_boxes = []
            for box in preds['boxes']:
                x_min = box[0] * width
                y_min = box[1] * height
                x_max = box[2] * width
                y_max = box[3] * height
                box_width = x_max - x_min
                box_height = y_max - y_min
                restored_boxes.append([x_min, y_min, box_width, box_height])
            restored_predictions[image_id]['boxes'] = restored_boxes
            restored_predictions[image_id]['scores'] = preds['scores']
            restored_predictions[image_id]['labels'] = preds['labels']
            restored_predictions[image_id]['image_ids'] = [image_id] * len(restored_boxes)
        return restored_predictions

    def save_predictions_to_json(self, predictions):
        json_data = []
        for image_id, preds in predictions.items():
            for box, score, label, img_id in zip(preds['boxes'], preds['scores'], preds['labels'], preds['image_ids']):
                json_data.append({
                    'image_id': int(img_id),
                    'category_id': label,
                    'bbox': box,
                    'score': score
                })
        with open(self.output_file, 'w') as file:
            json.dump(json_data, file, indent=4)

    def run(self):
        raw_predictions_A = self.load_predictions(self.predictions_file_A)
        raw_predictions_B = self.load_predictions(self.predictions_file_B)
        predictions_A = self.transform_predictions(raw_predictions_A)
        predictions_B = self.transform_predictions(raw_predictions_B)
        combined_predictions = self.combine_predictions(predictions_A, predictions_B)
        restored_predictions = self.restore_boxes(combined_predictions)
        self.save_predictions_to_json(restored_predictions)


if __name__ == '__main__':
    image_folder = '/home/daehyuk/Desktop/Course/computer_vision/Final/dataset/train/train'
    predictions_file_A = 'pseudo_json/PD_DR.json' # Path_to_model1_pred_json
    predictions_file_B = 'pseudo_json/PD_YOLO.json' # Path_to_model2_pred_json
    output_file = 'pseudo_combined.json'

    wbf = WeightedBoxFusion(
        image_folder=image_folder,
        predictions_file_A=predictions_file_A,
        predictions_file_B=predictions_file_B,
        output_file=output_file,
        iou_thr=0.5
    )

    wbf.run()
