import json
import os

class PseudoLabelMerger:
    def __init__(self, original_labels_path, pseudo_labels_path, image_dir, output_path):
        self.original_labels_path = original_labels_path
        self.pseudo_labels_path = pseudo_labels_path
        self.image_dir = image_dir
        self.output_path = output_path

    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def save_json(self, data, path):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def merge_labels(self):
        original_labels = self.load_json(self.original_labels_path)
        pseudo_labels = self.load_json(self.pseudo_labels_path)

        # Convert image ID to integer if needed
        for pseudo_label in pseudo_labels:
            pseudo_label['image_id'] = int(pseudo_label['image_id'])

        for pseudo_label in pseudo_labels:
            image_id = pseudo_label['image_id']
            category_id = pseudo_label['category_id']
            bbox = pseudo_label['bbox']
            score = pseudo_label['score']

            # Convert pseudo label to original label format
            annotation = {
                'id': image_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0,
                'segmentation': []
            }

            original_labels['annotations'].append(annotation)

        self.save_json(original_labels, self.output_path)
        return original_labels

    def update_image_entries(self, merged_labels):
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        image_files_in_json = {img['file_name']: img['id'] for img in merged_labels['images']}
        expected_filenames = [f'train/{i}.png' for i in range(len(image_files))]

        missing_files = [fname for fname in expected_filenames if fname not in image_files_in_json]

        for missing_file in missing_files:
            file_id = int(missing_file.split('/')[-1].split('.')[0])
            new_image_entry = {
                'id': file_id,
                'width': 640,
                'height': 345,
                'file_name': missing_file
            }
            merged_labels['images'].append(new_image_entry)

        merged_labels['images'] = sorted(merged_labels['images'], key=lambda x: x['id'])
        self.save_json(merged_labels, self.output_path)
        print(f"Added {len(missing_files)} missing files to the JSON.")

    def run(self):
        merged_labels = self.merge_labels()
        self.update_image_entries(merged_labels)


if __name__ == '__main__':
    original_labels_path = '../json_config/train_labeled.json'
    pseudo_labels_path = '../json_config/train_pseudo_labels.json'
    image_dir = '../dataset/train/train'
    output_path = 'temp.json'

    merger = PseudoLabelMerger(original_labels_path, pseudo_labels_path, image_dir, output_path)
    merger.run()
