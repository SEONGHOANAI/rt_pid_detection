# config.py

# Paths configuration
dataset_path = '/home/daehyuk/Desktop/Course/computer_vision/Final/dataset/test' # Inference Data Path
seg_yolo_model_path = "./yolo_seg_EF.pt" # Yolov8s Segmentation Model Path
reformed_image_path = './results/refactor' # Save Results Path about Yolov8s-seg and DeepLabv3
ensemble_image_path = './results/ensemble' # Save Results Path about combined Yolov8s-seg and DeepLabv3 
BGR_weight_path = './weights/BGR_yolov5l.pt' # Path to YOLO v5l for BackGround Remove 
NBGR_weight_path = './weights/NBGR_yolov5l.pt' # Path to Yolo v5l for Non BackGround Remove


yolo_v5_path = 'yolov5' # Yolo v5 model Path
BGR_output_folder = './results/BGR_output' # BackGround Remove Output Path
NBGR_output_folder = './results/NBGR_output' # Non BackGround Remove Output Path
wbf_output_file = './results/final_pred.json' # Final Pred.json Path
iou_thr = 0.5
