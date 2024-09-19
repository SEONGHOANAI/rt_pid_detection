import asyncio
import os
import subprocess
import time
import warnings
from pathlib import Path

from cfg import inference_config
from Detection_Stage import YOLOInference
from Segmentation_Stage import InferencePipeline
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from ultralytics import YOLO
from utilities.dataloader import CustomDataLoader
from utilities.pseudo_label_merger import PseudoLabelMerger
from utilities.weighted_box_fusion import WeightedBoxFusion

warnings.filterwarnings("ignore", category=UserWarning, message=".*_free_weak_ref")


def run_command(command):
    result = subprocess.run(command, shell=True, check=True)
    return result


yolo_repo_url = "https://github.com/ultralytics/yolov5"

yolo_dir = "yolov5"

if not os.path.exists(yolo_dir):
    print(f"Cloning YOLOv5 repository from {yolo_repo_url}")
    run_command(f"git clone {yolo_repo_url}")


def steps(step):
    if step == "step2":
        yolo_inference = YOLOInference(
            model_path=inference_config.BGR_weight_path,
            yolo_path=inference_config.yolo_v5_path,
            test_folder=inference_config.ensemble_image_path,
            output_folder=inference_config.BGR_output_folder,
        )
        yolo_inference.run_inference()

    else:
        yolo_inference = YOLOInference(
            model_path=inference_config.NBGR_weight_path,
            yolo_path=inference_config.yolo_v5_path,
            test_folder=inference_config.ensemble_image_path,
            output_folder=inference_config.NBGR_output_folder,
        )
        yolo_inference.run_inference()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    # Stage 1: Segmentation and Ensemble
    total_start_time = time.time()

    start_time = time.time()
    inference_pipeline = InferencePipeline(
        dataset_path=inference_config.dataset_path,
        yolo_model_path=inference_config.seg_yolo_model_path,
        reformed_image_path=inference_config.reformed_image_path,
        ensemble_image_path=inference_config.ensemble_image_path,
    )

    folder_path = Path(f"{inference_config.dataset_path}")
    image_list = sorted([str(folder_path / image) for image in os.listdir(folder_path)])
    print(f"total datasetnum {len(image_list)}")

    processes = []
    for model_name in ["DeepLabV3_ResNet50", "YOLOv8s-seg"]:
        p = mp.Process(
            target=inference_pipeline.run_inference,
            args=(
                model_name,
                image_list,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    inference_pipeline.run_ensemble()
    end_time = time.time()
    print(f"Stage 1 Time: {end_time - start_time}")

    start_time = time.time()
    processes2 = []
    for i in ["step2", "step3"]:
        p2 = mp.Process(
            target=steps,
            args=(i,),
        )
        p2.start()
        processes2.append(p2)

    for p in processes2:
        p.join()

    # # Stage 2: YOLO Inference for BGR
    # yolo_inference = YOLOInference(
    #     model_path=inference_config.BGR_weight_path,
    #     yolo_path=inference_config.yolo_v5_path,
    #     test_folder=inference_config.ensemble_image_path,
    #     output_folder=inference_config.BGR_output_folder,
    # )
    # yolo_inference.run_inference()

    # # Additional Stage: YOLO Inference for Background Non Remove
    # yolo_inference = YOLOInference(
    #     model_path=inference_config.NBGR_weight_path,
    #     yolo_path=inference_config.yolo_v5_path,
    #     test_folder=inference_config.ensemble_image_path,
    #     output_folder=inference_config.NBGR_output_folder,
    # )
    # yolo_inference.run_inference()

    end_time = time.time()
    print(f"Stage 2 Time:, {end_time - start_time}")

    # Weight Box Fusion
    wbf = WeightedBoxFusion(
        image_folder=inference_config.ensemble_image_path,
        predictions_file_A=os.path.join(
            inference_config.BGR_output_folder, "results.json"
        ),
        predictions_file_B=os.path.join(
            inference_config.NBGR_output_folder, "results.json"
        ),
        output_file=inference_config.wbf_output_file,
        iou_thr=inference_config.iou_thr,
    )
    wbf.run()

    total_end_time = time.time()
    print("total_time :", total_end_time - total_start_time)
