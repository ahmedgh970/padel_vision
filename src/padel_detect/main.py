import json
import hydra
from omegaconf import DictConfig
from padel_detect import Detector
from padel_detect import process_input

@hydra.main(version_base="1.3", config_name="padel_detect")
def cli(cfg: DictConfig):
    # cfg.input.path, cfg.output.json, cfg.threshold, cfg.models.pose, cfg.models.ball
    detector = Detector(
        pose_model_path=cfg.models.pose,
        ball_model_path=cfg.models.ball,
        conf_threshold=cfg.models.confidence
    )
    detections = process_input(
        path=cfg.input.path,
        detector=detector,
        proximity_thresh=cfg.threshold
    )
    with open(cfg.output.json, "w") as f:
        json.dump(detections, f, indent=2)
    print(f"Detections saved to {cfg.output.json}")

if __name__ == "__main__":
    cli()