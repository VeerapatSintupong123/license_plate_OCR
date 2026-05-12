import os
import numpy as np
from ultralytics import YOLO


AVAILABLE_MODELS = [
    'license-plate-finetune-v1l.pt',
    'license-plate-finetune-v1m.pt',
    'license-plate-finetune-v1n.pt',
    'license-plate-finetune-v1s.pt',
    'license-plate-finetune-v1x.pt',
]


class PlateDetector:
    """
    License plate detector using fine-tuned YOLO models.

    Available models:
        - license-plate-finetune-v1l.pt  (large)
        - license-plate-finetune-v1m.pt  (medium)
        - license-plate-finetune-v1n.pt  (nano)
        - license-plate-finetune-v1s.pt  (small)
        - license-plate-finetune-v1x.pt  (xlarge)

    Primary usage in pipeline (called by main.py):
        detector = PlateDetector()
        best_frame, plate_bbox = detector.find_best_plate(burst)
    """

    def __init__(
        self,
        model_name: str = 'license-plate-finetune-v1n.pt',
        model_dir: str = 'models',
        device: str = 'cuda',
        conf_threshold: float = 0.25,
    ):
        """
        Args:
            model_name:      Filename of the fine-tuned model (must be in AVAILABLE_MODELS).
            model_dir:       Directory that contains the model weights.
            device:          Inference device, e.g. 'cuda' or 'cpu'.
            conf_threshold:  Minimum confidence score to keep a detection.
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose one of: {AVAILABLE_MODELS}"
            )

        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")

        self.device = device
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)

    def find_best_plate(
        self,
        burst: list[np.ndarray],
    ) -> tuple[np.ndarray | None, list[float] | None]:
        """
        Run detection across a burst of high-res frames and return the single
        frame that contains the highest-confidence license plate detection.

        Called by main.py::
            best_frame, plate_bbox = detector.find_best_plate(burst)

        Args:
            burst: List of BGR frames (numpy arrays) from StreamHandler.get_highres_burst().

        Returns:
            (best_frame, plate_bbox) where:
                best_frame  – the BGR numpy array with the clearest plate, or
                              None if no plate was detected in any frame.
                plate_bbox  – bounding box [x1, y1, x2, y2] (float pixels, xyxy)
                              of the best detection, or None if not found.
        """
        best_frame: np.ndarray | None = None
        best_bbox: list[float] | None = None
        best_conf: float = -1.0

        for frame in burst:
            detections = self.inference(source=frame)
            if not detections:
                continue

            # Pick the highest-confidence plate in this frame
            top = max(detections, key=lambda d: d["confidence"])

            if top["confidence"] > best_conf and top['confidence'] >= self.conf_threshold:
                best_conf = top["confidence"]
                best_frame = frame
                best_bbox = top["bbox"]

        return best_frame, best_bbox

    def inference(
        self,
        source,
        save: bool = False,
        project: str = 'runs',
        name: str = 'detect',
        exist_ok: bool = True,
    ) -> list[dict]:
        """
        Run license-plate detection on *source* and return bounding boxes.

        Args:
            source:    Path to an image/video file, a directory of images,
                       or a numpy BGR frame (np.ndarray).
            save:      Whether to save annotated results to disk.
            project:   Root directory for saved results (used when save=True).
            name:      Sub-folder name inside *project* (used when save=True).
            exist_ok:  If True, existing output folders are reused silently.

        Returns:
            A list of detection dicts, one per detected license plate::

                [
                    {
                        "bbox":       [x1, y1, x2, y2],   # float pixels, xyxy format
                        "confidence": 0.94,                # float in [0, 1]
                        "class":      "license-plate",     # class label string
                    },
                    ...
                ]

            Returns an empty list when no plates are detected.
        """
        results = self.model.predict(
            source=source,
            save=save,
            project=project,
            name=name,
            exist_ok=exist_ok,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False,
        )

        return self._parse_results(results)

    def _parse_results(self, results) -> list[dict]:
        """Convert raw YOLO Results into a clean list of detection dicts."""
        detections: list[dict] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()    # (N, 4)
            scores = result.boxes.conf.cpu().numpy()   # (N,)
            classes = result.boxes.cls.cpu().numpy()   # (N,)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = float(scores[i])
                cls = int(classes[i])
                label = result.names[cls]

                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class": label,
                })

        return detections