import time
import logging
import yaml
from src.ingestion.stream_handler import StreamHandler
from src.logic.state_machine import StateMachine, State
from src.inference.plate_detector import PlateDetector
from src.processing.perspective_warp import PerspectiveWarp
from src.ocr.paddle_reader import PaddleReader
from src.reporting.notifier import Notifier
from src.reporting.database import Database

def load_config(path: str = r'config/settings.yaml') -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    config = load_config(r'config/settings.yaml')
    stream = StreamHandler(config)
    state_manager = StateMachine(
        stationary_threshold=config.get("stationary_threshold", 10),
        illegal_threshold=config.get("illegal_threshold", 60),
    )
    detector = PlateDetector()
    warper = PerspectiveWarp()
    ocr_engine = PaddleReader()
    db = Database()
    notifier = Notifier()

    # Remember OCR result across frames so EXIT can use it
    last_plate_text  = ""
    last_province    = ""
    last_best_frame  = None

    logging.info("System initialized. Monitoring restricted zone...")
    status = None

    try:
        while True:
            frame_sub = stream.get_substream()
            freeze    = status in ("ENTRY", "TRACKING", "STATIONARY_TRIGGER", "STATIONARY")
            detected  = stream.has_motion(frame_sub, freeze_bg=freeze)
            status    = state_manager.process_motion(detected=detected)

            # ── Transition: vehicle just became stationary ──────────────────
            if status == "STATIONARY_TRIGGER":
                logging.info("Vehicle stationary — starting AI pipeline...")

                burst = stream.get_highres_burst(n_frames=10)
                best_frame, plate_bbox = detector.find_best_plate(burst)

                # ── False positive: no vehicle plate found ──────────────────────────
                if best_frame is None:
                    logging.info("No plate detected — likely false positive. Resetting.")
                    state_manager.reset()
                    continue

                if best_frame is not None:
                    aligned_plate = warper.straighten(best_frame, plate_bbox)
                    last_plate_text, last_province = ocr_engine.extract_text(aligned_plate)
                    last_best_frame = best_frame
                    logging.info(f"Identity: {last_plate_text} ({last_province})")
                    db.save_event(last_plate_text, last_province, best_frame)

            # ── Transition: vehicle just left the zone ──────────────────────
            elif status == "EXIT":
                if state_manager.is_illegal_event() and last_plate_text:
                    logging.warning(f"Illegal event confirmed: {last_plate_text}")
                    notifier.send_evidence(last_plate_text, last_province, last_best_frame)

                # Clear cached OCR result for the next vehicle
                last_plate_text = ""
                last_province   = ""
                last_best_frame = None

            time.sleep(0.1)

    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        stream.release()

if __name__ == "__main__":
    main()
