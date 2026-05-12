import time
import logging
from src.ingestion.stream_handler import StreamHandler
from src.logic.state_machine import StateMachine
from src.inference.plate_detector import PlateDetector
from src.processing.perspective_warp import PerspectiveWarp
from src.ocr.paddle_reader import PaddleReader
from src.reporting.notifier import Notifier
from src.reporting.database import Database

def main():
    # 1. Initialize all components
    # These usually load configurations from a YAML or .env file
    stream = StreamHandler() 
    state_manager = StateMachine(idle_threshold=10) # 10 seconds stationary
    detector = PlateDetector()   # Loads YOLOv11
    warper = PerspectiveWarp()   # Geometry logic
    ocr_engine = PaddleReader()  # PaddleOCR with Thai support
    db = Database()
    notifier = Notifier()

    logging.info("System Initialized. Monitoring restricted zone...")

    try:
        while True:
            # STEP 1: Idle Monitoring (Low Resource)
            # Pulls a low-res frame from the substream
            frame_sub = stream.get_substream()
            
            if stream.has_motion(frame_sub):
                # STEP 2: State Management
                # Transition from 'Idle' to 'Active'
                status = state_manager.process_motion(detected=True)

                if status == "STATIONARY_TRIGGER":
                    logging.info("Vehicle detected as stationary. Starting AI Pipeline...")

                    # STEP 3: High-Res Capture (The Burst)
                    # Grabs high-quality frames for better OCR accuracy
                    burst = stream.get_highres_burst(n_frames=10)

                    # STEP 4: Detection & Selection
                    # Run YOLO and find the frame where the plate is clearest
                    best_frame, plate_bbox = detector.find_best_plate(burst)

                    if best_frame is not None:
                        # STEP 5: Perspective Warp
                        # Flatten the plate before sending to OCR
                        aligned_plate = warper.straighten(best_frame, plate_bbox)

                        # STEP 6: OCR
                        # Extract the text and province
                        plate_text, province = ocr_engine.extract_text(aligned_plate)
                        logging.info(f"Identity Found: {plate_text} ({province})")

                        # STEP 7: Logging & Reporting
                        db.save_event(plate_text, province, best_frame)
                        
                        # Conditional: If the vehicle eventually leaves after staying long enough
                        if state_manager.is_illegal_event():
                            notifier.send_evidence(plate_text, province, best_frame)

            else:
                # No motion; reset timers if the vehicle has left the zone
                state_manager.process_motion(detected=False)

            # Control the loop frequency to prevent CPU spikes
            time.sleep(0.1)

    except KeyboardInterrupt:
        logging.info("Shutting down watchdog...")
    finally:
        stream.release()

if __name__ == "__main__":
    main()