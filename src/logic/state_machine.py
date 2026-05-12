import time
import logging

class State:
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    STATIONARY = "STATIONARY"
    COMPLETED = "COMPLETED"

class StateMachine:
    def __init__(self, stationary_threshold=10, illegal_threshold=60):
        """
        :param stationary_threshold: Seconds of no-motion before triggering AI.
        :param illegal_threshold: Seconds in zone before flagging as illegal dumping.
        """
        self.state = State.IDLE
        self.stationary_threshold = stationary_threshold
        self.illegal_threshold = illegal_threshold
        
        # Timing trackers
        self.entry_time = None
        self.stationary_start_time = None
        self.last_motion_time = None
        
        # Event flags
        self.ai_triggered = False

    def process_motion(self, detected: bool):
        """
        The main transition logic called every frame.
        Returns the current system status to main.py.
        """
        current_time = time.time()

        # STATE: IDLE -> TRACKING
        if self.state == State.IDLE and detected:
            self.state = State.TRACKING
            self.entry_time = current_time
            self.last_motion_time = current_time
            logging.info("State: Tracking - Vehicle entered ROI.")
            return "ENTRY"

        # STATE: TRACKING -> STATIONARY
        elif self.state == State.TRACKING:
            if detected:
                self.last_motion_time = current_time
            else:
                # If motion stops, check how long it's been still
                still_duration = current_time - self.last_motion_time
                if still_duration >= self.stationary_threshold:
                    self.state = State.STATIONARY
                    self.stationary_start_time = current_time
                    logging.info(f"State: Stationary - Vehicle stopped for {self.stationary_threshold}s.")
                    return "STATIONARY_TRIGGER"

        # STATE: STATIONARY -> COMPLETED (Vehicle Leaves)
        elif self.state == State.STATIONARY:
            if detected:
                # If motion starts again after being stationary, they are likely leaving
                # Or they moved the car to a different spot in the zone.
                self.last_motion_time = current_time
            
            # Check if they have vanished from the ROI for a few seconds
            if not detected and (current_time - self.last_motion_time > 3):
                self.state = State.COMPLETED
                logging.info("State: Completed - Vehicle left ROI.")
                return "EXIT"

        # STATE: COMPLETED -> IDLE (Reset)
        elif self.state == State.COMPLETED:
            self.reset()
            return "RESET"

        return self.state

    def is_illegal_event(self):
        """Checks if the total duration meets the dumping criteria."""
        if self.entry_time is None:
            return False
        
        total_duration = time.time() - self.entry_time
        return total_duration >= self.illegal_threshold

    def reset(self):
        self.state = State.IDLE
        self.entry_time = None
        self.stationary_start_time = None
        self.last_motion_time = None
        self.ai_triggered = False
