import time
import logging

class State:
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    STATIONARY = "STATIONARY"
    COMPLETED = "COMPLETED"

class StateMachine:
    def __init__(
        self,
        stationary_threshold  = 10,
        illegal_threshold     = 60,
        entry_confirm_frames  = 3,
        exit_confirm_frames   = 5,
        cooldown_duration     = 5,
    ):
        """
        :param stationary_threshold: Seconds of no-motion before triggering AI.
        :param illegal_threshold: Seconds in zone before flagging as illegal dumping.
        """

        self.stationary_threshold = stationary_threshold
        self.illegal_threshold = illegal_threshold
        self.entry_confirm_frames = entry_confirm_frames
        self.exit_confirm_frames = exit_confirm_frames
        self.cooldown_duration = cooldown_duration
      
        self.state = State.IDLE
        self.entry_time = None
        self.stationary_start_time = None
        self.last_motion_time = None
        self.ai_triggered = False

        self._consecutive_detections = 0   # fix 2
        self._consecutive_absences   = 0   # fix 3
        self._cooldown_start         = None  # fix 4


    def process_motion(self, detected: bool):
        """
        The main transition logic called every frame.
        Returns the current system status to main.py.
        """
        current_time = time.time()

        # STATE: IDLE -> TRACKING
        if self.state == State.IDLE:
            if detected:
                self._consecutive_detections += 1
                if self._consecutive_detections >= self.entry_confirm_frames:
                    self.state = State.TRACKING
                    self.entry_time = current_time
                    self.last_motion_time = current_time
                    self._consecutive_detections = 0
                    logging.info("State: TRACKING - Vehicle entered ROI.")
                    return "ENTRY"
            else:
                self._consecutive_detections = 0

        # STATE: TRACKING -> STATIONARY
        elif self.state == State.TRACKING:
            if detected:
                self.last_motion_time = current_time
            elif self.last_motion_time is not None: 
                # If motion stops, check how long it's been still
                still_duration = current_time - self.last_motion_time
                if still_duration >= self.stationary_threshold:
                    self.state = State.STATIONARY
                    self.stationary_start_time = current_time
                    logging.info(f"State: Stationary - Vehicle stopped for {self.stationary_threshold}s.")
                    return "STATIONARY_TRIGGER"
            else:
                self.last_motion_time = current_time

        # STATE: STATIONARY -> COMPLETED (Vehicle Leaves)
        elif self.state == State.STATIONARY:
            if detected:
                self.last_motion_time = current_time
                self._consecutive_absences = 0
            else:
                self._consecutive_absences += 1
                if self._consecutive_absences >= self.exit_confirm_frames:
                    self.state = State.COMPLETED
                    logging.info("State: COMPLETED - Vehicle left ROI.")
                    return "EXIT"

        # STATE: COMPLETED -> IDLE (Reset)
        elif self.state == State.COMPLETED:
            if self._cooldown_start is None:
                self._cooldown_start = current_time
            if current_time - self._cooldown_start >= self.cooldown_duration:
                self.reset()
                return "RESET"

        return self.state

    def is_illegal_event(self):
        """Checks if the total duration meets the dumping criteria."""
        if self.entry_time is None:
            return False
        
        return (time.time() - self.entry_time) >= self.illegal_threshold

    def reset(self):
        self.state = State.IDLE
        self.entry_time = None
        self.stationary_start_time = None
        self.last_motion_time = None
        self.ai_triggered = False
        self._consecutive_detections = 0
        self._consecutive_absences = 0
        self._cooldown_start = None
        logging.info("State: IDLE - Reset complete.")
