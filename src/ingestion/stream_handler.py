import cv2
import numpy as np
import time

class StreamHandler:
    def __init__(self, config):
        """
        config should contain:
        - rtsp_sub: URL for low-res monitoring
        - rtsp_main: URL for high-res evidence
        - roi_points: List of (x,y) coordinates for the restricted area
        """
        self.sub_url = config['rtsp_sub']
        self.main_url = config['rtsp_main']
        self.roi_points = np.array(config['roi_points'], dtype=np.int32)
        
        # Connect to the low-res substream immediately
        self.cap_sub = cv2.VideoCapture(self.sub_url)
        
        # Background Subtractor for lightweight motion detection
        # history=500: learns the background over 500 frames
        # detectShadows=True: prevents moving shadows from triggering alerts
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        
        # Kernel for noise reduction in motion mask
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def get_substream(self):
        """Pulls a single frame from the low-res watchman stream."""
        ret, frame = self.cap_sub.read()
        if not ret:
            # Simple auto-reconnect logic
            self.cap_sub.open(self.sub_url)
            return None
        return frame

    def has_motion(self, frame, freeze_bg=False):
        """
        Returns True if meaningful motion is detected inside the ROI.
        
        :param frame:     raw BGR frame from camera
        :param freeze_bg: if True, stop updating the background model
                          (prevents parked vehicle from being absorbed into background)
        """
        if frame is None:
            return False
            
        # 1. Build the ROI mask (used AFTER subtraction, not before)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_points], 255)
        
        # 2. Apply MOG2 to the FULL frame so the background model stays clean
        #    freeze_bg=True → learningRate=0 (model frozen, parked vehicle stays visible)
        #    freeze_bg=False → learningRate=-1 (MOG2 auto-adjusts)
        learning_rate = 0 if freeze_bg else -1
        fgmask = self.fgbg.apply(frame, learningRate=learning_rate)
        
        # 3. NOW restrict to ROI only
        fgmask = cv2.bitwise_and(fgmask, fgmask, mask=mask)
        
        # 4. Clean up noise (wind, small birds, compression artifacts)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        
        # 5. Count only confirmed foreground pixels (255)
        #    Shadows are marked 127 by MOG2 — exclude them
        motion_pixel_count = np.count_nonzero(fgmask == 255)
        roi_pixel_count    = np.count_nonzero(mask)
        
        motion_percentage = (motion_pixel_count / roi_pixel_count) * 100
        return motion_percentage > 20.0

    def get_highres_burst(self, n_frames=10):
        """
        Briefly opens the high-res stream to capture a burst of frames.
        This saves massive bandwidth compared to streaming 4K 24/7.
        """
        cap_main = cv2.VideoCapture(self.main_url)
        burst = []
        
        # Give the camera a moment to initialize the high-res stream
        for _ in range(5): cap_main.grab() 

        for _ in range(n_frames):
            ret, frame = cap_main.read()
            if ret:
                burst.append(frame)
            time.sleep(0.05) # ~20 FPS capture rate for the burst
            
        cap_main.release() # Release network resources immediately
        return burst

    def release(self):
        self.cap_sub.release()
