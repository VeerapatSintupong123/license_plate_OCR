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

    def has_motion(self, frame):
        """
        Determines if there is a vehicle-sized object inside the ROI.
        """
        if frame is None:
            return False
            
        # 1. Mask the ROI: Only look at the restricted zone
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_points], 255)
        roi_area = cv2.bitwise_and(frame, frame, mask=mask)
        
        # 2. Apply Background Subtraction to the ROI
        fgmask = self.fgbg.apply(roi_area)
        
        # 3. Clean up the noise (removes wind in trees, small birds, etc.)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        
        # 4. Calculate if the 'blob' is big enough to be a car
        motion_pixel_count = np.count_nonzero(fgmask)
        roi_pixel_count = np.count_nonzero(mask)
        
        motion_percentage = (motion_pixel_count / roi_pixel_count) * 100
        
        # If > 3% of the restricted zone changed, something is there
        return motion_percentage > 3.0

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
