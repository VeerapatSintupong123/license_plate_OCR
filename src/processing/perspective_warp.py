import cv2
import numpy as np

class PerspectiveWarp:
    def __init__(self, target_width=300, target_height=100):
        """
        Thai plates are roughly a 3:1 or 2.5:1 aspect ratio.
        300x100 is a solid resolution for PaddleOCR to process.
        """
        self.target_w = target_width
        self.target_h = target_height

    def order_points(self, pts):
        """
        Coordinates must be in a specific order for OpenCV:
        [top-left, top-right, bottom-right, bottom-left]
        """
        rect = np.zeros((4, 2), dtype="float32")

        # Top-left will have the smallest sum (x+y)
        # Bottom-right will have the largest sum (x+y)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right will have the smallest difference (y-x)
        # Bottom-left will have the largest difference (y-x)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def straighten(self, frame, box_points):
        """
        :param frame: The high-resolution source image
        :param box_points: A list or array of 4 coordinates [(x,y), ...]
        :return: A flattened, cropped image of the plate
        """
        # 1. Ensure points are in the correct order
        pts = np.array(box_points, dtype="float32")
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # 2. Define the destination points (the "flat" rectangle)
        dst = np.array([
            [0, 0],
            [self.target_w - 1, 0],
            [self.target_w - 1, self.target_h - 1],
            [0, self.target_h - 1]
        ], dtype="float32")

        # 3. Calculate the Perspective Transform Matrix (Homography)
        M = cv2.getPerspectiveTransform(rect, dst)

        # 4. Warp the image
        warped = cv2.warpPerspective(frame, M, (self.target_w, self.target_h))

        # 5. Optional: Contrast Enhancement for OCR
        # Convert to grayscale and apply CLAHE to make text pop
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        return enhanced