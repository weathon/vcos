import cv2
import os
import numpy as np

def predict_camera_motion(frame_paths, return_all=False):
    frames = [cv2.imread(path) for path in frame_paths]
    current_location = []
    first_frame = frames[0]
    h, w, _ = first_frame.shape
    offset = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]], dtype=np.float32)
    
    M_cum = np.eye(3, dtype=np.float32)
    
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (prev_gray.shape[1] // 2, prev_gray.shape[0] // 2))
    
    for i in range(1, len(frames)):
        curr_frame = frames[i]
        curr_frame = cv2.resize(curr_frame, (curr_frame.shape[1] // 2, curr_frame.shape[0] // 2))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect good features in the previous frame.
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        if prev_pts is None:
            prev_gray = curr_gray
            continue
        # Track feature points to the current frame.
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        if curr_pts is None:
            prev_gray = curr_gray
            continue
        
        # Select only the valid tracked points.
        idx = np.where(status == 1)[0]
        if len(idx) < 3:
            prev_gray = curr_gray
            continue
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        # Estimate an affine transformation between frames.
        m, _ = cv2.estimateAffine2D(prev_pts, curr_pts)
        if m is None:
            prev_gray = curr_gray
            continue
        
        # Invert the transformation so that we compensate for the motion.
        m_inv = cv2.invertAffineTransform(m)
        # Convert the inverted 2x3 affine matrix to a 3x3 homogeneous matrix.
        m_inv_hom = np.vstack([m_inv, [0, 0, 1]])
        
        # Accumulate the inverse transformation.
        M_cum = M_cum @ m_inv_hom
        warp_matrix = offset @ M_cum
        current_location.append(warp_matrix)
        prev_gray = curr_gray
        
        x_min, x_max = float('inf'), 0
        y_min, y_max = float('inf'), 0
        
        coords = []
        for warp_matrix in current_location:
            corners = np.array([
                [0, 0, 1],
                [w, 0, 1],
                [0, h, 1],
                [w, h, 1]
            ])
            transformed_corners = warp_matrix @ corners.T
            x_coords = transformed_corners[0, :] / transformed_corners[2, :]
            y_coords = transformed_corners[1, :] / transformed_corners[2, :]
            
            
            x_min = min(x_min, x_coords.min())
            x_max = max(x_max, x_coords.max()) - w
            y_min = min(y_min, y_coords.min())
            y_max = max(y_max, y_coords.max()) - h
            coords.append([x_coords, y_coords])
            
    if not return_all:    
        return x_min, x_max, y_min, y_max, np.mean(np.abs([x_min, x_max, y_min, y_max])) > 7
    if return_all:
        return x_min, x_max, y_min, y_max, np.mean(np.abs([x_min, x_max, y_min, y_max])) > 7, np.array(coords)

if __name__ == "__main__":
    import pylab
    from scipy.interpolate import make_smoothing_spline
    path = "/mnt/fastdata/MoCA-Video-Test/snow_leopard_10/Frame"
    files = os.listdir(path) 
    print(predict_camera_motion([os.path.join(path, file) for file in files]))
