"""
Camera Calibration Script - GoPro Intrinsics Estimation
=========================================================
Records a video of a 9x6 chessboard pattern from multiple angles and distances,
then computes precise camera matrix and distortion coefficients.

Usage:
    1. Print a 9x6 chessboard pattern (at least A4 size, measure square size!)
    2. Record a 30-60s video with the pattern visible from many angles
    3. Run: python calibrate_camera.py <video_file.mp4>
    4. Output: camera_calibration.npz (used automatically by process_recording_sync.py)
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time

# ==================== CONFIGURATION ====================
CHESSBOARD_SIZE = (9, 6)  # Inner corners (columns, rows)
SQUARE_SIZE = 0.025       # Physical size of ONE square in meters (measure with ruler!)

# Calibration settings
MAX_FRAMES = 50           # Use best N frames for calibration
MIN_FRAMES = 15           # Minimum required for reliable calibration
FRAME_SKIP = 10           # Process every Nth frame (faster)
MIN_CORNER_QUALITY = 0.01 # Subpixel refinement termination threshold

# Visualization
SHOW_DETECTIONS = True    # Display detected corners in real-time
SAVE_DEBUG_IMAGES = False # Save images with detected corners

# ==================== MAIN SCRIPT ====================

def main():
    if len(sys.argv) < 2:
        print("Usage: python calibrate_camera.py <video_file.mp4>")
        print("\nMake sure to:")
        print("  - Use a printed 9x6 chessboard (inner corners)")
        print("  - Measure the square size and update SQUARE_SIZE in the script")
        print("  - Record from various angles and distances (30-60 seconds)")
        return
    
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print("=" * 60)
    print("Camera Calibration - OpenCV Chessboard Method")
    print("=" * 60)
    print(f"Video: {video_path.name}")
    print(f"Chessboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
    print(f"Square size: {SQUARE_SIZE * 1000:.1f}mm")
    print()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video file")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Resolution: {width}x{height} @ {fps:.2f} FPS")
    print(f"Total frames: {total_frames}")
    print()
    
    # Prepare object points (3D coordinates of chessboard corners in world space)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # Scale by physical square size
    
    # Storage for calibration data
    obj_points = []  # 3D points in world coordinate system
    img_points = []  # 2D points in image plane
    good_frames = []
    
    frame_idx = 0
    detections = 0
    
    # Subpixel corner refinement criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, MIN_CORNER_QUALITY)
    
    print("Detecting chessboard patterns...")
    print("(This may take a minute - processing every %d frames)" % FRAME_SKIP)
    print()
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for faster processing
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, 
            CHESSBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )
        
        if ret:
            # Refine corner positions to subpixel accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            obj_points.append(objp)
            img_points.append(corners_refined)
            good_frames.append(frame_idx)
            detections += 1
            
            print(f"Frame {frame_idx:5d}: ✓ Detected ({detections}/{MAX_FRAMES})", end='\r')
            
            # Visualization
            if SHOW_DETECTIONS:
                vis = frame.copy()
                cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners_refined, ret)
                
                # Status text
                status = f"Detections: {detections}/{MAX_FRAMES}"
                cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Show scaled down for large resolutions
                scale = 0.5 if width > 1920 else 1.0
                if scale != 1.0:
                    vis = cv2.resize(vis, None, fx=scale, fy=scale)
                
                cv2.imshow("Calibration - Press Q to stop early", vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nStopped by user.")
                    break
            
            if SAVE_DEBUG_IMAGES and detections <= 10:
                debug_path = video_path.parent / f"calib_debug_{detections:02d}.jpg"
                cv2.imwrite(str(debug_path), vis)
            
            # Stop when enough frames collected
            if detections >= MAX_FRAMES:
                print(f"\nCollected {MAX_FRAMES} frames - stopping.")
                break
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    print(f"\nProcessing complete in {elapsed:.1f}s")
    print(f"Found {detections} valid chessboard patterns")
    print()
    
    if detections < MIN_FRAMES:
        print(f"ERROR: Not enough detections ({detections} < {MIN_FRAMES})")
        print("Tips:")
        print("  - Ensure the chessboard is clearly visible and in focus")
        print("  - Move the board to different positions and angles")
        print("  - Avoid motion blur (hold steady)")
        print("  - Ensure good lighting (no shadows or glare)")
        return
    
    # ==================== CALIBRATION ====================
    print("Running camera calibration...")
    print("(This may take 10-30 seconds)")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (width, height), None, None
    )
    
    if not ret:
        print("ERROR: Calibration failed!")
        return
    
    # ==================== VALIDATION ====================
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    
    # Compute reprojection error
    total_error = 0
    for i in range(len(obj_points)):
        img_points_reproj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], 
                                                  camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], img_points_reproj, cv2.NORM_L2) / len(img_points_reproj)
        total_error += error
    
    mean_error = total_error / len(obj_points)
    
    print(f"\nCamera Matrix:")
    print(camera_matrix)
    print(f"\nDistortion Coefficients:")
    print(dist_coeffs.flatten())
    print(f"\nReprojection Error: {mean_error:.4f} pixels")
    
    if mean_error > 1.0:
        print("  ⚠ WARNING: High error - calibration may be inaccurate")
        print("    Try recording with more varied angles/distances")
    elif mean_error < 0.3:
        print("  ✓ Excellent calibration quality!")
    else:
        print("  ✓ Good calibration quality")
    
    # Extract focal lengths
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Estimate field of view
    fov_x = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
    
    print(f"\nFocal Length: fx={fx:.1f}, fy={fy:.1f}")
    print(f"Principal Point: cx={cx:.1f}, cy={cy:.1f}")
    print(f"Field of View: {fov_x:.1f}° x {fov_y:.1f}°")
    
    # ==================== SAVE ====================
    output_path = video_path.parent / "camera_calibration.npz"
    
    np.savez(
        str(output_path),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        width=width,
        height=height,
        reprojection_error=mean_error,
        num_frames=detections
    )
    
    print(f"\n✓ Calibration saved to: {output_path.name}")
    print("\nThis file will be automatically loaded by process_recording_sync.py")
    print("when processing videos from the same folder.")
    print("=" * 60)

if __name__ == "__main__":
    main()
