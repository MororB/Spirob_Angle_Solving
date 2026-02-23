import cv2
import numpy as np
import math
import csv
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time

# ===================== CONFIGURATION =====================
# Path to the folder containing video file, 'sync_events.csv', 'sensors.csv'
# If None, the script will ask or look in 'runs' for the latest.
RECORDING_FOLDER = None

# Performance Settings for 4K Video
DOWNSCALE_FACTOR = 0.5   # Process at 1080p (0.5 = half resolution, 4x fewer pixels)
PROCESS_EVERY_N_FRAMES = 1  # Process every Nth frame (1 = all frames, 2 = every other)
USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False
CPU_THREADS = max(1, (os.cpu_count() or 1))  # Use all logical cores
USE_OPENCL = True

# Optional export: copy of the video with angles overlaid
EXPORT_ANGLE_VIDEO = True
EXPORT_VIDEO_DOWNSCALED = True  # Write overlay video at processing resolution

# Markers
DICT_ROBOT = cv2.aruco.DICT_4X4_50   # For Robot Joints (IDs 0-5)
DICT_SYNC = cv2.aruco.DICT_4X4_1000  # For Sync (IDs 100+)

# NaN Interpolation
MAX_INTERP_GAP = 15   # Interpolate gaps up to N frames (0.5s at 30fps)

# Plausibility Filter
MAX_JOINT_ANGLE = 60.0  # degrees - physical limit of robot joints, reject outliers beyond this
SYNC_OVERLAP_DIST = 50  # pixels (in downscaled image) - filter sync marker misidentifications

# Robot Config
# IDs of the markers in order of the chain
# 14 Segments -> IDs 0 to 13 -> 13 Joints
CHAIN_IDS = list(range(14))

# 3D Pose Estimation Settings (GoPro - no distortion)
MARKER_SIZE = 0.012  # Marker side length in meters (measure black square!)
SMOOTHING_ALPHA = 0.3  # Temporal smoothing (0 = none, 0.9 = heavy)

# ===================== HELPER FUNCTIONS =====================

def wrap_pi(a):
    """Normalize angle to [-pi, pi)"""
    return (a + math.pi) % (2*math.pi) - math.pi

def marker_angle_from_corners(corners_4x2):
    """Calculate the rotation angle of a marker in the 2D plane (fallback)."""
    p0 = corners_4x2[0]
    p1 = corners_4x2[1]
    return math.atan2(p1[1] - p0[1], p1[0] - p0[0])


def draw_text_with_outline(img, text, org, font, scale, color, thickness=1):
    """Draw readable text with a dark outline."""
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_angle_overlay(frame, frame_idx, angles_dict, angle_cols):
    """Overlay joint angles onto the frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6 if w <= 1920 else 0.9
    thickness = 1 if scale <= 0.7 else 2

    x = 20
    y = 30
    line_h = int(22 * scale) if scale <= 0.7 else int(28 * scale)
    col_w = int(240 * scale)

    header = f"Frame: {frame_idx}"
    draw_text_with_outline(frame, header, (x, y), font, scale, (255, 255, 255), thickness)
    y += line_h + 4

    for col in angle_cols:
        val = angles_dict.get(col, np.nan)
        if val is None or np.isnan(val):
            text_val = "--"
        else:
            text_val = f"{val:+.1f} deg"
        line = f"{col}: {text_val}"
        draw_text_with_outline(frame, line, (x, y), font, scale, (255, 255, 255), thickness)
        y += line_h
        if y > h - 20:
            x += col_w
            y = 30 + line_h + 4


def get_marker_obj_points(marker_size):
    """3D corner coordinates in marker coordinate system (Z=0 plane)."""
    half = marker_size / 2.0
    return np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float64)


def estimate_camera_matrix(frame_width, frame_height):
    """
    Estimate camera intrinsics for GoPro (no calibration file).
    Assumes ~linear lens (GoPro Linear mode / already corrected).
    Focal length approximated from typical GoPro FOV.
    """
    # GoPro Linear mode: ~90° horizontal FOV (adjusted for better accuracy)
    # If angles are still wrong, try different FOV values: 80-100°
    # f = (w/2) / tan(hfov/2)
    hfov_rad = math.radians(45)  # half of 90° (try 40-50° if still wrong)
    fx = (frame_width / 2.0) / math.tan(hfov_rad)
    fy = fx  # Square pixels
    cx = frame_width / 2.0
    cy = frame_height / 2.0
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)


def load_camera_calibration(work_dir, video_width, video_height):
    """
    Load camera calibration from camera_calibration.npz if available.
    Returns (camera_matrix, dist_coeffs, source_description)
    """
    # Search in work_dir and parent directory
    calib_paths = [
        work_dir / "camera_calibration.npz",
        Path("camera_calibration.npz"),
        Path("Full_Spirob_new_pcb") / "camera_calibration.npz"
    ]
    
    for calib_path in calib_paths:
        if calib_path.exists():
            try:
                data = np.load(str(calib_path))
                cam_matrix = data['camera_matrix']
                dist_coeffs = data['dist_coeffs']
                calib_width = int(data.get('width', video_width))
                calib_height = int(data.get('height', video_height))
                reproj_error = float(data.get('reprojection_error', -1))
                
                # Verify resolution match
                if calib_width != video_width or calib_height != video_height:
                    print(f"[Warning] Calibration resolution mismatch:")
                    print(f"          Calibration: {calib_width}x{calib_height}")
                    print(f"          Video:       {video_width}x{video_height}")
                    print(f"          Scaling camera matrix...")
                    
                    # Scale camera matrix to match video resolution
                    scale_x = video_width / calib_width
                    scale_y = video_height / calib_height
                    cam_matrix = cam_matrix.copy()
                    cam_matrix[0, 0] *= scale_x  # fx
                    cam_matrix[1, 1] *= scale_y  # fy
                    cam_matrix[0, 2] *= scale_x  # cx
                    cam_matrix[1, 2] *= scale_y  # cy
                
                source = f"Calibrated (error: {reproj_error:.4f}px)" if reproj_error >= 0 else "Calibrated"
                return cam_matrix, dist_coeffs, source
            
            except Exception as e:
                print(f"[Warning] Failed to load calibration from {calib_path}: {e}")
                continue
    
    # Fallback to estimation
    cam_matrix = estimate_camera_matrix(video_width, video_height)
    dist_coeffs = np.zeros(5)  # No distortion for GoPro Linear
    return cam_matrix, dist_coeffs, "Estimated (FOV-based)"


def estimate_marker_rotation(corner_2d, marker_size, cam_matrix, dist_coeffs, prev_R=None):
    """
    Robust 3D pose estimation with ambiguity resolution.
    Uses solvePnPGeneric + IPPE_SQUARE to get both possible poses,
    then selects the geometrically correct one.
    """
    obj_pts = get_marker_obj_points(marker_size)
    img_pts = corner_2d.reshape(-1, 1, 2).astype(np.float64)

    try:
        n_solutions, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
            obj_pts, img_pts, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
    except Exception:
        return None

    if n_solutions == 0:
        return None

    candidates = []
    for i in range(n_solutions):
        R, _ = cv2.Rodrigues(rvecs[i])
        err = reproj_errors[i][0]
        score = 0

        marker_in_front = tvecs[i][2, 0] > 0.01
        normal_toward_camera = R[2, 2] < -0.1

        if marker_in_front and normal_toward_camera:
            score += 1000

        if prev_R is not None:
            diff = np.linalg.norm(R - prev_R, 'fro')
            score += max(0, 50 - diff * 20)

        score -= err * 10
        candidates.append({'R': R, 'score': score})

    best = max(candidates, key=lambda x: x['score'])
    return best['R']


def compute_joint_angle_3d(R_a, R_b):
    """
    Compute joint angle from two 3D rotation matrices.
    Projects both marker X-axes onto a common plane (perpendicular to
    mean normal) and computes the signed angle between projections.
    This is independent of camera viewing angle.
    """
    x_a = R_a[:, 0]
    x_b = R_b[:, 0]
    z_a = R_a[:, 2]
    z_b = R_b[:, 2]

    z_mean = (z_a + z_b) / 2
    norm = np.linalg.norm(z_mean)
    if norm < 1e-6:
        return np.nan
    z_mean = z_mean / norm

    x_a_proj = x_a - np.dot(x_a, z_mean) * z_mean
    x_b_proj = x_b - np.dot(x_b, z_mean) * z_mean

    norm_a = np.linalg.norm(x_a_proj)
    norm_b = np.linalg.norm(x_b_proj)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return np.nan
    x_a_proj = x_a_proj / norm_a
    x_b_proj = x_b_proj / norm_b

    cross = np.cross(x_a_proj, x_b_proj)
    dot = np.dot(x_a_proj, x_b_proj)
    angle = math.atan2(np.dot(cross, z_mean), dot)

    return (math.degrees(angle) + 180) % 360 - 180  # wrap to [-180, 180)

def find_latest_recording():
    """Finds the underlying directory of the most recent recording."""
    base = Path("runs")
    if not base.exists():
        return None
    subdirs = [x for x in base.iterdir() if x.is_dir()]
    if not subdirs:
        return None
    # Sort by name (timestamp format ensures correct order)
    return sorted(subdirs)[-1]

def find_video_file(work_dir):
    """Find any .mp4 file in the directory."""
    mp4_files = list(work_dir.glob("*.mp4"))
    if not mp4_files:
        return None
    if len(mp4_files) > 1:
        print(f"[Info] Multiple video files found, using: {mp4_files[0].name}")
    return mp4_files[0]

def interpolate_nan_gaps(values, max_gap):
    """
    Linearly interpolate NaN gaps that are <= max_gap frames long.
    Longer gaps are left as NaN.
    """
    result = values.copy()
    n = len(result)
    i = 0
    while i < n:
        if np.isnan(result[i]):
            # Found start of a NaN gap
            gap_start = i
            while i < n and np.isnan(result[i]):
                i += 1
            gap_end = i  # first non-NaN after gap
            gap_len = gap_end - gap_start
            
            # Only interpolate if gap is small enough AND we have values on both sides
            if gap_len <= max_gap and gap_start > 0 and gap_end < n:
                v_before = result[gap_start - 1]
                v_after = result[gap_end]
                for j in range(gap_start, gap_end):
                    t = (j - gap_start + 1) / (gap_len + 1)
                    result[j] = v_before + t * (v_after - v_before)
        else:
            i += 1
    return result

# ===================== SYNC LOGIC =====================

@dataclass
class SyncEvent:
    frame_idx: int
    sync_id: int

def estimate_time_model(video_sync_events: List[SyncEvent], csv_sync_path: Path, video_fps: float = 30.0):
    """
    Correlates video frame indices with PC timestamps using Linear Regression.
    Returns a function: frame_to_time_ns(frame_idx) -> ns
    """
    if not csv_sync_path.exists():
        print(f"[Error] Sync log not found: {csv_sync_path}")
        return None

    # 1. Load CSV Events
    # Format: t_pc_ns, sync_id, state, description
    csv_events = []
    
    print(f"[Sync] Reading CSV: {csv_sync_path}")
    try:
        # utf-8-sig handles BOM if present (common on Windows)
        with open(csv_sync_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            
            # Normalize headers (strip whitespace)
            if reader.fieldnames:
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
            
            print(f"  - Headers found: {reader.fieldnames}")
            
            for row in reader:
                # Normalize values
                row = {k: v.strip() if v else v for k, v in row.items()}
                
                state_val = row.get('state', '')
                desc_val = row.get('description', '')
                
                # Check 'state' column for MARKER_UPDATE/SWITCH (generated by record.py)
                # Also check if 'description' starts with ArUco just in case layout varies
                is_valid = (state_val == "MARKER_UPDATE") or (state_val == "SWITCH") or (state_val == "1") or (state_val == "0")
                
                if is_valid:
                    try:
                        t = int(row['t_pc_ns'])
                        sid = int(row['sync_id'])
                        csv_events.append((t, sid))
                    except (ValueError, KeyError) as e:
                        print(f"  - Skipped row (parsing error): {row} -> {e}")
                        continue
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return None
    
    if not csv_events:
        print("[Error] No valid events (MARKER_UPDATE/SWITCH) found in sync_events.csv")
        return None

    valid_csv_ids = set(sid for _, sid in csv_events)
    print(f"  - Loaded {len(csv_events)} events. Unique IDs: {sorted(list(valid_csv_ids))}")

    # 2. Filter Video Events (Whitelist)
    # This fixes the issue of "too many IDs" (false positives like 999)
    filtered_video_events = [e for e in video_sync_events if e.sync_id in valid_csv_ids]
    
    removed_count = len(video_sync_events) - len(filtered_video_events)
    if removed_count > 0:
        print(f"[Sync] Filtered out {removed_count} false positive marker detections (IDs not in CSV).")
    
    if not filtered_video_events:
        print("[Error] No video marker detections match the IDs in the CSV!")
        return None

    # 3. Match Events
    # Strategy: Use MEDIAN frame where each ID appears (robust against false positives)
    all_frames_per_id = {}  # ID -> list of frame indices
    first_id_seen_in_video = None
    
    # Collect ALL frames for each ID
    for e in filtered_video_events:
        if first_id_seen_in_video is None:
            first_id_seen_in_video = e.sync_id
            
        if e.sync_id not in all_frames_per_id:
            all_frames_per_id[e.sync_id] = []
        all_frames_per_id[e.sync_id].append(e.frame_idx)
    
    # Calculate median frame for each ID
    id_to_median_frame = {}
    for sync_id, frames in all_frames_per_id.items():
        median_frame = int(np.median(frames))
        id_to_median_frame[sync_id] = median_frame
    
    # CRITICAL: Ignore first ID transition
    if first_id_seen_in_video is not None:
        if first_id_seen_in_video in id_to_median_frame:
            print(f"[Sync] Ignoring first detected ID {first_id_seen_in_video} to ensure true transition timing.")
            del id_to_median_frame[first_id_seen_in_video]
    
    matches_x_frame = []
    matches_y_time = []
    
    print("\n[Sync] Matching Events (Common IDs found in both):")
    print(f"{'ID':<5} | {'Video Frame':<12} | {'PC Timestamp':<20}")
    print("-" * 45)
    
    msg_printed_count = 0
    for t_ns, sid in csv_events:
        if sid in id_to_median_frame:
            frame = id_to_median_frame[sid]
            matches_x_frame.append(frame)
            matches_y_time.append(t_ns)
            print(f"{sid:<5} | {frame:<12} | {t_ns:<20}")
            msg_printed_count += 1
    
    print(f"[Sync] Found {len(matches_x_frame)} valid matches for regression.")

    if len(matches_x_frame) < 1:
        print("[Error] No sync matches found after alignment! Cannot synchronize.")
        return None

    if len(matches_x_frame) == 1:
        print(f"[Warning] Only 1 sync match found. Using single-point sync with FPS={video_fps:.2f}.")
        f0 = matches_x_frame[0]
        t0 = matches_y_time[0]
        # Slope = ns per frame = 1e9 / FPS
        slope = 1_000_000_000.0 / video_fps
        return lambda f: t0 + (f - f0) * slope

    # 4. Linear Regression
    try:
        slope, intercept = np.polyfit(matches_x_frame, matches_y_time, 1)
        fps_est = 1e9 / slope
        print(f"\n[Sync] Model Fit Details:")
        print(f"       Slope (ns/frame) : {slope:.2f}")
        print(f"       Intercept (ns)   : {intercept:.2f}")
        print(f"       Estimated FPS    : {fps_est:.4f}")
        
        # Check Frame 0 Time
        t_frame_0 = intercept
        print(f"       Frame 0 Time     : {t_frame_0:.0f} (PC Time)")
        print(f"       First Match Time : {matches_y_time[0] if matches_y_time else 'N/A'}")
        
        return lambda f: slope * f + intercept
    except Exception as e:
        print(f"[Error] Regression failed: {e}")
        return None

# ===================== MAIN PROCESSING =====================

def main():
    # OpenCV performance tuning
    try:
        cv2.setUseOptimized(True)
        if CPU_THREADS > 0:
            cv2.setNumThreads(CPU_THREADS)
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(USE_OPENCL)
    except Exception:
        pass

    # 1. Setup Paths
    if len(sys.argv) > 1:
        work_dir = Path(sys.argv[1])
    elif RECORDING_FOLDER:
        work_dir = Path(RECORDING_FOLDER)
    else:
        print("Searching for latest recording...")
        work_dir = find_latest_recording()
    
    if not work_dir or not work_dir.exists():
        print(f"Error: Could not find recording directory: {work_dir}")
        return

    # Find video file automatically
    video_path = find_video_file(work_dir)
    if not video_path:
        print(f"Error: No .mp4 video file found in: {work_dir}")
        print("Please ensure a video file (.mp4) is placed in the folder.")
        return
    
    print(f"[Video] Using: {video_path.name}")
    
    sync_csv_path = work_dir / "sync_events.csv"
    output_csv_path = work_dir / "labels.csv"

    print(f"Processing: {work_dir}")
    print("-" * 50)

    # 2. ArUco Setup - Optimized for 4K with small markers
    # IMPORTANT: Use TWO separate detectors!
    # DICT_4X4_50 for robot markers (IDs 0-13) - smaller dictionary = better error correction
    # DICT_4X4_1000 for sync markers (IDs 100+) - needs larger dictionary for high IDs
    aruco_dict_robot = cv2.aruco.getPredefinedDictionary(DICT_ROBOT)
    aruco_dict_sync  = cv2.aruco.getPredefinedDictionary(DICT_SYNC)
    
    aruco_params = cv2.aruco.DetectorParameters()
    
    # --- Optimized for small markers in high-res ---
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_params.cornerRefinementMaxIterations = 30  # More iterations for accuracy
    
    # Smaller adaptive threshold windows for tiny markers
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 15  # Reduced from 30 for speed
    aruco_params.adaptiveThreshWinSizeStep = 4   # Larger steps = fewer iterations
    
    # Relaxed constraints for small markers
    aruco_params.minMarkerPerimeterRate = 0.01  # Allow smaller markers (was 0.03)
    aruco_params.maxMarkerPerimeterRate = 4.0
    
    # Error correction
    aruco_params.errorCorrectionRate = 0.6
    
    # Speed optimizations
    aruco_params.perspectiveRemovePixelPerCell = 4  # Reduce from default 8
    aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.1

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector_robot = cv2.aruco.ArucoDetector(aruco_dict_robot, aruco_params)
        detector_sync  = cv2.aruco.ArucoDetector(aruco_dict_sync, aruco_params)
    else:
        detector_robot = None
        detector_sync  = None

    # 3. Video Processing
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    print(f"OpenCV threads: {CPU_THREADS}, OpenCL: {USE_OPENCL}")
    if USE_CUDA:
        print("CUDA acceleration: ENABLED (GPU preprocessing)")
        cuda_available = True
    else:
        print("CUDA acceleration: DISABLED")
        cuda_available = False
    if DOWNSCALE_FACTOR != 1.0:
        new_w = int(width * DOWNSCALE_FACTOR)
        new_h = int(height * DOWNSCALE_FACTOR)
        print(f"Downscaling to {new_w}x{new_h} for processing ({DOWNSCALE_FACTOR*100:.0f}%)")
    if PROCESS_EVERY_N_FRAMES > 1:
        print(f"Processing every {PROCESS_EVERY_N_FRAMES} frame(s)")
    
    # CUDA setup for image preprocessing
    gpu_frame = None
    gpu_resized = None
    if cuda_available and DOWNSCALE_FACTOR != 1.0:
        print("[CUDA] Initializing GPU memory for video preprocessing...")

    video_sync_events = []
    frame_data = []
    angle_cols = [f"theta_{i+1}" for i in range(len(CHAIN_IDS) - 1)]

    # Angle overlay video writer
    angle_video_path = None
    angle_writer = None
    if EXPORT_ANGLE_VIDEO:
        angle_video_path = work_dir / f"{video_path.stem}_angles.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if EXPORT_VIDEO_DOWNSCALED and DOWNSCALE_FACTOR != 1.0:
            out_w = int(width * DOWNSCALE_FACTOR)
            out_h = int(height * DOWNSCALE_FACTOR)
        else:
            out_w = width
            out_h = height
        angle_writer = cv2.VideoWriter(str(angle_video_path), fourcc, fps, (out_w, out_h))
        if not angle_writer.isOpened():
            print(f"[Warning] Could not open video writer: {angle_video_path}")
            angle_writer = None
    
    # 3D Pose: Load camera calibration (or estimate if not available)
    cam_matrix, dist_coeffs, calib_source = load_camera_calibration(work_dir, width, height)
    print(f"[3D Pose] Camera Matrix: {calib_source}")
    print(f"          fx={cam_matrix[0,0]:.0f}, fy={cam_matrix[1,1]:.0f}")
    print(f"          cx={cam_matrix[0,2]:.0f}, cy={cam_matrix[1,2]:.0f}")
    if np.any(dist_coeffs != 0):
        print(f"          Distortion: {dist_coeffs.flatten()}")
    print(f"[3D Pose] Marker size: {MARKER_SIZE*1000:.2f}mm, Smoothing: {SMOOTHING_ALPHA:.2f}")
    
    # Downscale inverse factor for rescaling marker coords
    scale_inv = 1.0 / DOWNSCALE_FACTOR if DOWNSCALE_FACTOR != 1.0 else 1.0
    
    # Temporal state for smoothing
    prev_rotations = {}     # marker_id -> R (previous frame rotation)
    prev_joint_angles = {}  # joint_key -> angle (previous frame)
    last_display_angles = {}  # angle label -> last valid value for overlay
    
    frame_idx = 0
    processed_count = 0
    start_time = time.time()
    
    while True:
        ok, frame = cap.read()
        if not ok: 
            break

        frame_raw = frame
        
        # Skip frames if configured
        if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
            # Still store empty data for skipped frames
            current_data = {"frame_idx": frame_idx}
            for i in range(len(CHAIN_IDS) - 1):
                current_data[f"theta_{i+1}"] = np.nan
            frame_data.append(current_data)

            if angle_writer is not None:
                if EXPORT_VIDEO_DOWNSCALED and DOWNSCALE_FACTOR != 1.0:
                    overlay_frame = cv2.resize(
                        frame_raw, None, fx=DOWNSCALE_FACTOR, fy=DOWNSCALE_FACTOR,
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    overlay_frame = frame_raw
                display_angles = {}
                for col in angle_cols:
                    if col in last_display_angles:
                        display_angles[col] = last_display_angles[col]
                    else:
                        display_angles[col] = np.nan
                draw_angle_overlay(overlay_frame, frame_idx, display_angles, angle_cols)
                angle_writer.write(overlay_frame)

            frame_idx += 1
            continue
        
        # CUDA-accelerated preprocessing (up to 10x faster than CPU)
        if cuda_available and DOWNSCALE_FACTOR != 1.0:
            # Upload frame to GPU
            gpu_frame = cv2.cuda.GpuMat()
            gpu_frame.upload(frame_raw)
            
            # GPU resize (much faster than cv2.resize)
            gpu_resized = cv2.cuda.resize(gpu_frame, 
                                         (int(width * DOWNSCALE_FACTOR), 
                                          int(height * DOWNSCALE_FACTOR)))
            
            # GPU color conversion
            gpu_gray = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2GRAY)
            
            # Download back to CPU for ArUco detection
            frame = gpu_resized.download()
            gray = gpu_gray.download()
        else:
            # CPU fallback
            if DOWNSCALE_FACTOR != 1.0:
                frame = cv2.resize(frame_raw, None, fx=DOWNSCALE_FACTOR, fy=DOWNSCALE_FACTOR, 
                                 interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers with TWO separate detectors
        # 1. Robot markers (DICT_4X4_50) - smaller dict = more reliable detection
        if detector_robot:
            corners_r, ids_r, _ = detector_robot.detectMarkers(gray)
        else:
            corners_r, ids_r, _ = cv2.aruco.detectMarkers(gray, aruco_dict_robot, parameters=aruco_params)
        
        # 2. Sync markers (DICT_4X4_1000) - needs larger dict for IDs 100+
        if detector_sync:
            corners_s, ids_s, _ = detector_sync.detectMarkers(gray)
        else:
            corners_s, ids_s, _ = cv2.aruco.detectMarkers(gray, aruco_dict_sync, parameters=aruco_params)
        
        # Storage for this frame
        current_data = {"frame_idx": frame_idx}
        detected_rotations = {}  # marker_id -> R (3x3 rotation matrix)
        
        # Build set of sync marker centers (from DICT_4X4_1000 detector)
        # Used to filter out sync markers that DICT_4X4_50 misidentifies as robot markers
        sync_centers = []
        if ids_s is not None:
            for c, mid in zip(corners_s, ids_s.flatten()):
                mid = int(mid)
                if mid >= 100:
                    video_sync_events.append(SyncEvent(frame_idx, mid))
                    center = c[0].mean(axis=0)
                    sync_centers.append(center)
        
        # Process robot markers with 3D pose estimation
        if ids_r is not None:
            for c, mid in zip(corners_r, ids_r.flatten()):
                mid = int(mid)
                if mid in CHAIN_IDS:
                    # Check if this detection overlaps with a known sync marker
                    robot_center = c[0].mean(axis=0)
                    is_sync_misid = False
                    for sc in sync_centers:
                        dist = np.sqrt((robot_center[0] - sc[0])**2 + (robot_center[1] - sc[1])**2)
                        if dist < SYNC_OVERLAP_DIST:
                            is_sync_misid = True
                            break
                    
                    if is_sync_misid:
                        continue  # Skip - this is actually the sync marker on screen
                    
                    # 3D Pose Estimation: rescale corners to ORIGINAL resolution
                    corners_original = c[0] * scale_inv
                    prev_R = prev_rotations.get(mid, None)
                    R = estimate_marker_rotation(corners_original, MARKER_SIZE, cam_matrix, dist_coeffs, prev_R)
                    
                    if R is not None:
                        # Temporal smoothing of rotation matrix
                        if prev_R is not None and SMOOTHING_ALPHA > 0:
                            R = (1 - SMOOTHING_ALPHA) * R + SMOOTHING_ALPHA * prev_R
                            U, _, Vt = np.linalg.svd(R)
                            R = U @ Vt  # Re-orthogonalize
                        
                        detected_rotations[mid] = R
                        prev_rotations[mid] = R
        
        # Calculate Joint Angles from 3D rotations
        for i in range(len(CHAIN_IDS) - 1):
            id_start = CHAIN_IDS[i]
            id_end = CHAIN_IDS[i+1]
            label = f"theta_{i+1}"
            
            val = np.nan
            if id_start in detected_rotations and id_end in detected_rotations:
                val = compute_joint_angle_3d(detected_rotations[id_start], detected_rotations[id_end])
                
                if not np.isnan(val):
                    # Temporal smoothing of joint angles
                    joint_key = f"{id_start}_{id_end}"
                    if joint_key in prev_joint_angles and SMOOTHING_ALPHA > 0:
                        val = (1 - SMOOTHING_ALPHA) * val + SMOOTHING_ALPHA * prev_joint_angles[joint_key]
                    prev_joint_angles[joint_key] = val
                    
                    # Plausibility filter: reject physically impossible joint angles
                    if abs(val) > MAX_JOINT_ANGLE:
                        val = np.nan
            
            current_data[label] = val
        
        frame_data.append(current_data)
        processed_count += 1

        if angle_writer is not None:
            if EXPORT_VIDEO_DOWNSCALED and DOWNSCALE_FACTOR != 1.0:
                overlay_frame = frame
            else:
                overlay_frame = frame_raw
            display_angles = {}
            for col in angle_cols:
                val = current_data.get(col, np.nan)
                if val is not None and not np.isnan(val):
                    last_display_angles[col] = val
                display_angles[col] = last_display_angles.get(col, np.nan)
            draw_angle_overlay(overlay_frame, frame_idx, display_angles, angle_cols)
            angle_writer.write(overlay_frame)
        
        # Progress with speed indication
        if frame_idx % 100 == 0 and frame_idx > 0:
            elapsed = time.time() - start_time
            fps_actual = processed_count / elapsed
            eta = (total_frames - frame_idx) / fps_actual / PROCESS_EVERY_N_FRAMES
            print(f"Frame {frame_idx}/{total_frames} ({fps_actual:.1f} fps, ETA: {eta:.0f}s)  ", end='\r')
        
        frame_idx += 1

    cap.release()
    if angle_writer is not None:
        angle_writer.release()
        print(f"[Video] Angle overlay saved: {angle_video_path.name}")
    elapsed = time.time() - start_time
    fps_actual = processed_count / elapsed
    print(f"\nProcessing complete in {elapsed:.1f}s ({fps_actual:.1f} fps)")
    print(f"  - Total Frames: {frame_idx}")
    print(f"  - Processed: {processed_count}")
    print(f"  - Sync Marker Sightings: {len(video_sync_events)}")
    
    unique_video_ids = sorted(list(set(e.sync_id for e in video_sync_events)))
    print(f"  - Unique Sync IDs seen in video: {unique_video_ids}")

    # --- NaN Interpolation Pass ---
    # Interpolate small gaps per-column to preserve data
    if MAX_INTERP_GAP > 0:
        angle_cols = [f"theta_{i+1}" for i in range(len(CHAIN_IDS) - 1)]
        print(f"\n[Interpolation] Filling NaN gaps <= {MAX_INTERP_GAP} frames...")
        for col in angle_cols:
            values = np.array([row.get(col, np.nan) for row in frame_data], dtype=float)
            nan_before = np.isnan(values).sum()
            values = interpolate_nan_gaps(values, MAX_INTERP_GAP)
            nan_after = np.isnan(values).sum()
            filled = nan_before - nan_after
            if filled > 0:
                print(f"  {col}: {nan_before} NaN -> {nan_after} NaN (filled {filled})")
            # Write back
            for i, row in enumerate(frame_data):
                row[col] = values[i]

    if not video_sync_events:
        print("\n[WARNING] No Sync Markers (ID >= 100) were detected in the video!")
        print("Possible reasons:")
        print("  1. The 'Sync Tracker' window was not visible in the video recording.")
        print("  2. The markers were too small, blurry, or overexposed.")
        print("  3. Check if the video file is correct.")
        
        # Debug: Print what WAS found
        unique_all_ids = set()
        for frame in frame_data:
            # We didn't store all IDs in frame_data, but we know robot IDs were found
            pass
        print("  (Robot markers 0-13 seem to be detected based on angles being calculated)")

    # 5. Build Time Model
    print("\nBuilding synchronization model...")
    time_fn = estimate_time_model(video_sync_events, sync_csv_path, video_fps=fps)

    if time_fn is None:
        print("CRITICAL: Synchronization failed. CSV will contain only Frame Indices.")
    
    # 6. Export Data
    print(f"Writing results to {output_csv_path}...")
    
    # Determine columns
    # We have 'frame_idx', 'ts_estimated_ns', then 'theta_1'...'theta_N'
    fieldnames = ["frame_idx", "t_estimated_ns"] + angle_cols
    
    # If no sync, matching columns will be empty
    start_time_ns = 0 
    # Just a placeholder, t_estimated is the important one
    
    with open(output_csv_path, 'w', newline='') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in frame_data:
            fidx = row["frame_idx"]
            
            # Calculate Time
            t_est = ""
            if time_fn:
                t_est = int(time_fn(fidx))
            
            # Prepare row
            out_row = {
                "frame_idx": fidx,
                "t_estimated_ns": t_est
            }
            # Fill angles
            for col in angle_cols:
                out_row[col] = f"{row.get(col, np.nan):.4f}"
            
            writer.writerow(out_row)

    print("Done!")

if __name__ == "__main__":
    main()
