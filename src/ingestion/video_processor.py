import os
import cv2
import logging
from scenedetect import detect, AdaptiveDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(video_path: str, output_dir: str, resolution: str = "1280x720"):
    """
    Extracts keyframes from a video using Adaptive Scene Detection.
    It takes the middle frame of each detected scene to represent that scene.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_filename = os.path.basename(video_path).split('.')[0]
    
    try:
        logger.info(f"Detecting scenes in video {video_path}...")
        scene_list = detect(video_path, AdaptiveDetector())
        logger.info(f"Detected {len(scene_list)} scenes.")
        
        frames_paths = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        
        target_w, target_h = map(int, resolution.split('x'))
        
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            
            mid_frame = start_frame + (end_frame - start_frame) // 2
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if ret:
                timestamp_sec = mid_frame / fps
                frame_resized = cv2.resize(frame, (target_w, target_h))
                
                out_path = os.path.join(output_dir, f"{video_filename}_scene_{i:04d}.jpg")
                cv2.imwrite(out_path, frame_resized)
                
                frames_paths.append({
                    "path": out_path,
                    "timestamp_sec": timestamp_sec,
                    "scene_index": i
                })
        
        cap.release()
        logger.info(f"Successfully processed {video_path}. {len(frames_paths)} keyframes saved.")
        return frames_paths
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return []

if __name__ == "__main__":
    pass
