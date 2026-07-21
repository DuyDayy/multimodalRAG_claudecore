import os
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_and_transcribe_audio(video_path: str) -> list:
    """
    [TURBO MODE] Đã vô hiệu hóa Whisper để tăng tốc độ tối đa cho V3C dataset.
    """
    return []

def process_video(video_path: str, output_dir: str, resolution: str = "1280x720", step_seconds: float = 3.0, audio_segments: list = None):
    """
    [TURBO MODE] Extracts keyframes from a video using Uniform Sampling.
    Nếu audio_segments được cung cấp, nó sẽ tự động map câu thoại vào bức ảnh dựa trên mốc thời gian.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_filename = os.path.basename(video_path).split('.')[0]
    
    try:
        logger.info(f"Đang chạy Turbo Mode (Uniform Sampling {step_seconds}s) cho video {video_path}...")
        
        frames_paths = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        
        # Tính toán tổng thời lượng
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_sec = frame_count / fps
        
        target_w, target_h = map(int, resolution.split('x'))
        
        current_sec = 0.0
        scene_idx = 0
        
        while current_sec < duration_sec:
            # Nhảy cóc thẳng tới mốc giây mục tiêu
            cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000)
            ret, frame = cap.read()
            
            if ret:
                frame_resized = cv2.resize(frame, (target_w, target_h))
                out_path = os.path.join(output_dir, f"{video_filename}_scene_{scene_idx:04d}.jpg")
                cv2.imwrite(out_path, frame_resized)
                
                # Tìm câu thoại trùng khớp với mốc thời gian của khung hình
                frame_text = ""
                if audio_segments:
                    for seg in audio_segments:
                        # Nếu frame rơi vào đoạn thời gian của câu thoại này
                        if seg["start"] <= current_sec <= seg["end"]:
                            frame_text = seg["text"]
                            break
                
                frames_paths.append({
                    "path": out_path,
                    "timestamp_sec": current_sec,
                    "scene_index": scene_idx,
                    "audio_transcript": frame_text
                })
            else:
                break
                
            current_sec += step_seconds
            scene_idx += 1
            
        cap.release()
        logger.info(f"Successfully processed {video_path} (Turbo Mode). {len(frames_paths)} keyframes saved.")
        return frames_paths
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return []

if __name__ == "__main__":
    pass
