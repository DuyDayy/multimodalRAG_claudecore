import os
import cv2
import logging
import subprocess
try:
    import whisper
except ImportError:
    whisper = None

from scenedetect import detect, AdaptiveDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_and_transcribe_audio(video_path: str) -> list:
    """
    Trích xuất âm thanh bằng ffmpeg và bóc băng bằng Whisper.
    Trả về danh sách các segments [{"start": float, "end": float, "text": str}].
    """
    if not whisper:
        logger.warning("Thư viện openai-whisper chưa được cài đặt. Bỏ qua bóc băng âm thanh.")
        return []
        
    audio_path = video_path.rsplit('.', 1)[0] + ".wav"
    try:
        if not os.path.exists(audio_path):
            logger.info(f"Đang trích xuất audio từ {video_path} bằng ffmpeg...")
            cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        logger.info(f"Đang bóc băng âm thanh bằng Whisper (Base model)...")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        logger.info("Hoàn tất bóc băng âm thanh.")
        return result.get("segments", [])
    except Exception as e:
        logger.error(f"Lỗi khi bóc băng âm thanh: {e}")
        return []

def process_video(video_path: str, output_dir: str, resolution: str = "1280x720"):
    """
    Extracts keyframes from a video using Adaptive Scene Detection.
    It takes the middle frame of each detected scene to represent that scene.
    Cũng đồng thời bóc băng âm thanh và gán text cho từng Scene.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_filename = os.path.basename(video_path).split('.')[0]
    
    try:
        # 1. Bóc băng âm thanh toàn bộ video
        audio_segments = extract_and_transcribe_audio(video_path)
        
        # 2. Phát hiện chuyển cảnh
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
            
            start_sec = start_frame / fps
            end_sec = end_frame / fps
            duration = end_sec - start_sec
            
            # Adaptive Dense Sampling: Lấy 1 frame mỗi 5s cho các cảnh kéo dài
            frames_to_extract = []
            if duration > 5.0:
                current_sec = start_sec + 2.5
                while current_sec < end_sec:
                    frames_to_extract.append(int(current_sec * fps))
                    current_sec += 5.0
            else:
                mid_frame = start_frame + (end_frame - start_frame) // 2
                frames_to_extract.append(mid_frame)
                
            for sub_idx, frame_idx in enumerate(frames_to_extract):
                # Đồng bộ Audio cho sub-frame (Khoảng +- 2.5s xung quanh frame)
                sub_sec = frame_idx / fps
                sub_start_sec = max(start_sec, sub_sec - 2.5)
                sub_end_sec = min(end_sec, sub_sec + 2.5)
                
                scene_audio = []
                for seg in audio_segments:
                    if seg["start"] < sub_end_sec and seg["end"] > sub_start_sec:
                        scene_audio.append(seg["text"].strip())
                audio_transcript = " ".join(scene_audio)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_resized = cv2.resize(frame, (target_w, target_h))
                    
                    suffix = f"_{sub_idx}" if len(frames_to_extract) > 1 else ""
                    out_path = os.path.join(output_dir, f"{video_filename}_scene_{i:04d}{suffix}.jpg")
                    cv2.imwrite(out_path, frame_resized)
                    
                    frames_paths.append({
                        "path": out_path,
                        "timestamp_sec": sub_sec,
                        "scene_index": i,
                        "audio_transcript": audio_transcript
                    })
        
        cap.release()
        logger.info(f"Successfully processed {video_path}. {len(frames_paths)} keyframes saved.")
        return frames_paths
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return []

if __name__ == "__main__":
    pass
