import random

class MockResult:
    def __init__(self, video_id, timestamp_ms, score):
        self.payload = {"video_id": video_id, "timestamp_ms": timestamp_ms, "timestamp_sec": timestamp_ms / 1000}
        self.score = score

def generate_mock_data(num_frames=1000, num_videos=5):
    results = []
    for _ in range(num_frames):
        vid = f"video_{random.randint(1, num_videos)}"
        ts = random.randint(0, 100000) # Random timestamp up to 100s
        score = random.uniform(0.1, 0.9)
        results.append(MockResult(vid, ts, score))
    return results

def run_old_retriever_logic(raw_results):
    sorted_results = sorted(raw_results, key=lambda x: x.payload.get("timestamp_ms", 0))
    segments = []
    current_segment = []
    for r in sorted_results:
        if not current_segment:
            current_segment.append(r)
        else:
            if r.payload.get("timestamp_ms", 0) - current_segment[-1].payload.get("timestamp_ms", 0) <= 15000:
                current_segment.append(r)
            else:
                segments.append(current_segment)
                current_segment = [r]
    if current_segment:
        segments.append(current_segment)
    return segments

def run_new_retriever_logic(raw_results):
    video_groups = {}
    for r in raw_results:
        vid = r.payload.get("video_id", "unknown")
        if vid not in video_groups:
            video_groups[vid] = []
        video_groups[vid].append(r)
        
    segments = []
    for vid, vid_results in video_groups.items():
        sorted_results = sorted(vid_results, key=lambda x: x.payload.get("timestamp_ms", 0))
        current_segment = []
        for r in sorted_results:
            if not current_segment:
                current_segment.append(r)
            else:
                if r.payload.get("timestamp_ms", 0) - current_segment[-1].payload.get("timestamp_ms", 0) <= 15000:
                    current_segment.append(r)
                else:
                    segments.append(current_segment)
                    current_segment = [r]
        if current_segment:
            segments.append(current_segment)
    return segments

def run_old_generator_logic(top_frames):
    return sorted(top_frames, key=lambda x: x.payload.get("timestamp_sec", 0))

def run_new_generator_logic(top_frames):
    return sorted(top_frames, key=lambda x: (x.payload.get("video_id", ""), x.payload.get("timestamp_sec", 0)))

def evaluate():
    random.seed(42)
    data = generate_mock_data(500, 3) # 500 frames, 3 videos
    
    # 1. Evaluate Retriever Logic
    old_segments = run_old_retriever_logic(data)
    new_segments = run_new_retriever_logic(data)
    
    old_corrupted = 0
    for seg in old_segments:
        video_ids = set(r.payload["video_id"] for r in seg)
        if len(video_ids) > 1:
            old_corrupted += 1
            
    new_corrupted = 0
    for seg in new_segments:
        video_ids = set(r.payload["video_id"] for r in seg)
        if len(video_ids) > 1:
            new_corrupted += 1
            
    print("=== RETRIEVER LOGIC BENCHMARK ===")
    print(f"Tổng số đoạn (segments) sinh ra (Cũ): {len(old_segments)}")
    print(f"Số đoạn BỊ LỖI trộn lẫn Video (Cũ): {old_corrupted} / {len(old_segments)} ({old_corrupted/len(old_segments)*100:.2f}%)")
    print(f"Độ chính xác (Cũ): {(1 - old_corrupted/len(old_segments))*100:.2f}%")
    
    print(f"Tổng số đoạn (segments) sinh ra (Mới): {len(new_segments)}")
    print(f"Số đoạn BỊ LỖI trộn lẫn Video (Mới): {new_corrupted} / {len(new_segments)} ({new_corrupted/len(new_segments)*100:.2f}%)")
    print(f"Độ chính xác (Mới): {(1 - new_corrupted/len(new_segments))*100:.2f}%\n")
    
    # 2. Evaluate Generator Logic
    # Select 20 random frames
    top_frames = random.sample(data, 20)
    old_sorted = run_old_generator_logic(top_frames)
    new_sorted = run_new_generator_logic(top_frames)
    
    # Check if videos are grouped
    def is_grouped_correctly(sorted_frames):
        seen_videos = set()
        last_video = None
        for f in sorted_frames:
            vid = f.payload["video_id"]
            if vid != last_video:
                if vid in seen_videos:
                    return False
                seen_videos.add(vid)
                last_video = vid
        return True

    old_correct = is_grouped_correctly(old_sorted)
    new_correct = is_grouped_correctly(new_sorted)
    
    print("=== GENERATOR LOGIC BENCHMARK ===")
    print(f"Giữ được tính liền mạch của bối cảnh (Context Continuity):")
    print(f"Logic Cũ: {'Đạt' if old_correct else 'Thất bại (Bị phân mảnh)'}")
    print(f"Logic Mới: {'Đạt' if new_correct else 'Thất bại (Bị phân mảnh)'}")

if __name__ == "__main__":
    evaluate()
