import os
import time
import uuid
import json
from src.agents.graph import build_graph

def run_benchmark():
    print("Khởi động quá trình Benchmark Kiến trúc Gemini 1.5 Pro (Dual-RAG)...")
    
    graph_app = build_graph()
    
    test_queries = [
        {"query": "Khi nào thì Aurelion Sol xuất hiện?", "type": "TEXTUAL_KIS"},
        {"query": "Tại sao Aurelion Sol lại mạnh như vậy ở cuối game?", "type": "QA"},
        {"query": "Chuyện gì xảy ra sau khi nổ hũ siêu thú?", "type": "TRAKE"}
    ]
    
    results = []
    
    for i, tq in enumerate(test_queries):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Câu hỏi: {tq['query']}")
        
        state_input = {
            "session_id": str(uuid.uuid4()),
            "user_query": tq["query"],
            "query_image_path": None,
            "explicit_query_type": tq["type"],
            "retrieved_context": [],
            "decoupled_requests": {},
            "auxiliary_texts": "",
            "draft_answer": "",
            "iteration_count": 0,
            "error_logs": [],
            "is_passing": True
        }
        
        start_time = time.time()
        
        try:
            # Chạy LangGraph
            final_state = graph_app.invoke(state_input)
            latency = time.time() - start_time
            
            print(f"Latency: {latency:.2f}s")
            print(f"Decoupled JSON: {json.dumps(final_state.get('decoupled_requests', {}), ensure_ascii=False)}")
            print(f"Answer: {final_state.get('draft_answer')[:100]}...")
            
            results.append({
                "query": tq["query"],
                "latency_sec": latency,
                "status": "Success",
                "answer_length": len(final_state.get("draft_answer", ""))
            })
        except Exception as e:
            latency = time.time() - start_time
            print(f"Lỗi: {e}")
            results.append({
                "query": tq["query"],
                "latency_sec": latency,
                "status": f"Failed: {str(e)}",
                "answer_length": 0
            })

    print("\n=== KẾT QUẢ BENCHMARK ===")
    print(f"{'Câu hỏi':<50} | {'Độ trễ':<10} | {'Trạng thái':<10} | {'Độ dài TL'}")
    print("-" * 90)
    for r in results:
        print(f"{r['query'][:48]:<50} | {r['latency_sec']:.2f}s     | {r['status']:<10} | {r['answer_length']}")

if __name__ == "__main__":
    run_benchmark()
