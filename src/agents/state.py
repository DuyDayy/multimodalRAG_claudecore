from typing import TypedDict, List, Optional, Dict, Any

class GraphState(TypedDict):
    """
    Represents the state of our LangGraph system.
    """
    session_id: str
    user_query: str
    query_image_path: Optional[str]
    query_type: str # "VIDEO_KIS", "TEXTUAL_KIS", "QA", "TRAKE", "UNKNOWN"
    
    retrieved_context: List[Dict[str, Any]]
    draft_answer: str
    
    iteration_count: int
    error_logs: List[str]
    is_passing: bool
