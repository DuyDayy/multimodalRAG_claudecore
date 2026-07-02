from langgraph.graph import StateGraph, END
from src.agents.state import GraphState
from src.agents.router_agent import route_query
from src.agents.retriever_agent import retrieve_context
from src.agents.generator_agent import generate_answer

def build_graph():
    # Khởi tạo Graph
    workflow = StateGraph(GraphState)
    
    # Thêm các nodes
    workflow.add_node("router", route_query)
    workflow.add_node("retriever", retrieve_context)
    workflow.add_node("generator", generate_answer)
    
    # Định nghĩa các cạnh (Edges)
    workflow.set_entry_point("router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "generator")
    
    # Ở phiên bản đầy đủ, sẽ có vòng lặp Self-Correction (Evaluator node)
    # Tại đây, Generator sẽ sinh ra và trực tiếp kết thúc
    workflow.add_edge("generator", END)
    
    # Biên dịch đồ thị
    app = workflow.compile()
    return app

# Singleton để gọi từ UI
graph_app = build_graph()
