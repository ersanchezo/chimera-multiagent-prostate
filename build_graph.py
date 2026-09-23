"""
build_graph.py
Adapted LangGraph construction for the CHIMERA-agent baseline runner.
"""

from typing import List, Any
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from utils.state import MultiAgentState
from agents.data_extractor import data_extractor_agent
from agents.ml_inference import ml_inference_node
from agents.risk_stratifier import risk_stratifier_agent
from agents.biopsy_decider import biopsy_decider_agent
from agents.ensemble_fusion import ensemble_fusion_node

def create_graph(
    tools: List[BaseTool],
    model: Any,
    system_prompt: str,
    **kwargs
):
    """
    Factory function invoked by run.py. 
    Binds the CHIMERA MCP tools to the agent graph.
    """
    workflow = StateGraph(MultiAgentState)

    # Bind tools to the LLM model instance for the extractor node
    llm_with_tools = model.bind_tools(tools)

    # Pass the tool-bound LLM to the extractor via a closure or class
    def extractor_wrapper(state: dict):
        return data_extractor_agent(state, llm_with_tools, tools)

    workflow.add_node("extractor", extractor_wrapper)
    workflow.add_node("ml_inference", ml_inference_node)
    workflow.add_node("risk_stratifier", risk_stratifier_agent)
    workflow.add_node("biopsy_decider", biopsy_decider_agent)
    workflow.add_node("fusion", ensemble_fusion_node)

    # Simplified linear flow for demonstration
    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "ml_inference")
    workflow.add_edge("ml_inference", "risk_stratifier")
    workflow.add_edge("risk_stratifier", "biopsy_decider")
    workflow.add_edge("biopsy_decider", "fusion")
    workflow.add_edge("fusion", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)