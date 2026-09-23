"""
data_extractor.py
Agent 1: Data Extractor — Uses CHIMERA MCP tools to uncover masked EHR data.
"""

import json
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from schemas.schemas import ExtractedFeatures
from utils.audit import log_entry

EXTRACTOR_PROMPT = """You are a clinical data extraction agent. 
The patient's full EHR is masked. Use the available tools to fetch their PSA history, 
MRI reports, and previous notes. Once you have gathered sufficient clinical variables, 
extract the data into the structured schema."""

def data_extractor_agent(state: dict, llm_with_tools, tools: list) -> dict:
    """
    LLM-driven tool caller interacting with the CHIMERA MCP server.
    """
    messages = state.get("messages", [])
    if not messages:
        messages.append(HumanMessage(content=state.get("context", "")))
        
    messages.insert(0, SystemMessage(content=EXTRACTOR_PROMPT))

    # Tool invocation loop (simplified)
    for _ in range(3): 
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            # Map tool name to the actual tool execution
            tool_instance = next((t for t in tools if t.name == tool_call["name"]), None)
            if tool_instance:
                tool_result = tool_instance.invoke(tool_call["args"])
                messages.append(ToolMessage(
                    content=str(tool_result), 
                    tool_call_id=tool_call["id"]
                ))

    # Final structured extraction step
    extraction_llm = llm_with_tools.with_structured_output(ExtractedFeatures)
    structured_data = extraction_llm.invoke(messages)

    return {
        "extracted_features": structured_data.model_dump(),
        "messages": messages,
        "audit_log": log_entry(
            state, "data_extractor", "mcp_tool_extraction",
            structured_data.model_dump(),
        ),
    }