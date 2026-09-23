<p align="center">
  <img src="docs/images/thumbnail.png" alt="CHIMERA Agent" width="250">
</p>

# Chimera Multi Agent System

My implementation of multi-agentic system for the
[CHIMERA-Agent challenge](https://chimera-agent.grand-challenge.org/chimera-agent/).
A LangGraph ReAct loop calls clinical tools served via MCP, retrieves
guidelines via RAG, and emits a structured per-case decision through a
terminal form-fill node.

My implementation is an 8-agent LangGraph pipeline that performs parallel query-decomposed RAG and concurrent deep learning/clinical ML inference. It then elevates decision reliability by cross-checking reasoning through a guideline validator and combining all risk signals into a rule-based ensemble fusion node for calibrated, auditable outputs.
# Architecture Overview

This repository implements a localized multi-agent decision support system orchestrated via LangGraph. It replaces the baseline's monolithic ReAct loop with a dedicated Directed Acyclic Graph (DAG) designed to enhance reasoning reliability through specialized agents, concurrent localized ML inference, and query-decomposed RAG.

The system integrates directly into the official CHIMERA runner framework. It interfaces with the organizing framework’s Model Context Protocol (MCP) server to simulate realistic EHR masking while executing localized, state-of-the-art machine learning models on MRI embeddings and clinical features.

## Visual System Architecture

```mermaid
graph TD
    %% Base Styling
    classDef baseline fill:#f9f,stroke:#333,stroke-width:2px;
    classDef agent fill:#e1f5fe,stroke:#0277bd,stroke-width:1px;
    classDef tool fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px,stroke-dasharray: 5 5;
    classDef localModel fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef state fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px;

    %% Input/Output
    subgraph GC ["Grand Challenge Environment"]
        Input["Case Input (data/task1/agent_input/)"]
        Output["GC Outputs (output/task1/)"]
    end

    %% Baseline Wrapper
    subgraph Runner ["Baseline Runner (run.py)"]
        direction TB
        Hydra["Hydra Configuration"]
        CaseLoad["Case Loader & Jinja Template"]
        Orchestrator["src.chimera_agent_baseline.agent.graph.create_graph()"]
    end

    %% The Core State
    SharedState[("MultiAgentState
    (State Dictionary)")]:::state

    %% The LangGraph DAG
    subgraph CustomGraph ["Custom LangGraph DAG (Refactored graph.py)"]
        direction TB
        NodeExtractor[["1. Data Extractor
        (Tool Caller Agent)"]]:::agent
        NodeML[["2. Concurrent ML Inference
        (ThreadPool)"]]:::agent
        NodeRisk[["3. Risk Stratifier"]]:::agent
        NodeDecomp[["4. Query Decomposer"]]:::agent
        NodeRetrieval[["5. Parallel Retrieval"]]:::agent
        NodeDecider[["6. Biopsy Decider
        (CoT Reasoning)"]]:::agent
        NodeValidator[["7. Guideline Validator"]]:::agent
        NodeFusion[["8. Deterministic Ensemble Fusion
        (Terminal Node)"]]:::agent
    end

    %% External Systems Interfaced
    subgraph MCP ["CHIMERA MCP Server (Provided)"]
        direction TB
        ToolEHR{{"EHR Simulation Tools
        (PSA, Notes, Labs)"}}:::tool
        ToolRAG{{"search_guidelines
        (ChromaDB Tool)"}}:::tool
    end

    subgraph LocalModels ["Custom localized Models"]
        direction TB
        PtMRI[["localized PyTorch
        MRI Model"]]:::localModel
        SklearnClin[["localized Sklearn
        Clinical Model"]]:::localModel
    end

    %% UTILS
    AuditLog[("Audit Logger
    (Immutable JSONL logs)")]:::baseline

    %% --- Connections & Flow ---

    %% Input to Graph Start
    Input --> CaseLoad
    CaseLoad --> SharedState
    SharedState -. Initialize .-> Orchestrator

    %% Graph Internal Connections
    Orchestrator --> NodeExtractor
    NodeExtractor --> NodeML
    NodeML --> NodeRisk
    NodeRisk --> NodeDecomp
    NodeDecomp --> NodeRetrieval
    NodeRetrieval --> NodeDecider
    NodeDecider --> NodeValidator
    
    %% Conditional Logic for safety loop
    NodeValidator -- "Wait/Fix Loop" --> NodeDecomp
    NodeValidator -- "Validated" --> NodeFusion
    NodeFusion --> END[END node]

    %% Shared State Merging (Partial Updates)
    NodeExtractor -. Merges Updates .-> SharedState
    NodeML -. Merges Updates .-> SharedState
    NodeRisk -. Merges Updates .-> SharedState
    NodeFusion -. Fills schema .-> SharedState

    %% Tool/Model Invocations
    NodeExtractor ==> ToolEHR
    NodeML ==> PtMRI
    NodeML ==> SklearnClin
    NodeRetrieval ==> ToolRAG

    %% Utility Connections
    NodeExtractor -. Logs transition .-> AuditLog
    NodeDecider -. Logs transition .-> AuditLog
    NodeFusion -. Logs transition .-> AuditLog

    %% Output Final JSON
    SharedState -- "formats structured response" --> Output

    %% Formatting links
    linkStyle 10,11,12 stroke:#7b1fa2,stroke-width:1px,stroke-dasharray: 3 3;
    linkStyle 13,14,15,16 stroke:#2e7d32,stroke-width:2px;
    linkStyle 17,18,19 stroke:#7b1fa2,stroke-width:1px,stroke-dasharray: 3 3;
```

The architecture operates on a cyclical shared state dictionary (MultiAgentState), where each specialized agent node performs a discrete action, updates the state with partial results, and triggers the next node.Framework Execution: The framework (run.py) initializes the environment, mounts the organizing team’s MCP Server, and builds the agent graph. It loads patient data via Structured-prompt.json but holds the "Extended EHR View" masked behind MCP tools.   
Node 1: Data Extractor (Active EHR Uncovering): Unlike baselines that ingest flat dictionaries, this agent is a localized ReAct loop. It actively invokes MCP Tools (e.g., retrieving masked pathology, full MRI reports, and longitudinal notes) to simulate clinical chart review. It merges this uncovered data into a predefined schema in the Shared State.   
Node 2: Concurrent ML Inference: This node executes two localized, quantitative machine learning models in parallel using a python thread pool, ensuring efficient execution without blocking the main LLM thread.   MRI Classifier: Executes a localized PyTorch deep learning model against frozen modality-level embeddings (derived from patient imaging) to determine clinically significant prostate cancer (csPCa) probability.   Clinical Readiness Model: Executes a localized Scikit-Learn classifier on extracted patient clinical features to compute a baseline biopsy readiness probability.   
Nodes 3–5: Structured Knowledge Retrieval stack: Instead of ad-hoc retrieval, this stack applies Structural Query Decomposition.Risk Stratifier (Node 3): Combines LLM reasoning with rules to perform dynamic NCCN risk categorization.   
Query Decomposer (Node 4): Breaks the complex record into 3–5 targeted clinical sub-queries (e.g., PSA velocity criteria, PI-RADS management pathways).   Parallel Retrieval (Node 5): Executes these sub-queries concurrently against the organized framework’s guidelines search tool, performing deduplication and reranking chunks before feeding verified clinical evidence into the state.   
Node 6: Biopsy Decider (CoT Reasoning): This specialized urology decision agent receives all available signals: extracted clinical data, localized ML csPCa/readiness probabilities, and RAG-grounded guidelines. It performs step-by-step Chain-of-Thought (CoT) reasoning, reconciling signals, and noting any disagreements between the LLM and the numerical ML models.   
Node 7: Guideline Validator (Safety Guard): A dedicated validator agent acts as a safety gate. It cross-checks the Biopsy Decider’s output and key factors against the retrieved guidelines. If the reasoning contradicts known guidelines (e.g., recommending against biopsy on a PI-RADS 5 lesion with high PSA density), it flags the decision as REJECTED and triggers a conditional routing loop back to the Decomposer (Node 4) for clarification.   
Node 8: Deterministic Ensemble Fusion (Terminal Node): To avoid final-pass hallucination, this node is purely rule-based (non-LLM). It mathematically combines all risk signals (LLM confidence, localized ML probabilities, and validator verification scores) to produce the final biopsy recommendation (yes/no). It formats the output precisely into the Pydantic schema required by the challenge runner.  

For details on the challenge, tasks tested and Input/Output, please see the baseline repo: https://github.com/DIAGNijmegen/chimera-agent-baseline/
