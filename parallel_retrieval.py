"""
parallel_retrieval.py
Agent 4: Parallel Retrieval — executes all sub-queries against the guideline store.
"""

import concurrent.futures
import yaml

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from utils.audit import log_entry


with open("config/config.yaml") as f:
    _CFG = yaml.safe_load(f)

_VECTORSTORE = None


def _get_vectorstore():
    global _VECTORSTORE
    if _VECTORSTORE is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=_CFG["embeddings"]["model"],
            model_kwargs={"device": _CFG["embeddings"]["device"]},
        )
        _VECTORSTORE = FAISS.load_local(
            _CFG["retrieval"]["guideline_index_path"],
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _VECTORSTORE


def _retrieve_for_query(query: str, k: int) -> dict:
    vs = _get_vectorstore()
    docs = vs.similarity_search_with_score(query, k=k)
    return {
        "query": query,
        "documents": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "score": float(score),
            }
            for doc, score in docs
        ],
    }


def parallel_retrieval_node(state: dict) -> dict:
    """
    Execute all decomposed sub-queries in parallel and merge results.
    """
    decomposed = state.get("decomposed_queries") or {}
    sub_queries = decomposed.get("sub_queries", [])

    if not sub_queries:
        return {
            "retrieved_evidence": [],
            "audit_log": log_entry(
                state, "parallel_retrieval", "no_queries",
                {"reason": "empty sub_queries"},
            ),
        }

    k = _CFG["retrieval"]["top_k_per_query"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(_retrieve_for_query, q, k) for q in sub_queries]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Deduplicate and merge
    seen = set()
    merged = []
    for result in results:
        for doc in result["documents"]:
            h = hash(doc["content"][:200])
            if h not in seen:
                seen.add(h)
                doc["source_query"] = result["query"]
                merged.append(doc)

    merged.sort(key=lambda x: x["score"])

    # Cap to max evidence chunks
    max_chunks = _CFG["retrieval"]["max_evidence_chunks"]
    merged = merged[:max_chunks]

    return {
        "retrieved_evidence": merged,
        "audit_log": log_entry(
            state, "parallel_retrieval", "retrieved_evidence",
            {"num_sub_queries": len(sub_queries),
             "num_documents": len(merged)},
        ),
    }