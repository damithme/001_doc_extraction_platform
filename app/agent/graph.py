"""
LangGraph agent for DocMind.

Graph flow:
  START → retrieve → reason ⇄ tool_executor → answer → END

Node responsibilities:
  retrieve      — embed user query, fetch top-5 chunks automatically before Claude runs
  reason        — call Claude with context + bound tools; Claude decides next action
  tool_executor — execute every tool call Claude requested in its last message
  answer        — scan ToolMessage history, deduplicate cited chunks → state.citations
"""

import json
from typing import Annotated, Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOLS
from app.ingestion.embedder import embed_text
from app.storage import vector_store as vs


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str              # formatted text from the initial automatic retrieval
    citations: list[dict]     # deduplicated chunks referenced in the final answer
    doc_filter: list[str] | None  # optional scope: restrict search to these doc_ids


# ── Model + tool registry ───────────────────────────────────────────────────────

_llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2048)
_llm_with_tools = _llm.bind_tools(TOOLS)
_tool_map = {t.name: t for t in TOOLS}


# ── Nodes ──────────────────────────────────────────────────────────────────────

def retrieve(state: AgentState) -> dict[str, Any]:
    """
    Automatic first step: embed the user's query and fetch top-k chunks.
    The results are stored in state.context (not in messages) so they can be
    injected into the system prompt without breaking the human/assistant turn order.
    """
    user_msg = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    query = user_msg.content if isinstance(user_msg.content, str) else str(user_msg.content)

    query_vector = embed_text(query)
    chunks = vs.search(query_vector, top_k=5, doc_ids=state.get("doc_filter"))

    if not chunks:
        context = "No relevant document chunks found for the initial query."
    else:
        parts = [
            (
                f"[id={c['id']}, doc_id={c['doc_id']}, "
                f"chunk={c['chunk_index']}, page={c.get('page_num')}]\n{c['text']}"
            )
            for c in chunks
        ]
        context = "\n\n---\n\n".join(parts)

    return {"context": context}


def reason(state: AgentState) -> dict[str, Any]:
    """
    Call Claude with:
      - System prompt (+ injected retrieval context)
      - Full message history (user turn + any prior tool calls/results)
    Claude either calls a tool or produces the final answer.
    """
    system_content = SYSTEM_PROMPT
    if state.get("context"):
        system_content += (
            "\n\n## Initially Retrieved Context\n\n"
            + state["context"]
            + "\n\nIf this context is insufficient, use vector_search to find more."
        )

    response = _llm_with_tools.invoke(
        [SystemMessage(content=system_content)] + state["messages"]
    )
    return {"messages": [response]}


def tool_executor(state: AgentState) -> dict[str, Any]:
    """
    Execute every tool call from Claude's last AIMessage.
    Each result is wrapped in a ToolMessage so Claude can read it in the next turn.
    """
    last_msg = state["messages"][-1]
    tool_results: list[BaseMessage] = []

    for tc in last_msg.tool_calls:
        tool_fn = _tool_map.get(tc["name"])
        if tool_fn is None:
            result_content = f"Tool '{tc['name']}' not found."
        else:
            try:
                raw = tool_fn.invoke(tc["args"])
                result_content = raw if isinstance(raw, str) else json.dumps(raw)
            except Exception as e:
                result_content = f"Error executing {tc['name']}: {e}"

        tool_results.append(
            ToolMessage(
                content=result_content,
                tool_call_id=tc["id"],
                name=tc["name"],
            )
        )

    return {"messages": tool_results}


def answer(state: AgentState) -> dict[str, Any]:
    """
    Scan all ToolMessages in history to collect unique chunks that were retrieved.
    These become the citations returned alongside the final answer.

    Why here and not inline in reason?
    Separating citation extraction keeps reason() focused on LLM calls and makes
    the citation logic easy to test independently.
    """
    seen: set[str] = set()
    citations: list[dict] = []

    for msg in state["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name not in ("vector_search", "get_chunk_by_id"):
            continue
        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue

        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict) and item.get("id") and item["id"] not in seen:
                seen.add(item["id"])
                citations.append({
                    "chunk_id": item["id"],
                    "doc_id": item.get("doc_id"),
                    "chunk_index": item.get("chunk_index"),
                    "page_num": item.get("page_num"),
                })

    return {"citations": citations}


# ── Edge condition ─────────────────────────────────────────────────────────────

def _should_continue(state: AgentState) -> str:
    """Route: if Claude's last message has tool calls → tool_executor, else → answer."""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        return "tool_executor"
    return "answer"


# ── Build + compile graph ──────────────────────────────────────────────────────

_builder = StateGraph(AgentState)

_builder.add_node("retrieve", retrieve)
_builder.add_node("reason", reason)
_builder.add_node("tool_executor", tool_executor)
_builder.add_node("answer", answer)

_builder.set_entry_point("retrieve")
_builder.add_edge("retrieve", "reason")
_builder.add_conditional_edges(
    "reason",
    _should_continue,
    {"tool_executor": "tool_executor", "answer": "answer"},
)
_builder.add_edge("tool_executor", "reason")
_builder.add_edge("answer", END)

graph = _builder.compile()