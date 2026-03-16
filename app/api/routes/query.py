from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from app.agent.graph import graph
from app.observability.tracing import query_span

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]


@router.post("/")
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Invoke the LangGraph agent with the user's question.

    The agent will:
    1. Automatically retrieve relevant chunks (retrieve node)
    2. Call Claude, which may use tools to gather more context (reason ↔ tool_executor loop)
    3. Return the final answer + all cited chunks

    Set doc_ids to restrict the search to specific documents.
    """
    with query_span(request.question) as span:
        try:
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=request.question)],
                    "context": "",
                    "citations": [],
                    "doc_filter": request.doc_ids,
                },
                config={"recursion_limit": 10},
            )
        except Exception as e:
            msg = str(e)
            if "recursion" in msg.lower():
                raise HTTPException(
                    status_code=500,
                    detail="Agent exceeded maximum reasoning steps (10). Try a more specific question.",
                )
            raise HTTPException(status_code=500, detail=msg)

        # Count tool calls made (for observability)
        tool_calls = sum(
            1 for m in result["messages"] if isinstance(m, ToolMessage)
        )
        span["tool_calls_made"] = tool_calls
        span["top_k_chunks"] = len(result.get("citations", []))

    # Extract the final answer from the last AIMessage
    last_ai = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    if last_ai is None:
        raise HTTPException(status_code=500, detail="Agent produced no answer.")

    content = last_ai.content
    if isinstance(content, list):
        # Anthropic returns list-of-blocks when mixing text + tool use
        answer_text = " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        ).strip()
    else:
        answer_text = str(content)

    return QueryResponse(
        answer=answer_text,
        citations=result.get("citations", []),
    )