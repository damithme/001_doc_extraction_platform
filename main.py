import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.api.routes.documents import router as documents_router
from app.api.routes.query import router as query_router
from app.storage.db import create_tables
from app.storage.vector_store import ensure_collection

app = FastAPI(title="DocMind", description="Intelligent Document Q&A Agent")


@app.on_event("startup")
def startup():
    # Enable LangSmith tracing for all LangChain/LangGraph calls
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "docmind"))
        print("LangSmith tracing enabled.")

    create_tables()
    ensure_collection()
    print("DocMind ready.")


app.include_router(documents_router)
app.include_router(query_router)


@app.get("/health")
def health():
    return {"status": "ok"}