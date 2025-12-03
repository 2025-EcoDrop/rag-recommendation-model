from fastapi import APIRouter
from fastapi.params import Depends
from sqlalchemy.orm import Session
from app.schemas import QueryRequest, QueryResponse
from app.rag import search_knowledge_base
from app.llm_model import generate_answer

from app.database import deps
import os

router = APIRouter()
IS_TEST = os.getenv("TEST_ENV") == "true"

@router.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest, db: Session = Depends(deps.get_db)):
    if not IS_TEST:
        query = request.query

        docs = [[]]
        answer = ""

        try:
            docs = await search_knowledge_base(db, query)
            answer = await generate_answer(query, docs)
        except Exception as e:
            print("Something went wrong")
            print(e)
            raise

        return QueryResponse(
            answer=answer,
            sources=docs
        )

    return "test"