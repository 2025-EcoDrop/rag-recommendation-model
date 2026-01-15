from sqlalchemy.orm import Session
import sqlalchemy as sa
from torch import Tensor

from app.database.models import DrugEmbedding, Drug
import os

IS_TEST = os.getenv("TEST_ENV") == "true"

if not IS_TEST:
    from sentence_transformers import SentenceTransformer

    # Snowflake를 Fine-tuning한 모델
    model = SentenceTransformer("dragonkue/snowflake-arctic-embed-l-v2.0-ko" , device="cpu")

async def embed_text(text: str) -> str | Tensor:
    if not IS_TEST:
        return model.encode(text)

    return "test"

async def search_embed_text(db: Session, squery: str, k: int = 3):
    if not IS_TEST:
        q_embed = await embed_text(squery)

        rows = (db.query(DrugEmbedding.id)
                .order_by(DrugEmbedding.embedding.op("<->")(q_embed.tolist())) # L2 distance
                .limit(k)
                .all())

        return [row[0] for row in rows]

    return "test"

async def search_knowledge_base(db: Session, squery: str, k: int = 3):
    if not IS_TEST:
        results = await search_embed_text(db, squery, k)

        rows = (db.query(Drug.company_name, Drug.product_name, Drug.efficacy, Drug.use, Drug.caution_warn, Drug.caution, Drug.interaction, Drug.side_effect)
                .filter(Drug.id.in_(results))
                .order_by(
                    sa.case(
                        {id_: index for index, id_ in enumerate(results)},
                        value=Drug.id
                    )
                )
                .all())

        return [list(row) for row in rows]

    return "test"
