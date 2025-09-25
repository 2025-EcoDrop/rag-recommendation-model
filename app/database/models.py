from sqlalchemy import Column, String, BIGINT
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class DrugEmbedding(Base):
    __tablename__ = "drug_embedding"

    id = Column(BIGINT, primary_key=True, index=True)
    company_name = Column(String)
    product_name = Column(String)
    embedding = Column(Vector(1024))

class Drug(Base):
    __tablename__ = "drug"

    id = Column(BIGINT, primary_key=True, index=True)
    company_name = Column(String)
    product_name = Column(String)
    efficacy = Column(String)
    use = Column(String)
    caution_warn = Column(String)
    caution = Column(String)
    interaction = Column(String)
    side_effect = Column(String)
