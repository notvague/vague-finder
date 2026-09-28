"""
src/vector_db/pinecone_client.py

Pinecone 클라이언트 생성 담당
"""
from __future__ import annotations

from pinecone import Pinecone


def get_pinecone_client() -> Pinecone:
    return Pinecone()