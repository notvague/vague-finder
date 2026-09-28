import os
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "vaguefinder")

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def _connect() -> None:
    global _client, _db

    if _client is not None:
        return

    _client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
    )
    _client.admin.command("ping")
    _db = _client[DB_NAME]


def get_db() -> Database:
    _connect()
    if _db is None:
        raise RuntimeError("MongoDB database initialization failed")
    return _db


def get_collection(name: str) -> Collection:
    return get_db()[name]
