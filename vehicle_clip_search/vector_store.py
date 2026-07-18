"""ChromaDB-backed vector store for car / plate embeddings.
Two collections: full-car crops and plate crops, both cosine space."""

import time

import chromadb

from .config import DB_PATH

CAR_COLLECTION = "vehicle_embeddings"
PLATE_COLLECTION = "plate_embeddings"

_client = None
_car_collection = None
_plate_collection = None


def _init():
    global _client, _car_collection, _plate_collection
    if _client is not None:
        return
    _client = chromadb.PersistentClient(path=DB_PATH)
    _car_collection = _client.get_or_create_collection(CAR_COLLECTION, metadata={"hnsw:space": "cosine"})
    _plate_collection = _client.get_or_create_collection(PLATE_COLLECTION, metadata={"hnsw:space": "cosine"})


def _clean(metadata: dict) -> dict:
    return {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in metadata.items()}


def insert_vehicle(car_vector, plate_vector, metadata: dict):
    """metadata: track_id, brand, color, plate, direction, time, car_image, plate_image, description"""
    _init()
    meta = _clean(metadata)
    uid = f"{metadata.get('track_id', 0)}_{int(time.time() * 1000)}"
    _car_collection.add(ids=[f"car_{uid}"], embeddings=[car_vector], metadatas=[meta])
    if plate_vector is not None:
        _plate_collection.add(ids=[f"plate_{uid}"], embeddings=[plate_vector], metadatas=[meta])


def _normalize(results) -> list:
    if not results.get("ids") or not results["ids"][0]:
        return []
    return [
        {"id": _id, "entity": meta, "distance": dist}
        for _id, meta, dist in zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
    ]


def _search(collection, query_vector, limit=12):
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(query_embeddings=[query_vector], n_results=min(limit, count))
    return _normalize(results)


def search_cars(query_vector, limit=12):
    _init()
    return _search(_car_collection, query_vector, limit)


def search_plates(query_vector, limit=12):
    _init()
    return _search(_plate_collection, query_vector, limit)
