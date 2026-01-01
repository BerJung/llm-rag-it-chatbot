import os
import glob
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from .config import settings #공통 설정 사용

# 로컬 벡터 DB 저장 경로
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "docs"

# OpenAI 클라이언트 (main.py와 같은 키 사용)
client = OpenAI(
	api_key=os.getenv("OPENAI_API_KEY"),
	timeout=10.0, #초 단위
)


def load_text_files(data_dir: str = "data") -> List[Tuple[str, str]]:
    """
    data/ 폴더 안의 .txt 파일을 모두 읽어서 (파일명, 텍스트) 리스트로 반환.
    """
    paths = glob.glob(os.path.join(data_dir, "*.txt"))
    docs = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append((os.path.basename(path), text))
        except Exception as e:
            print(f"[RAG] 파일 읽기 실패: {path} ({e})")
    return docs


def simple_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    매우 단순한 문자 기준 chunking.
    chunk_size: 각 조각 길이
    overlap: 조각 사이 겹치는 부분 길이
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 겹치게 이동
        if start < 0:
            start = 0
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    OpenAI Embedding API로 텍스트 리스트를 벡터로 변환.
    """
    if not texts:
        return []

    response = client.embeddings.create(
        model=settings.embedding_model, # .env에서 설정 가능
        input=texts,
    )
    return [item.embedding for item in response.data]


def build_vector_store(data_dir: str = "data") -> chromadb.Collection:
    """
    data/ 폴더의 텍스트 파일을 읽어서 ChromaDB 컬렉션 생성.
    서버 시작 시 한 번 호출.
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    # Chroma 클라이언트 (Persistent)
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(
            anonymized_telemetry=False,
        ),
    )

    # 기존 컬렉션 있으면 삭제 후 새로 생성 (간단하게)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    docs = load_text_files(data_dir=data_dir)

    if not docs:
        print("[RAG] data/ 폴더에 .txt 문서가 없습니다. 빈 컬렉션을 반환합니다.")
        return collection

    all_texts = []
    metadatas = []
    ids = []

    idx = 0
    for filename, text in docs:
        chunks = simple_chunk_text(text)
        for chunk in chunks:
            all_texts.append(chunk)
            metadatas.append({"source": filename})
            ids.append(f"{filename}_{idx}")
            idx += 1

    print(f"[RAG] 총 {len(all_texts)}개 chunk 생성. Embedding 진행 중...")
    embeddings = embed_texts(all_texts)
    print("[RAG] Embedding 완료. 컬렉션에 추가 중...")

    collection.add(
        ids=ids,
        documents=all_texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("[RAG] 벡터 스토어 구축 완료.")
    return collection


def get_relevant_context(
    question: str,
    top_k: int = None,
) -> str:
    if top_k is None:
       top_k = settings.rag_top_k  #기본값은 설정에서
    """
    질문에 대해 상위 top_k개 관련 chunk를 찾아 하나의 context 문자열로 합침.
    """
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(
            anonymized_telemetry=False,
        ),
    )
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        # 컬렉션이 아직 없으면 빈 context
        return ""

    # 질문도 embedding
    q_emb = embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    merged = []
    for doc, meta in zip(docs, metadatas):
        source = meta.get("source", "unknown")
        merged.append(f"[{source}]\n{doc}")

    context_text = "\n\n---\n\n".join(merged)
    return context_text

import os
import glob
from typing import List, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from .config import settings  # ✅ 공통 설정 사용


CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "docs"

# ✅ 공통 OpenAI 클라이언트 (타임아웃 포함)
client = OpenAI(
    api_key=settings.openai_api_key,
    timeout=10.0,  # 초 단위, 너무 길게 두지 않기
)


def load_text_files(data_dir: str = "data") -> List[Tuple[str, str]]:
    paths = glob.glob(os.path.join(data_dir, "*.txt"))
    docs = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append((os.path.basename(path), text))
        except Exception as e:
            print(f"[RAG] 파일 읽기 실패: {path} ({e})")
    return docs


def simple_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    response = client.embeddings.create(
        model=settings.embedding_model,  # ✅ .env에서 설정 가능
        input=texts,
    )
    return [item.embedding for item in response.data]


def build_vector_store(data_dir: str = "data") -> chromadb.Collection:
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=ChromaSettings(
            anonymized_telemetry=False,
        ),
    )

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    docs = load_text_files(data_dir=data_dir)

    if not docs:
        print("[RAG] data/ 폴더에 .txt 문서가 없습니다. 빈 컬렉션을 반환합니다.")
        return collection

    all_texts = []
    metadatas = []
    ids = []

    idx = 0
    for filename, text in docs:
        chunks = simple_chunk_text(text)
        for chunk in chunks:
            all_texts.append(chunk)
            metadatas.append({"source": filename})
            ids.append(f"{filename}_{idx}")
            idx += 1

    print(f"[RAG] 총 {len(all_texts)}개 chunk 생성. Embedding 진행 중...")
    embeddings = embed_texts(all_texts)
    print("[RAG] Embedding 완료. 컬렉션에 추가 중...")

    collection.add(
        ids=ids,
        documents=all_texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("[RAG] 벡터 스토어 구축 완료.")
    return collection


def get_relevant_context(
    question: str,
    top_k: int = None,
) -> str:
    if top_k is None:
        top_k = settings.rag_top_k  # ✅ 기본값은 설정에서

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=ChromaSettings(
            anonymized_telemetry=False,
        ),
    )
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        return ""

    q_emb = embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    merged = []
    for doc, meta in zip(docs, metadatas):
        source = meta.get("source", "unknown")
        merged.append(f"[{source}]\n{doc}")

    context_text = "\n\n---\n\n".join(merged)
    return context_text

