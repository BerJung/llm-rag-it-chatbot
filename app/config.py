import os
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# .env 로드
load_dotenv()


class Settings(BaseModel):
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    rag_top_k: int = 4
    log_level: str = "INFO"


def get_settings() -> Settings:
    try:
        return Settings(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            rag_top_k=int(os.getenv("RAG_TOP_K", "4")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    except (ValidationError, ValueError) as e:
        # 설정 오류를 빠르게 발견하기 위한 에러
        raise RuntimeError(f"환경 변수 설정 오류: {e}")


settings = get_settings()

