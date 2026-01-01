import time
import logging
from datetime import datetime
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI

from .config import settings
from . import rag

# ✅ 로거 설정
logger = logging.getLogger("llm_bot")
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

app = FastAPI()

# ✅ 공통 OpenAI 클라이언트
client = OpenAI(
    api_key=settings.openai_api_key,
    timeout=10.0,  # 타임아웃 (초)
)


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    channel: Optional[Literal["web", "slack", "teams"]] = "web"
    use_rag: bool = True


class ChatResponse(BaseModel):
    reply: str
    timestamp: str
    echo_mode: bool = False
    model: str
    used_rag: bool = False


@app.on_event("startup")
def on_startup():
    logger.info("서버 시작 - RAG 벡터 스토어 구축 시작")
    try:
	# IT 전용 문서 사용
        rag.build_vector_store(data_dir="data/it")
        logger.info("RAG 벡터 스토어 구축 완료 (data/it)")
    except Exception as e:
        logger.exception("RAG 벡터 스토어 구축 중 오류 발생: %s", e)
        # 여기서 바로 죽을지, 빈 RAG로 계속 갈지는 선택 사항
        # raise e


@app.get("/")
def read_root():
    return {"message": "Hello Nathan, RAG LLM chat API is ready!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


def generate_chat_reply(request: ChatRequest) -> ChatResponse:
    user_text = request.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="message 필드는 비어 있을 수 없습니다.")

    channel_context = {
        "web": "This request is coming from a web client.",
        "slack": "This request is coming from a Slack workspace.",
        "teams": "This request is coming from Microsoft Teams.",
    }.get(request.channel or "web", "The request source is unknown.")

    used_rag = False
    context_text = ""

    start_time = time.monotonic()

    try:
        if request.use_rag:
            context_text = rag.get_relevant_context(user_text)
            if context_text:
                used_rag = True

        system_prompt = (
            "You are a helpful assistant for an internal business chatbot. "
            "Use the provided context to answer the user's question, if it is relevant. "
            "If the context does not fully answer the question, you may use your own knowledge, "
            "but clearly state when you are unsure.\n\n"
            f"{channel_context}\n\n"
        )

        messages = [{"role": "system", "content": system_prompt}]

        if context_text:
            messages.append(
                {
                    "role": "system",
                    "content": f"Here is relevant context from internal documents:\n\n{context_text}",
                }
            )

        messages.append({"role": "user", "content": user_text})

        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0.3,
        )
        reply_text = completion.choices[0].message.content
        model_used = completion.model

    except Exception as e:
        logger.exception(
            "LLM 호출 중 오류: user_id=%s channel=%s use_rag=%s",
            request.user_id,
            request.channel,
            request.use_rag,
        )
        raise HTTPException(status_code=500, detail="LLM 호출 중 오류가 발생했습니다.")
    finally:
        duration = time.monotonic() - start_time
        logger.info(
            "chat 요청 처리: user_id=%s channel=%s used_rag=%s len_q=%d duration=%.2fs",
            request.user_id,
            request.channel,
            used_rag,
            len(user_text),
            duration,
        )

    return ChatResponse(
        reply=reply_text,
        timestamp=datetime.utcnow().isoformat() + "Z",
        echo_mode=False,
        model=model_used,
        used_rag=used_rag,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return generate_chat_reply(request)


@app.post("/slack/slash", response_class=PlainTextResponse)
def slack_slash(
    text: str = Form(...),
    user_id: str = Form(None),
):
    logger.info("Slack Slash 호출: user_id=%s text_preview=%s", user_id, text[:80])
    req = ChatRequest(
        message=text,
        user_id=user_id,
        channel="slack",
        use_rag=True,
    )
    resp = generate_chat_reply(req)
    return resp.reply
