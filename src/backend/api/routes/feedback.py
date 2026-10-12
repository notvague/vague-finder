import asyncio
import logging

from fastapi import APIRouter, HTTPException, Response

from src.backend.schemas.feedback import FeedbackRequest, FeedbackResponse
from src.feedback.store import FeedbackError, collection_enabled, get_feedback_store

router = APIRouter(prefix="/feedback", tags=["Feedback"])
logger = logging.getLogger(__name__)


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, response: Response):
    response.headers["Cache-Control"] = "no-store"
    if not collection_enabled():
        raise HTTPException(503, "지금은 정답 피드백을 받지 않습니다.", headers={"Cache-Control": "no-store"})
    try:
        return await asyncio.to_thread(lambda: get_feedback_store().submit(request))
    except FeedbackError as exc:
        detail = {"message": exc.message}
        if exc.current:
            detail["current"] = exc.current.model_dump()
        raise HTTPException(exc.status, detail=detail, headers={"Cache-Control": "no-store"}) from None
    except Exception as exc:
        logger.warning("feedback submit unavailable: %s", type(exc).__name__)
        raise HTTPException(503, "저장 여부를 확인하지 못했습니다. 같은 내용으로 다시 제출해 주세요.",
                            headers={"Cache-Control": "no-store"}) from None
