from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SearchId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
FeedbackToken = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{43}$")]


class FeedbackReceipt(BaseModel):
    """표시 결과를 저장했을 때만 발급하는 제출 권한. 브라우저 메모리에만 둔다."""

    search_id: SearchId
    token: FeedbackToken
    retention_days: int = 30


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    search_id: SearchId
    token: FeedbackToken
    submission_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{16,64}$")]
    expected_revision: int = Field(default=0, ge=0, le=21)
    action: Literal["select", "withdraw"] = "select"
    selected_song_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    reason: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == "select" and not self.selected_song_id:
            raise ValueError("정답 곡을 선택해 주세요.")
        if self.action == "withdraw" and (self.selected_song_id is not None or self.reason):
            raise ValueError("철회에는 곡이나 사유를 보내지 않습니다.")
        return self


class FeedbackResponse(BaseModel):
    search_id: SearchId
    revision: int
    status: Literal["pending", "withdrawn"]
    selected_song_id: Optional[str] = None
    reason: str = ""
