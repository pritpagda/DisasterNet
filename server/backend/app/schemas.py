from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    prediction_id: int
    correct: str
    comments: str | None = None


class FeedbackSchema(BaseModel):
    correct: Optional[bool]
    comments: Optional[str]
    submitted_at: Optional[datetime]

    class Config:
        orm_mode = True


class PredictionHistorySchema(BaseModel):
    id: int
    text: str
    image_path: Optional[str]
    informative: Optional[bool]
    humanitarian: Optional[bool]
    damage: Optional[bool]
    created_at: datetime
    feedback: Optional[FeedbackSchema]

    class Config:
        orm_mode = True
