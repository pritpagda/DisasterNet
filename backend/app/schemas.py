from typing import Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str
    image_url: Optional[str] = None
