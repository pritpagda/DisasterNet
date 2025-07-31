from io import BytesIO
from PIL import Image
from sqlalchemy.orm import Session

from .pred import predict as run_huggingface_prediction
from .models import Prediction


def predict(text: str, image_bytes: bytes, image_filename: str, user_id: int = None, db: Session = None):
    prediction_result = run_huggingface_prediction(text, image_bytes)

    prediction_result["text"] = text
    prediction_result["image_url"] = image_filename

    if db is not None and user_id is not None:
        prediction = Prediction(
            user_id=user_id,
            text=text,
            informative=prediction_result.get("informative"),
            humanitarian=prediction_result.get("humanitarian"),
            damage=prediction_result.get("damage"),
            image_path=image_filename,
            error=None,
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        prediction_result["id"] = prediction.id

    return prediction_result
