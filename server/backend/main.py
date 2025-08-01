import datetime
import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from imagekitio import ImageKit
from sqlalchemy.orm import Session

from .app.auth import get_current_db_user, get_current_user
from .app.batch_predict import process_batch
from .app.db import get_db
from .app.models import Prediction, Feedback
from .app.predict_explain import predict_explain
from .app.schemas import FeedbackRequest

app = FastAPI(title="DisasterNet")
load_dotenv
imagekit = ImageKit(private_key=os.getenv("PRIVATE_KEY"),
                    public_key=os.getenv("PUBLIC_KEY"), url_endpoint=os.getenv("URL_ENDPOINT"))

origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
origins = [origin.strip() for origin in origins if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"], )


@app.get("/")
def root():
    return {"status": "not ok", "time": datetime.datetime.utcnow().isoformat()}


@app.get("/auth/me")
async def auth_me(user=Depends(get_current_db_user)):
    return {"id": user.id, "email": user.email,
            "created_at": user.created_at.isoformat() if user.created_at else None, }


@app.get("/imagekit-auth")
def get_imagekit_auth():
    auth_params = imagekit.get_authentication_parameters()
    return auth_params


@app.post("/predict")
async def predict_post(text: str = Form(...), image_url: str = Form(...), user=Depends(get_current_db_user),
                       db: Session = Depends(get_db), ):
    response = requests.get(image_url)
    response.raise_for_status()
    image_bytes = response.content

    result = predict_explain(text=text, image_bytes=image_bytes, user_id=user.id, db=db, image_filename=image_url, )
    return JSONResponse(result)


@app.post("/predict-batch")
async def predict_batch_endpoint(images_zip: UploadFile = File(...), texts_csv: UploadFile = File(...),
                                 user=Depends(get_current_user)):
    images_bytes = await images_zip.read()
    texts_bytes = await texts_csv.read()
    return process_batch(images_bytes, texts_bytes)


@app.post("/feedback", status_code=201)
def submit_feedback(data: FeedbackRequest, db: Session = Depends(get_db), current_user=Depends(get_current_db_user)):
    prediction = db.query(Prediction).filter(Prediction.id == data.prediction_id,
                                             Prediction.user_id == current_user.id).first()

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    if prediction.feedback:
        raise HTTPException(status_code=400, detail="Feedback already submitted.")

    feedback = Feedback(prediction_id=prediction.id, user_id=current_user.id, correct=data.correct,
                        comments=data.comments)

    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return {"message": "Feedback submitted successfully"}


@app.get("/history")
def get_prediction_history(db: Session = Depends(get_db), current_user=Depends(get_current_db_user)):
    history = db.query(Prediction).filter(Prediction.user_id == current_user.id).all()

    return [{"id": p.id, "text": p.text, "image_path": p.image_path, "informative": p.informative,
             "humanitarian": p.humanitarian, "damage": p.damage, "created_at": p.created_at.isoformat(),
             "feedback": {"correct": p.feedback.correct, "comments": p.feedback.comments,
                          "submitted_at": p.feedback.submitted_at.isoformat()} if p.feedback else None} for p in
            history]
