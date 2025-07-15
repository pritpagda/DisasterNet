import datetime

from fastapi import FastAPI, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .app.batch_predict import process_batch
from .app.auth import get_current_user
from .app.database import connect_to_db
from .app.predict_explain import predict_explain

app = FastAPI(title="DisasterNet")

# ✅ CORS middleware — open to all for now (adjust in prod)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"], )


# ✅ Connect to DB on startup
@app.on_event("startup")
async def startup_db():
    await connect_to_db(app)


# ✅ Health check (basic)
@app.get("/")
def root():
    return {"status": "ok"}


# ✅ Health check with timestamp
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat()}


# ✅ Authenticated user info (Firebase)
@app.get("/auth/me")
async def auth_me(user=Depends(get_current_user)):
    return {"uid": user["uid"], "email": user.get("email")}


@app.post("/predict")
async def predict_post(text: str = Form(...), image: UploadFile = File(...)):
    image_bytes = await image.read()
    result = predict_explain(text, image_bytes)
    return JSONResponse(result)


@app.post("/predict-batch")
async def predict_batch_endpoint(images_zip: UploadFile = File(...), texts_csv: UploadFile = File(...)):
    images_bytes = await images_zip.read()
    texts_bytes = await texts_csv.read()
    return process_batch(images_bytes, texts_bytes)
