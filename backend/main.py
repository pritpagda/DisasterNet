import datetime

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.auth import verify_token, get_current_user

app = FastAPI()

# CORS middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"], )


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/health")
def health_health():
    return {"status": "ok", "time": datetime.datetime.utcnow()}


@app.get("/secure")
def secure_endpoint(decoded_token: dict = Depends(verify_token)):
    return {"message": f"✅ Authenticated as {decoded_token['email']}"}
