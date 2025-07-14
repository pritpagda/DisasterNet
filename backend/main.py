import datetime

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.auth import get_current_user
from app.database import connect_to_db

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
