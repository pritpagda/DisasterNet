import os

import firebase_admin
from fastapi import HTTPException, Depends, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import credentials, auth as firebase_auth
from sqlalchemy.orm import Session

from .db import get_db
from .models import User

CREDENTIAL_FILE_PATH = os.path.join("backend", "app", "disaster-feea2-firebase-adminsdk-fbsvc-054b28de48.json")

if not os.path.exists(CREDENTIAL_FILE_PATH):
    raise RuntimeError(f"Firebase credential file not found at {CREDENTIAL_FILE_PATH}")

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(CREDENTIAL_FILE_PATH)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Firebase app: {e}")
bearer_scheme = HTTPBearer()


async def verify_token(token: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    try:
        decoded_token = await run_in_threadpool(firebase_auth.verify_id_token, token.credentials)
        uid = decoded_token.get("uid")
        email = decoded_token.get("email")
        if not uid or not email:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token is missing user information.")
        return {"uid": uid, "email": email}
    except (firebase_auth.InvalidIdTokenError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid or expired Firebase token: {e}",
                            headers={"WWW-Authenticate": "Bearer"}, )


def _get_or_create_user_in_db(db: Session, email: str) -> User:
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


async def get_current_db_user(token_data: dict = Depends(verify_token), db: Session = Depends(get_db)) -> User:
    email = token_data.get("email")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User email not found in token.")
    user = await run_in_threadpool(_get_or_create_user_in_db, db, email)
    return user


async def get_current_user(user=Depends(verify_token)):
    return user
