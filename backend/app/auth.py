import firebase_admin
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import credentials, auth as firebase_auth

# Path to Firebase service account
SERVICE_ACCOUNT_PATH = "disaster-feea2-firebase-adminsdk-fbsvc-054b28de48.json"

# Initialize Firebase app only once
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)

# Bearer scheme for extracting the Authorization header
bearer_scheme = HTTPBearer()


# Middleware to verify token from Authorization header
async def verify_token(token: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        decoded_token = firebase_auth.verify_id_token(token.credentials)
        uid = decoded_token.get("uid")
        email = decoded_token.get("email")
        return {"uid": uid, "email": email}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired Firebase token")


# Optional: Dependency to use inside endpoint functions
async def get_current_user(user=Depends(verify_token)):
    return user
