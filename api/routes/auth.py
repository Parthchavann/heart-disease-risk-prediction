"""
Authentication endpoints — register, login, profile, prediction history.
Uses JWT tokens (30-day expiry) + bcrypt password hashing.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from api.database import get_db
from api.models_db import User, Prediction
from config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", "heartguard-dev-secret-change-in-prod")
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    from passlib.context import CryptContext
    ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return ctx.hash(password)


def _verify_password(plain: str, hashed: str) -> bool:
    from passlib.context import CryptContext
    ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return ctx.verify(plain, hashed)


def _create_token(user_id: int, email: str) -> str:
    from jose import jwt
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict:
    from jose import jwt, JWTError
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required.")
    payload = _decode_token(credentials.credentials)
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    email: str
    user_id: int


class PredictionHistoryItem(BaseModel):
    prediction_id: str
    risk_probability: float
    risk_level: str
    patient_data: dict
    created_at: str


class PredictionHistoryResponse(BaseModel):
    total: int
    predictions: List[PredictionHistoryItem]


class SavePredictionRequest(BaseModel):
    prediction_id: str
    risk_probability: float
    risk_level: str
    patient_data: dict
    result_data: Optional[dict] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=AuthResponse, status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user account."""
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    if len(req.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")

    user = User(email=req.email, hashed_password=_hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"New user registered: {user.email} (id={user.id})")
    return AuthResponse(token=_create_token(user.id, user.email), email=user.email, user_id=user.id)


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Login with email and password, returns JWT token."""
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not _verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    logger.info(f"User logged in: {user.email}")
    return AuthResponse(token=_create_token(user.id, user.email), email=user.email, user_id=user.id)


@router.get("/me")
def get_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    return {
        "user_id": current_user.id,
        "email": current_user.email,
        "member_since": current_user.created_at.isoformat(),
    }


@router.post("/predictions/save", status_code=201)
def save_prediction(
    req: SavePredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save a prediction result to the user's history."""
    # Avoid duplicates
    if db.query(Prediction).filter(Prediction.prediction_id == req.prediction_id).first():
        return {"message": "Already saved."}

    pred = Prediction(
        user_id=current_user.id,
        prediction_id=req.prediction_id,
        risk_probability=req.risk_probability,
        risk_level=req.risk_level,
        patient_data=json.dumps(req.patient_data),
        result_data=json.dumps(req.result_data) if req.result_data else None,
    )
    db.add(pred)
    db.commit()
    return {"message": "Prediction saved.", "prediction_id": req.prediction_id}


@router.get("/predictions/history", response_model=PredictionHistoryResponse)
def get_history(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the authenticated user's prediction history (most recent first)."""
    preds = (
        db.query(Prediction)
        .filter(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )
    items = [
        PredictionHistoryItem(
            prediction_id=p.prediction_id,
            risk_probability=p.risk_probability,
            risk_level=p.risk_level,
            patient_data=p.patient_data_dict(),
            created_at=p.created_at.isoformat(),
        )
        for p in preds
    ]
    return PredictionHistoryResponse(total=len(items), predictions=items)


@router.delete("/predictions/{prediction_id}", status_code=200)
def delete_prediction(
    prediction_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a specific prediction from history."""
    pred = (
        db.query(Prediction)
        .filter(
            Prediction.prediction_id == prediction_id,
            Prediction.user_id == current_user.id,
        )
        .first()
    )
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    db.delete(pred)
    db.commit()
    return {"message": "Deleted."}
