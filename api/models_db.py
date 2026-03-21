"""
SQLAlchemy ORM models for user accounts and prediction history.
"""

import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from api.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    prediction_id = Column(String(64), unique=True, index=True)
    risk_probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    patient_data = Column(Text, nullable=False)   # JSON string
    result_data = Column(Text, nullable=True)      # Full result JSON (optional)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")

    def patient_data_dict(self) -> dict:
        return json.loads(self.patient_data)

    def result_data_dict(self) -> dict:
        return json.loads(self.result_data) if self.result_data else {}
