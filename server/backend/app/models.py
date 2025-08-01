from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from .db import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)

    predictions = relationship("Prediction", back_populates="user")
    feedbacks = relationship("Feedback", back_populates="user")


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    text = Column(Text)
    image_path = Column(String)
    informative = Column(String)
    humanitarian = Column(String)
    damage = Column(String)
    error = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="predictions")
    feedback = relationship("Feedback", back_populates="prediction", uselist=False)


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    correct = Column(String)
    comments = Column(Text)
    submitted_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="feedbacks")
    prediction = relationship("Prediction", back_populates="feedback")
