from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from src.database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    runs = relationship(
        "Run",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class ModelConfig(Base):
    __tablename__ = "model_config"

    config_id = Column(Integer, primary_key=True, index=True)
    config_key = Column(String(100), unique=True, nullable=False)
    config_value = Column(String(255), nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class Run(Base):
    __tablename__ = "runs"

    run_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.user_id"),
        nullable=False,
        index=True,
    )

    run_type = Column(String(20), nullable=False)
    source_filename = Column(String(255), nullable=True)

    total_scanned = Column(Integer, default=0, nullable=False)
    total_anomalies = Column(Integer, default=0, nullable=False)
    total_high_confidence = Column(Integer, default=0, nullable=False)

    threshold_used = Column(Float, nullable=True)
    contamination_used = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="runs")

    high_risk_alerts = relationship(
        "HighRiskAlert",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    low_confidence_anomalies = relationship(
        "LowConfidenceAnomaly",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class HighRiskAlert(Base):
    __tablename__ = "high_risk_alerts"

    alert_id = Column(Integer, primary_key=True, index=True)

    run_id = Column(
        Integer,
        ForeignKey("runs.run_id"),
        nullable=False,
        index=True,
    )

    row_time = Column(DateTime, nullable=False)
    confidence = Column(Float, nullable=False)
    action = Column(String(255), nullable=True)
    reason = Column(Text, nullable=True)

    run = relationship("Run", back_populates="high_risk_alerts")

    feedback = relationship(
        "Feedback",
        back_populates="alert",
        cascade="all, delete-orphan",
    )


class LowConfidenceAnomaly(Base):
    __tablename__ = "low_confidence_anomalies"

    anomaly_id = Column(Integer, primary_key=True, index=True)

    run_id = Column(
        Integer,
        ForeignKey("runs.run_id"),
        nullable=False,
        index=True,
    )

    row_time = Column(DateTime, nullable=False)
    confidence = Column(Float, nullable=False)
    prediction = Column(String(50), nullable=True)
    reason = Column(Text, nullable=True)

    run = relationship(
        "Run",
        back_populates="low_confidence_anomalies",
    )


class Feedback(Base):
    __tablename__ = "feedback"

    feedback_id = Column(Integer, primary_key=True, index=True)

    alert_id = Column(
        Integer,
        ForeignKey("high_risk_alerts.alert_id"),
        nullable=False,
        index=True,
    )

    actual_outcome = Column(String(20), nullable=False)
    feedback_notes = Column(Text, nullable=True)

    submitted_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    alert = relationship(
        "HighRiskAlert",
        back_populates="feedback",
    )