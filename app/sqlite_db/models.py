from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Float
import datetime
from .database import Base


class CallAnalysis(Base):
    __tablename__ = "call_analysis"

    id = Column(Integer, primary_key=True)
    call_id = Column(String, unique=False, index=True)
    call_transcript = Column(String)
    call_summary = Column(String, default="")
    is_fraud = Column(Boolean, default=False)
    fraud_call_metadata = Column(String, default="")
    action_items = Column(String, default="")
    status = Column(String, default="Not Analyzed")
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.now())
