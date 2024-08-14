from sqlalchemy.orm import Session
from .database import SessionLocal
from . import models


def create_call_log(db, payload):
    print(payload)
    db_call_log = models.CallAnalysis(**payload)
    db.add(db_call_log)
    db.commit()
    db.refresh(db_call_log)
    return db_call_log


def get_call_logs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.CallAnalysis).offset(skip).limit(limit).all()


def retrieve_call_log(db: Session, call_log_id: int):
    return db.query(models.CallAnalysis).filter(models.CallAnalysis.id == call_log_id).first()


def delete_call_log(db: Session, call_log_id: int):
    call_log = db.query(models.CallAnalysis).filter(models.CallAnalysis.id == call_log_id).first()
    if call_log:
        db.delete(call_log)
        db.commit()
