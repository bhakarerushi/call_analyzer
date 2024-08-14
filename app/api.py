import json
from fastapi import APIRouter, Request, status, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from .sqlite_db.database import SessionLocal
from .sqlite_db.crud import create_call_log, get_call_logs, retrieve_call_log, delete_call_log
from .sqlite_db.schemas import CallLog
from datetime import datetime

router = APIRouter()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/perform_call_analysis")
def perform_call_analysis(request_data: CallLog, db: Session = Depends(get_db)):
    print("req_data", request_data)
    data = {
        "call_id": request_data.call_id,
        "status": "Not Analyzed",
        "start_time": datetime.strptime(request_data.start_time, "%Y-%m-%d %H:%M:%S.%f"),
        "end_time": datetime.strptime(request_data.end_time, "%Y-%m-%d %H:%M:%S.%f"),
        "duration": request_data.duration,
        "call_transcript": request_data.call_transcript
    }
    call_log = create_call_log(db, data)
    # model
    return JSONResponse("Call Transcript uploaded successfully.", status_code=status.HTTP_200_OK)


@router.get("/call_logs")
def list_call_logs(db: Session = Depends(get_db)):
    call_logs = get_call_logs(db)
    call_log_list = []
    for call in call_logs:
        call_log_list.append({
            "id": call.id,
            "call_id": call.call_id,
            "call_summary": call.call_summary,
            "is_fraud": call.is_fraud,
            "action_items": call.action_items,
            "status": call.status,
            "start_time": str(call.start_time),
            "end_time": str(call.end_time),
            "duration": call.duration
        })
    return JSONResponse(call_log_list, status_code=status.HTTP_200_OK)


@router.get("/call_logs/{id}")
def retrieve_call_details(id: int, db: Session = Depends(get_db)):
    call = retrieve_call_log(db=db, call_log_id=id)
    data = {
        "id": call.id,
        "call_id": call.call_id,
        "call_summary": call.call_summary,
        "is_fraud": call.is_fraud,
        "action_items": call.action_items,
        "status": call.status,
        "start_time": str(call.start_time),
        "end_time": str(call.end_time),
        "duration": call.duration
    }
    return JSONResponse(data, status_code=status.HTTP_200_OK)


@router.delete("/call_logs/{id}")
def retrieve_call_details(id: int, db: Session = Depends(get_db)):
    delete_call_log(db=db, call_log_id=id)
    return JSONResponse("Calllog deleted successfully.", status_code=status.HTTP_200_OK)
