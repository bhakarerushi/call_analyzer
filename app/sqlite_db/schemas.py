from pydantic import BaseModel


class CallLog(BaseModel):
    call_id: str
    start_time: str
    end_time: str
    duration: float
    call_transcript: str
