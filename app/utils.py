import multiprocessing
import time
from transformers import BartForConditionalGeneration, BartTokenizer
from .sqlite_db.database import SessionLocal
from fastapi import APIRouter, Request, status, Depends
from sqlalchemy.orm import Session
from .sqlite_db.models import CallAnalysis


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_call_summary(db_generator: Session = get_db(), call_log_object=None):
    """
    pip install transformers
    pip install torch
    pip install tensorflow
    """
    print("db", db_generator)
    db = [db_obj for db_obj in db_generator][0]
    print("db obj", db)
    # Load pre-trained model and tokenizer
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    # Input text
    # text = """
    # We begin our analysis on the neighborhood by considering the degree (k) and
    # the strength (s) distributions. Degree and strength distributions give informa-
    # tion about the level of interaction of a mobile user on the basis of the number of
    # people contacting him/her, the number of people s/he contacts and how often.
    # 295

    # """
    # Preprocess the text
    input_ids = tokenizer.encode(call_log_object.call_transcript, return_tensors="pt", max_length=1024, truncation=True)

    # Generate summary
    summary_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("summary", summary)
    print("call obj id ", call_log_object.id)
    db.bulk_update_mappings(CallAnalysis, [
        {"id": call_log_object.id, "call_summary": summary, "status": "Success"}
    ])
    db.commit()
    print("call summary saved successfully ..")


def test_func(arg1, arg2):
    time.sleep(2)  # Simulate a time-consuming task
    result = arg1 + arg2
    print(f"Result: {result}")


def create_process(target_func, **kwargs):
    print("args - ", kwargs)
    process = multiprocessing.Process(target=target_func, kwargs=kwargs)
    process.start()
    # No join here, so the main process won't wait for the subprocess to finish


if __name__ == "__main__":
    create_process(generate_call_summary, arg1=10, arg2=20)
    print("Main process continues immediately...")
