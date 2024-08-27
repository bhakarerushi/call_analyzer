import multiprocessing
import time
from transformers import BartForConditionalGeneration, BartTokenizer
from .sqlite_db.database import SessionLocal
import os
import sys

sys.path.append(os.getcwd() + '/app')

from sqlalchemy.orm import Session
from .sqlite_db.models import CallAnalysis
import torch



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_call_summary(db_generator: Session = get_db(), call_log_object=None):

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    db = [db_obj for db_obj in db_generator][0]

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


def detect_fraud_call(db_generator: Session = get_db(), call_log_object=None):
    from .main import model, tokenizer, device

    db = [db_obj for db_obj in db_generator][0]
    print("db obj", db)
    call_summary = call_log_object.call_transcript
    # Tokenize the new texts
    inputs = tokenizer([call_summary], padding=True, truncation=True, return_tensors="pt", max_length=128)

    # Move the inputs to the GPU if available
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Set the model to evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Convert predictions to a list
    predicted_labels = predictions.cpu().numpy().tolist()

    # 1 Fraud
    # 0 Non Fraud
    label_str = ""

    print("labels", predicted_labels)
    # predicted_labels = [1]
    # Interpret and print the results
    # for text, label in zip(new_texts, predicted_labels):
    #     label_str = "Fraudulent" if label == 1 else "Non-Fraudulent"
    #     print(f"Text: {text}\nPredicted Label: {label_str}\n")
    # return label_str
    print("predicted_labels", predicted_labels)
    is_fraud = True if predicted_labels[0] == 1 else False
    print("is_fraud", is_fraud)
    db.bulk_update_mappings(CallAnalysis, [
        {"id": call_log_object.id, "is_fraud": is_fraud}
    ])
    db.commit()
    print("fraud call status saved successfully ..")


if __name__ == "__main__":
    create_process(generate_call_summary, arg1=10, arg2=20)
    print("Main process continues immediately...")
