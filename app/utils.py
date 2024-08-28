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

import pandas as pd
import pickle
import librosa
import numpy as np
import whisper
from transformers import pipeline
import warnings
import copy


warnings.filterwarnings("ignore")

classifier = pipeline('sentiment-analysis')

svm_path = f"{os.getcwd()}/" + "/models/svm.pkl"

with open(svm_path, 'rb') as file:
    SVM_model = pickle.load(file)

svm_model = SVM_model


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_call_summary(db_generator: Session = get_db(), call_log_object=None,transcript=""):

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    db = [db_obj for db_obj in db_generator][0]
    print(f"call_obj_transcript - {call_log_object.call_transcript}, transcript - {transcript}")

    transcript = transcript if transcript else call_log_object.call_transcript

    print("call transcript", transcript)

    # Preprocess the text
    input_ids = tokenizer.encode(transcript, return_tensors="pt", max_length=1024, truncation=True)

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






# ## Feature extraction using Librosa (Duration, intensity, pitches, spectral characteristics, tempo, fluency)
def extract_features(call_log_obj ,db_generator: Session = get_db()):

    db = [db_obj for db_obj in db_generator][0]
    audio_path = call_log_obj.audio_file_name

    y, sr = librosa.load(audio_path, duration=10.0)  # Load the audio file, 30 seconds duration

    features = {}

    features['duration'] = librosa.get_duration(y=y, sr=sr)

    features['intensity'] = np.mean(librosa.feature.rms(y=y).flatten())

    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    features['pitch'] = np.mean(pitches) if len(pitches) > 0 else 0

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    for i, mfcc in enumerate(mfccs_mean):
        features[f'mfcc_{i + 1}'] = mfcc

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    features['tempo'] = tempo[0]

    zero_crossings = librosa.zero_crossings(y, pad=False)
    features['fluency'] = np.sum(zero_crossings)

    # 7. Extracting Keywords using Speech Recognition

    model = whisper.load_model("base")
    result = model.transcribe(audio_path)

    features['transcribed_text'] = result['text']
    print("transcript >>>>> ", result['text'])
    audio_features = copy.deepcopy(features)
    audio_features.pop("transcribed_text")
    print("transcript >>>>> 2", result['text'])
    db.bulk_update_mappings(CallAnalysis, [
        {"id": call_log_obj.id, "call_transcript": f"{features['transcribed_text']}",
         "fraud_call_metadata":f"{features}"
         }

    ])
    db.commit()
    print("call transcript saved successfully ..")
    print("saved call transcript >>>>>>>>>>>>", call_log_obj.call_transcript)
    return features


# Apply sentiment analysis to each text in the dataset
def get_sentiment(text):
    result = classifier(text)[0]
    sentiment_score = 1 if result['label'] == 'POSITIVE' else 0  # 1 for positive, 0 for negative
    return sentiment_score

def predict_fraud(call_log_object, features, db_generator: Session = get_db()):

    db = [db_obj for db_obj in db_generator][0]
    # Feature extraction (sentiment and text length)
    # features = extract_features(audio_path)
    # print(features)
    # print("transcript", features["transcribed_text"])
    x_df = pd.DataFrame([features])
    x_df['sentiment'] = x_df['transcribed_text'].apply(get_sentiment)
    # df['fraud_call'] = [1 if x == 0 else 0 for x in df['sentiment']]
    x_df['text_length'] = x_df['transcribed_text'].apply(len)

    x_df = x_df.loc[:, x_df.columns != 'transcribed_text']
    # print(x_df)
    # Predict fraud or non-fraud using SVM
    prediction = svm_model.predict(x_df)
    print(prediction)
    is_fraud = True if prediction[0] == 1 else False
    print("is_fraud", is_fraud)
    db.bulk_update_mappings(CallAnalysis, [
        {"id": call_log_object.id, "is_fraud": is_fraud}
    ])
    db.commit()
    print("fraud call status saved successfully ..")

def perform_analysis(call_log_obj):
    print("inside >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. ")
    audio_features = extract_features(call_log_obj)

    # generate call summary
    create_process(predict_fraud, call_log_object=call_log_obj, features=audio_features)
    #
    # fraud call detection
    create_process(generate_call_summary, call_log_object=call_log_obj,transcript=audio_features['transcribed_text'])



if __name__ == "__main__":
    create_process(generate_call_summary, arg1=10, arg2=20)
    print("Main process continues immediately...")
