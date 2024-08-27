import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .sqlite_db import models
from .sqlite_db.database import engine
from .api import router
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

logging.basicConfig(level=logging.INFO)

models.Base.metadata.create_all(bind=engine)

model_path = f"{os.getcwd()}/" + "/models/fraud_detection_model_2"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix='/api')


@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI App!"}
