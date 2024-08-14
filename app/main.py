import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .sqlite_db import models
from .sqlite_db.database import engine
from .api import router

logging.basicConfig(level=logging.INFO)

models.Base.metadata.create_all(bind=engine)

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
