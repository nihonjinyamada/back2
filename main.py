import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import re
import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import crud
from db import get_db
import models
import schemas
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://front-eta-khaki.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ログ設定
logging.basicConfig(level=logging.INFO)

# モデルのロード
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE_PATH = os.path.join(DATA_DIR, "data.json")

if not os.path.exists(MODEL_DIR):
    logging.error(f"Model directory does not exist: {MODEL_DIR}")
    raise RuntimeError(f"Model directory does not exist: {MODEL_DIR}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    logging.info("Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, local_files_only=False)
    
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR, local_files_only=False).to(device)

    if torch.cuda.is_available():
        model = model.half()
    else:
        model = model.to(torch.float32)

    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")


# リクエストスキーマ
class ModelRequest(BaseModel):
    ジャンル: str
    技術分野: str

# 推論エンドポイント
@app.post("/generate/")
async def generate_text(request: ModelRequest, db: Session = Depends(get_db)):
    try:
        logging.info(f"Received Request: {request.dict()}")  

        genre = request.ジャンル
        tech = request.技術分野

        logging.info(f"ジャンル: {genre}, 技術分野: {tech}")  

        prompt = f"ジャンル: {genre}\n技術分野: {tech}"

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                num_beams=1,         
                do_sample=True,      
                temperature=0.7,     
                top_p=0.7,           
                early_stopping=True,
                return_dict_in_generate=False
            )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logging.info(f"Generated Text: {output_text}")  

        dataset_url_match = re.findall(r"https?://[^\s]+", output_text)
        dataset_desc_match = re.search(r"データ概要:\s*(.+?)\s*アプリまたはツール名:", output_text, re.DOTALL)
        app_name_match = re.search(r"アプリまたはツール名:\s*(.+?)\s*アプリまたはツールの説明:", output_text, re.DOTALL)
        app_desc_match = re.search(r"アプリまたはツールの説明:\s*(.*)", output_text, re.DOTALL)

        response_data = {
            "データセットURL": dataset_url_match[0] if dataset_url_match else "データセットURLが見つかりません。",
            "データ概要": dataset_desc_match.group(1).strip() if dataset_desc_match else "データ概要が提供されていません。",
            "アプリまたはツール名": app_name_match.group(1).strip() if app_name_match else "アプリ名が見つかりません。",
            "アプリまたはツールの説明": app_desc_match.group(1).strip() if app_desc_match else "アプリの説明がありません。"
        }

        logging.info(f"Processed Response Data: {response_data}") 

        return JSONResponse(status_code=200, content=response_data)

    except Exception as e:
        logging.error(f"Error in generate_text: {str(e)}")  
        raise HTTPException(status_code=500, detail={"エラー": "サーバーエラー"})

@app.post("/training_data/")
async def create_training_data(data: List[schemas.TrainingDataCreate], db: Session = Depends(get_db)):
    for item in data:
        if isinstance(item.output_text, dict):
            item.output_text = json.dumps(item.output_text, ensure_ascii=False)
        elif not isinstance(item.output_text, str):
            raise ValueError(f"output_text should be of type 'str' or 'dict', but got {type(item.output_text)}")
    
    return crud.create_training_data(db=db, data=data)

@app.get("/training_data/", response_model=List[schemas.TrainingDataResponse])
async def read_training_data(db: Session = Depends(get_db)):
    training_data = crud.get_training_data(db)

    json_path = os.path.join(DATA_DIR, "data.json")
    
    data_to_save = [{"input_text": item.input_text, "output_text": item.output_text} for item in training_data]
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    
    return training_data

@app.delete("/training_data/range")
async def delete_training_data_range(start_id: int, end_id: int, db: Session = Depends(get_db)):
    data_items = db.query(models.TrainingData).filter(models.TrainingData.id >= start_id, models.TrainingData.id <= end_id).all()
    if not data_items:
        raise HTTPException(status_code=404, detail="指定されたIDのデータが見つかりません")

    for item in data_items:
        db.delete(item)
    db.commit()

    training_data = crud.get_training_data(db)
    data_to_save = [{"input_text": item.input_text, "output_text": item.output_text} for item in training_data]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    
    return JSONResponse(status_code=200, content={"メッセージ": f"ID {start_id} から {end_id} のデータを削除しました。"})

# エラーハンドリング
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    logging.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(status_code=500, content={"メッセージ": "内部サーバーエラー", "エラー": str(exc)})
