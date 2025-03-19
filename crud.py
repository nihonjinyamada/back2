from sqlalchemy.orm import Session
from models import TrainingData
from schemas import TrainingDataCreate
from fastapi import HTTPException

def create_training_data(db: Session, data: list[TrainingDataCreate]):
    try:
        db_items = [
            TrainingData(
                input_text=item.input_text,
                output_text=item.get_output_text()  
            ) for item in data
        ]
        db.add_all(db_items)
        db.commit()

        for db_item in db_items:
            db.refresh(db_item)

        return db_items
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def get_training_data(db: Session):
    return db.query(TrainingData).all()

def search_similar_training_data(db: Session, input_text: str):
    return db.query(TrainingData).filter(TrainingData.input_text.contains(input_text)).all()
