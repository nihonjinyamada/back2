from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

# ベースクラスの作成
Base = declarative_base()

class TrainingData(Base):
    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(Text, nullable=False)
    output_text = Column(Text, nullable=False)
