import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# データベースの設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'test.db')}"

# セッションの作成
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# テーブルの作成
Base.metadata.create_all(bind=engine)