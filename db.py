from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# データベースの設定
DATABASE_URL = "sqlite:///test.db"

# セッションの作成
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# セッションを取得する関数
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# テーブルの作成
Base.metadata.create_all(bind=engine)