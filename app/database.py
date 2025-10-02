from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ✅ 換成你的 Supabase 資料庫連線字串
# 注意：SQLAlchemy 正確寫法要加上 +psycopg2
DATABASE_URL = "postgresql+psycopg2://postgres.fnqdcmgpcdewyynoqdhy:Call142425lemon@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres?sslmode=require"

# 連線池設定
engine = create_engine(
    DATABASE_URL,
    pool_size=5,        # 最多保留 5 條連線
    max_overflow=10,    # 超過的部分臨時建立
    pool_timeout=30,    # 最多等 30 秒
    pool_recycle=1800   # 1800 秒 (30 分鐘) 回收一次連線，避免 server 強制關閉
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
