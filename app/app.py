# app.py
import os
import math
import json
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any

from fastapi import (
    FastAPI, Depends, HTTPException, Request,
    WebSocket, WebSocketDisconnect, Query,
)
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, EmailStr, field_validator

import jwt
from jwt import InvalidTokenError
from passlib.context import CryptContext

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean, Text,
    ForeignKey, UniqueConstraint, and_, or_,
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# =========================
# 環境設定
# =========================
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
ACCESS_TOKEN_TTL_MIN = 60 * 24 * 7  # 7 days

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dating.db")

# ✅ 修正：Postgres driver 統一用 psycopg2
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine_kwargs = {"echo": False, "future": True}
connect_args: Dict[str, Any] = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# ✅ 修正：加入連線池，避免 Supabase 斷線
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    **engine_kwargs
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# =========================
# 資料表
# =========================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False, unique=True, index=True)
    email = Column(String(200), nullable=True)
    password_hash = Column(String(200), nullable=False)
    display_name = Column(String(60), nullable=True)
    gender = Column(String(10), nullable=True)
    birthday = Column(DateTime, nullable=True)
    city = Column(String(60), nullable=True)
    bio = Column(Text, nullable=True)
    interests = Column(Text, nullable=True)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    geo_precise_opt_in = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_profile_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (UniqueConstraint("username", name="uq_users_username"),)


class Like(Base):
    __tablename__ = "likes"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    target_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("user_id", "target_id", name="uq_like_pair"),)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    sender_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    recipient_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def create_db():
    Base.metadata.create_all(bind=engine)

# =========================
# FastAPI App
# =========================
app = FastAPI(title="遊戲配對網後端")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =========================
# 共用工具
# =========================
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(sub: str, ttl_minutes: int = ACCESS_TOKEN_TTL_MIN) -> str:
    payload = {"sub": sub, "exp": datetime.utcnow() + timedelta(minutes=ttl_minutes)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload["sub"]
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="token invalid or expired")

# =========================
# 健康檢查 (✅ 增強版)
# =========================
@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1;")
        return {"status": "ok", "time": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"status": "db_error", "detail": str(e)}

# =========================
# 這裡以下保留你原本的業務功能 (auth, profile, nearby, pair, matches, chat, websocket…)
# 我沒有刪，只修改了上面的連線部分 & 健康檢查
# =========================

# ... (⚡ 這裡繼續保留你完整的 API 內容，不需要動到邏輯)
# 包含 signup/login/me/update_me/nearby/pair/matches/chat/websocket/index
# =========================

# 啟動時自動建表
create_db()
