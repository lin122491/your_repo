import os
import json
import math
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Request, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, EmailStr, field_validator

import jwt  # PyJWT
from jwt import InvalidTokenError
from passlib.context import CryptContext

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text,
    ForeignKey, UniqueConstraint, select
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import inspect   # ← 新增：跨資料庫的欄位查詢

# =========================================================
# 基本設定
# =========================================================
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
ACCESS_TOKEN_TTL_MIN = 60 * 24 * 7  # 7 天

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dating.db")

engine_kwargs = {"echo": False, "future": True}
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, connect_args=connect_args, **engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# 密碼雜湊：避開 bcrypt 相依問題
pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


# =========================================================
# ORM 模型
# =========================================================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(200), nullable=True)
    password_hash = Column(String(200), nullable=False)

    # 個人檔案
    nickname = Column(String(120), nullable=True)
    gender = Column(String(10), nullable=True)        # 'male' / 'female' / None
    birthday = Column(String(20), nullable=True)      # ISO date string
    bio = Column(Text, nullable=True)
    city = Column(String(120), nullable=True)
    interests_json = Column(Text, nullable=True)      # JSON 字串，list[str]

    # 定位
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True)
    user_a_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # 較小 id
    user_b_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # 較大 id
    created_at = Column(DateTime, default=datetime.utcnow)

    user_a = relationship("User", foreign_keys=[user_a_id])
    user_b = relationship("User", foreign_keys=[user_b_id])

    __table_args__ = (UniqueConstraint("user_a_id", "user_b_id", name="uq_chat_pair"),)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id"), nullable=False)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    chat = relationship("Chat", foreign_keys=[chat_id])
    sender = relationship("User", foreign_keys=[sender_id])


# =========================================================
# 輕量遷移（修正：SQLAlchemy 2.0 需用 exec_driver_sql 或 inspector）
# =========================================================
def create_db():
    Base.metadata.create_all(engine)


def _table_has_column(conn, table: str, col: str) -> bool:
    # SQLite 走 PRAGMA，但要用 exec_driver_sql（SQLA 2.0）
    if engine.dialect.name == "sqlite":
        res = conn.exec_driver_sql(f"PRAGMA table_info('{table}')").fetchall()
        cols = {r[1] for r in res}  # 第二欄是欄名
        return col in cols
    # 其他資料庫使用 inspector
    ins = inspect(conn)
    try:
        cols = {c["name"] for c in ins.get_columns(table)}
    except Exception:
        return False
    return col in cols


def migrate_users_table():
    with engine.begin() as conn:
        if not _table_has_column(conn, "users", "nickname"):
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN nickname VARCHAR(120)")
        if not _table_has_column(conn, "users", "gender"):
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN gender VARCHAR(10)")
        if not _table_has_column(conn, "users", "birthday"):
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN birthday VARCHAR(20)")
        if not _table_has_column(conn, "users", "bio"):
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN bio TEXT")
        if not _table_has_column(conn, "users", "city"):
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN city VARCHAR(120)")
        if not _table_has_column(conn, "users", "interests_json"):
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN interests_json TEXT")
        if not _table_has_column(conn, "users", "lat"):
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN lat REAL")
        if not _table_has_column(conn, "users", "lng"):
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN lng REAL")
        if not _table_has_column(conn, "users", "updated_at"):
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN updated_at DATETIME")


# =========================================================
# Pydantic Schemas
# =========================================================
class SignupIn(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None

    @field_validator("username", "password")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("不得為空白")
        return v.strip()


class LoginIn(BaseModel):
    username: str
    password: str


class ProfileIn(BaseModel):
    nickname: Optional[str] = None
    gender: Optional[str] = None  # "male" / "female" / None
    birthday: Optional[str] = None  # "YYYY-MM-DD"
    bio: Optional[str] = None
    city: Optional[str] = None
    interests: List[str] = []

    @field_validator("nickname", "gender", "birthday", "bio", "city", mode="before")
    @classmethod
    def empty_to_none(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("gender")
    @classmethod
    def normalize_gender(cls, v):
        if not v:
            return None
        t = str(v).strip().lower()
        if t in ("male", "m", "男"):
            return "male"
        if t in ("female", "f", "女"):
            return "female"
        return None

    @field_validator("interests", mode="before")
    @classmethod
    def normalize_interests(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            out, seen = [], set()
            for p in v:
                if not isinstance(p, str):
                    continue
                pv = p.upper() if any(ch.isalnum() for ch in p) else p
                if pv not in seen:
                    seen.add(pv)
                    out.append(pv)
            return out
        if isinstance(v, str):
            import re
            parts = [x.strip() for x in re.split(r"[,\s，、]+", v) if x.strip()]
            out, seen = [], set()
            for p in parts:
                pv = p.upper() if any(ch.isalnum() for ch in p) else p
                if pv not in seen:
                    seen.add(pv)
                    out.append(pv)
            return out
        return []


class LocationIn(BaseModel):
    lat: float
    lng: float


class NearbyIn(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None
    radius_km: float = 5.0
    gender: Optional[str] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    interests: Optional[List[str]] = None


class ChatStartIn(BaseModel):
    other_username: str


class ChatSendIn(BaseModel):
    text: str


# =========================================================
# FastAPI 初始化
# =========================================================
app = FastAPI(title="Dating Prototype + Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 如需嚴格限制可改你的前端網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =========================================================
# 工具 / 依賴
# =========================================================
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_pw(pw: str) -> str:
    return pwd_ctx.hash(pw)


def verify_pw(pw: str, hashed: str) -> bool:
    try:
        return pwd_ctx.verify(pw, hashed)
    except Exception:
        return False


def make_token(sub: str, ttl_minutes: int = ACCESS_TOKEN_TTL_MIN) -> str:
    payload = {"sub": sub, "exp": datetime.utcnow() + timedelta(minutes=ttl_minutes)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def parse_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload["sub"]
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="無效或過期的授權")


def get_bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    raise HTTPException(status_code=401, detail="缺少 Bearer Token")


def current_user(request: Request, db: Session) -> User:
    username = parse_token(get_bearer_token(request))
    user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="用戶不存在")
    return user


def interests_to_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def list_to_json(lst: Optional[List[str]]) -> str:
    return json.dumps(lst or [], ensure_ascii=False)


def age_from_birthday(bday: Optional[str]) -> Optional[int]:
    if not bday:
        return None
    try:
        d = datetime.strptime(bday, "%Y-%m-%d").date()
    except Exception:
        return None
    today = datetime.utcnow().date()
    age = today.year - d.year - ((today.month, today.day) < (d.month, d.day))
    return max(0, age)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088  # km
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2) + math.cos(p1) * math.cos(p2) * (math.sin(dlmb / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


# =========================================================
# 例外處理（統一 500）
# =========================================================
@app.exception_handler(Exception)
async def all_exception_handler(_r: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"server error: {str(exc)}"})


# =========================================================
# 路由：健康、首頁
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/", response_class=HTMLResponse)
def index():
    fp = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(fp):
        return FileResponse(fp)
    html = """
    <h2>🚀 服務已啟動</h2>
    <p>找不到 <code>static/index.html</code>，請確認前端檔案已放好。</p>
    <ul>
      <li><a href="/docs">Swagger API 文件</a></li>
      <li><a href="/health">健康檢查</a></li>
    </ul>
    """
    return HTMLResponse(html)


# =========================================================
# Auth
# =========================================================
@app.post("/auth/signup")
def signup(payload: SignupIn, db: Session = Depends(get_db)):
    u = User(
        username=payload.username.strip(),
        email=(payload.email or None),
        password_hash=hash_pw(payload.password),
    )
    db.add(u)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="username 已使用")
    return {"ok": True, "user_id": u.id}


@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.execute(select(User).where(User.username == payload.username.strip())).scalar_one_or_none()
    if not user or not verify_pw(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    return {"access_token": make_token(user.username)}


# =========================================================
# Me / Profile
# =========================================================
@app.get("/me")
def me(request: Request, db: Session = Depends(get_db)):
    user = current_user(request, db)
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "nickname": user.nickname,
        "gender": user.gender,
        "birthday": user.birthday,
        "bio": user.bio,
        "city": user.city,
        "interests": interests_to_list(user.interests_json),
        "lat": user.lat,
        "lng": user.lng,
        "updated_at": user.updated_at.isoformat() if user.updated_at else None,
    }


@app.post("/me/profile")
def save_profile(payload: ProfileIn, request: Request, db: Session = Depends(get_db)):
    user = current_user(request, db)

    user.nickname = payload.nickname
    user.gender = payload.gender
    user.birthday = payload.birthday
    user.bio = payload.bio
    user.city = payload.city
    user.interests_json = list_to_json(payload.interests)

    db.add(user)
    db.commit()
    return {"ok": True}


@app.post("/me/location")
def update_location(payload: LocationIn, request: Request, db: Session = Depends(get_db)):
    user = current_user(request, db)
    user.lat = payload.lat
    user.lng = payload.lng
    db.add(user)
    db.commit()
    return {"ok": True}


# =========================================================
# 附近 / 配對
# =========================================================
@app.post("/nearby")
def nearby(payload: NearbyIn, request: Request, db: Session = Depends(get_db)):
    me = current_user(request, db)

    origin_lat = payload.lat if payload.lat is not None else me.lat
    origin_lng = payload.lng if payload.lng is not None else me.lng
    if origin_lat is None or origin_lng is None:
        raise HTTPException(status_code=400, detail="尚未回報定位")

    radius_km = float(payload.radius_km or 5.0)

    stmt = select(User).where(
        User.id != me.id,
        User.lat.isnot(None),
        User.lng.isnot(None)
    )
    users = db.execute(stmt).scalars().all()

    out = []
    for u in users:
        d = haversine_km(origin_lat, origin_lng, u.lat, u.lng)
        if d <= radius_km:
            if payload.gender:
                g = (payload.gender or "").strip().lower()
                want = "male" if g in ("male", "m", "男") else "female" if g in ("female", "f", "女") else None
                if want and (u.gender or "") != want:
                    continue
            age = age_from_birthday(u.birthday)
            if payload.min_age is not None and age is not None and age < payload.min_age:
                continue
            if payload.max_age is not None and age is not None and age > payload.max_age:
                continue
            if payload.interests:
                target = set(interests_to_list(u.interests_json))
                want = set([s.upper() if any(ch.isalnum() for ch in s) else s for s in payload.interests])
                if not target.intersection(want):
                    continue
            out.append({
                "id": u.id,
                "username": u.username,
                "nickname": u.nickname,
                "gender": u.gender,
                "birthday": u.birthday,
                "city": u.city,
                "bio": u.bio,
                "interests": interests_to_list(u.interests_json),
                "lat": u.lat,
                "lng": u.lng,
                "distance_km": round(d, 3),
            })

    out.sort(key=lambda x: x["distance_km"])
    return {"users": out}


@app.get("/matches")
def matches(request: Request, db: Session = Depends(get_db)):
    me = current_user(request, db)
    if me.lat is None or me.lng is None:
        return {"matches": []}

    users = db.execute(select(User).where(User.id != me.id, User.lat.isnot(None), User.lng.isnot(None))).scalars().all()
    data = []
    for u in users:
        d = haversine_km(me.lat, me.lng, u.lat, u.lng)
        data.append({
            "id": u.id,
            "username": u.username,
            "nickname": u.nickname,
            "gender": u.gender,
            "birthday": u.birthday,
            "city": u.city,
            "bio": u.bio,
            "interests": interests_to_list(u.interests_json),
            "lat": u.lat,
            "lng": u.lng,
            "distance_km": round(d, 3),
        })
    data.sort(key=lambda x: x["distance_km"])
    return {"matches": data[:20]}


# =========================================================
# 聊天室
# =========================================================
def get_or_create_chat(db: Session, uid1: int, uid2: int) -> Chat:
    a, b = sorted([uid1, uid2])
    chat = db.execute(select(Chat).where(Chat.user_a_id == a, Chat.user_b_id == b)).scalar_one_or_none()
    if chat:
        return chat
    chat = Chat(user_a_id=a, user_b_id=b)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


def ensure_participant(chat: Chat, user_id: int):
    if user_id not in (chat.user_a_id, chat.user_b_id):
        raise HTTPException(status_code=403, detail="無權限")


@app.post("/chats/start")
def chat_start(payload: ChatStartIn, request: Request, db: Session = Depends(get_db)):
    me = current_user(request, db)
    other = db.execute(select(User).where(User.username == payload.other_username.strip())).scalar_one_or_none()
    if not other:
        raise HTTPException(status_code=404, detail="對方不存在")
    if other.id == me.id:
        raise HTTPException(status_code=400, detail="不能與自己建立聊天")
    chat = get_or_create_chat(db, me.id, other.id)
    return {"chat_id": chat.id}


@app.get("/chats/{chat_id}/messages")
def chat_messages(chat_id: int = Path(..., ge=1), request: Request = None, db: Session = Depends(get_db)):
    me = current_user(request, db)
    chat = db.get(Chat, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="聊天室不存在")
    ensure_participant(chat, me.id)

    msgs = db.execute(
        select(ChatMessage)
        .where(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.created_at.asc())
    ).scalars().all()

    out = []
    for m in msgs:
        out.append({
            "id": m.id,
            "text": m.text,
            "created_at": m.created_at.isoformat(),
            "is_me": (m.sender_id == me.id)
        })
    return {"messages": out}


@app.post("/chats/{chat_id}/send")
def chat_send(
    chat_id: int = Path(..., ge=1),
    payload: ChatSendIn = None,
    request: Request = None,
    db: Session = Depends(get_db)
):
    me = current_user(request, db)
    chat = db.get(Chat, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="聊天室不存在")
    ensure_participant(chat, me.id)

    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="訊息不得為空白")

    msg = ChatMessage(chat_id=chat_id, sender_id=me.id, text=text)
    db.add(msg)
    db.commit()
    return {"ok": True, "message_id": msg.id}


# =========================================================
# 啟動前
# =========================================================
create_db()
migrate_users_table()
