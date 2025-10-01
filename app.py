# app.py
import os
import json
import math
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, EmailStr, field_validator
import jwt
from jwt import InvalidTokenError

from passlib.context import CryptContext  # 使用 PBKDF2，避開 bcrypt 版本干擾

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import IntegrityError


# ---------------- 基本設定 ----------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")  # 正式環境請改用環境變數
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

# 密碼雜湊：使用 PBKDF2（不需要 bcrypt 原生擴充）
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


# ---------------- 資料模型 ----------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(200), nullable=True)
    password_hash = Column(String(200), nullable=False)

    # 位置
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    updated_at = Column(DateTime, nullable=True)

    # 個人檔案
    nickname = Column(String(100), nullable=True)
    gender = Column(String(20), nullable=True)     # 'male' | 'female' | None
    birthday = Column(String(20), nullable=True)   # YYYY-MM-DD（簡單存字串）
    bio = Column(String(500), nullable=True)
    city = Column(String(100), nullable=True)
    interests = Column(String(2000), nullable=True)  # JSON 字串 list[str]


def create_db():
    Base.metadata.create_all(bind=engine)


def migrate_sqlite_users_table():
    """
    輕量遷移：若缺少欄位（lat/lng/updated_at/nickname/gender/birthday/bio/city/interests）就補。
    不刪資料、不中斷服務。
    """
    if not DATABASE_URL.startswith("sqlite"):
        return
    with engine.connect() as conn:
        try:
            rows = conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()
            cols = {r[1] for r in rows}  # r[1] = column name

            stmts = []
            if "lat" not in cols:
                stmts.append("ALTER TABLE users ADD COLUMN lat REAL")
            if "lng" not in cols:
                stmts.append("ALTER TABLE users ADD COLUMN lng REAL")
            if "updated_at" not in cols:
                stmts.append("ALTER TABLE users ADD COLUMN updated_at DATETIME")
            if "nickname" not in cols:
                stmts.append("ALTER TABLE users ADD COLUMN nickname TEXT")
            if "gender" not in cols:
                stmts.append("ALTER TABLE users ADD COLUMN gender TEXT")
            if "birthday" not in cols:
                stmts.append("ALTER TABLE users ADD COLUMN birthday TEXT")
            if "bio" not in cols:
                stmts.append("ALTER TABLE users ADD COLUMN bio TEXT")
            if "city" not in cols:
                stmts.append("ALTER TABLE users ADD COLUMN city TEXT")
            if "interests" not in cols:
                stmts.append("ALTER TABLE users ADD COLUMN interests TEXT")

            for s in stmts:
                conn.exec_driver_sql(s)

            if stmts:
                print("[migrate] users 補欄位：", ", ".join(s.split()[3] for s in stmts))
        except Exception as e:
            print("[migrate] 跳過/錯誤：", e)


# ---------------- Pydantic Schemas ----------------
class SignupIn(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None

    @field_validator("email", mode="before")
    @classmethod
    def blank_email_as_none(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


class LoginIn(BaseModel):
    username: str
    password: str


class ProfileIn(BaseModel):
    nickname: Optional[str] = None
    gender: Optional[str] = None          # '男' / '女' / 'male' / 'female' / ''
    birthday: Optional[str] = None        # YYYY-MM-DD
    bio: Optional[str] = None
    city: Optional[str] = None
    interests: Optional[List[str]] = None # 字串陣列


class LocationIn(BaseModel):
    lat: float
    lng: float


class NearbyIn(BaseModel):
    lat: float
    lng: float
    radius_km: float = 5.0


class MeOut(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    nickname: Optional[str] = None
    gender: Optional[str] = None          # 'male' / 'female' / None
    gender_text: Optional[str] = None     # '男' / '女' / None
    birthday: Optional[str] = None
    age: Optional[int] = None
    bio: Optional[str] = None
    city: Optional[str] = None
    interests: List[str] = []
    lat: Optional[float] = None
    lng: Optional[float] = None
    updated_at: Optional[datetime] = None


# ---------------- FastAPI 本體與中介層 ----------------
app = FastAPI(title="交友APP 後端（修正版）")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 前後端分離時方便測試；正式可改白名單
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- 通用工具 ----------------
@app.exception_handler(Exception)
async def all_exception_handler(_: Request, exc: Exception):
    # 統一 500 型式
    return JSONResponse(status_code=500, content={"detail": f"server error: {str(exc)}"})


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_access_token(sub: str, ttl_minutes: int = ACCESS_TOKEN_TTL_MIN) -> str:
    payload = {
        "sub": sub,
        "exp": datetime.utcnow() + timedelta(minutes=ttl_minutes),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload["sub"]
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="無效或過期的授權")


def get_bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    if not auth:
        raise HTTPException(status_code=401, detail="缺少 Authorization 標頭")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authorization 格式應為 Bearer")
    return auth.split(" ", 1)[1].strip()


def current_user(db: Session, token: str) -> User:
    username = decode_access_token(token)
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="用戶不存在")
    return user


def user_to_public(u: User) -> dict:
    """輸出給前端的公開欄位（不含敏感資料）"""
    interests: List[str] = []
    if u.interests:
        try:
            interests = json.loads(u.interests)
            if not isinstance(interests, list):
                interests = []
        except Exception:
            interests = []
    return {
        "id": u.id,
        "username": u.username,
        "email": u.email,
        "nickname": u.nickname,
        "gender": u.gender,
        "gender_text": ("男" if u.gender == "male" else "女" if u.gender == "female" else None),
        "birthday": u.birthday,
        "bio": u.bio,
        "city": u.city,
        "interests": interests,
    }


def calc_age(birthday: Optional[str]) -> Optional[int]:
    if not birthday:
        return None
    try:
        y, m, d = [int(x) for x in birthday.split("-")]
        born = datetime(y, m, d)
        today = datetime.utcnow()
        age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        return max(age, 0)
    except Exception:
        return None


def build_me_dict(u: User) -> dict:
    d = user_to_public(u)
    d.update({
        "age": calc_age(u.birthday),
        "lat": u.lat,
        "lng": u.lng,
        "updated_at": u.updated_at,
    })
    return d


def normalize_gender(g: Optional[str]) -> Optional[str]:
    if not g:
        return None
    g = g.strip().lower()
    if g in ("male", "男", "m"):
        return "male"
    if g in ("female", "女", "f"):
        return "female"
    return None


def normalize_interests(items: Optional[List[str]]) -> List[str]:
    if not items:
        return []
    out = []
    for it in items:
        if not isinstance(it, str):
            continue
        s = it.strip()
        if not s:
            continue
        # 規則：英數轉大寫；中文保留
        out.append(s.upper())
    # 去重（保留順序）
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float, R: float = 6371.0) -> float:
    if None in (lat1, lon1, lat2, lon2):
        return float("inf")
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/auth/signup")
def signup(payload: SignupIn, db: Session = Depends(get_db)):
    if db.query(User.id).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="username 已使用")

    pw_hash = pwd_context.hash(payload.password)
    u = User(
        username=payload.username,
        email=payload.email,
        password_hash=pw_hash,
        updated_at=datetime.utcnow(),
    )
    try:
        db.add(u)
        db.commit()
        db.refresh(u)
        return {"ok": True, "user_id": u.id}
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="username 已使用")
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"detail": f"server error: {str(e)}"})


@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.username == payload.username).first()
    if not u or not pwd_context.verify(payload.password, u.password_hash):
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    token = create_access_token(u.username)
    return {"access_token": token, "token_type": "bearer"}


@app.get("/me", response_model=MeOut)
def me(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    u = current_user(db, token)
    return build_me_dict(u)


@app.post("/me/profile")
def set_profile(payload: ProfileIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    u = current_user(db, token)

    if payload.nickname is not None:
        u.nickname = payload.nickname.strip() or None
    if payload.gender is not None:
        u.gender = normalize_gender(payload.gender)
    if payload.birthday is not None:
        u.birthday = payload.birthday.strip() or None
    if payload.bio is not None:
        u.bio = payload.bio.strip() or None
    if payload.city is not None:
        u.city = payload.city.strip() or None
    if payload.interests is not None:
        ints = normalize_interests(payload.interests)
        u.interests = json.dumps(ints, ensure_ascii=False)

    u.updated_at = datetime.utcnow()
    db.add(u)
    db.commit()
    return {"ok": True, "me": build_me_dict(u)}  # ← 修正：不再呼叫 me(Request, db)


@app.post("/me/location")
def update_location(payload: LocationIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    u = current_user(db, token)
    u.lat = payload.lat
    u.lng = payload.lng
    u.updated_at = datetime.utcnow()
    db.add(u)
    db.commit()
    return {"ok": True}


@app.post("/nearby")
def nearby(payload: NearbyIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me_u = current_user(db, token)

    users = (
        db.query(User)
        .filter(User.id != me_u.id)
        .filter(User.lat.isnot(None), User.lng.isnot(None))
        .all()
    )

    out: List[dict] = []
    for u in users:
        d = haversine_km(payload.lat, payload.lng, u.lat, u.lng)
        if d <= payload.radius_km:
            rec = user_to_public(u)
            rec["age"] = calc_age(u.birthday)
            rec["distance_km"] = round(d, 3)
            out.append(rec)
    out.sort(key=lambda x: x["distance_km"])
    return {"users": out}


@app.get("/matches")
def matches(request: Request, db: Session = Depends(get_db)):
    # 簡易示範：目前與 /nearby 邏輯分離，由前端傳經緯度呼叫 /nearby
    token = get_bearer_token(request)
    _ = current_user(db, token)
    return {"matches": []}


# ---------------- 靜態與首頁 ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# /static 供前端檔案（若有）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


INDEX_HTML = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>交友APP 後端</title>
<style>body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans TC",Arial;margin:32px;color:#222}</style>
</head><body>
<h2>🚀 服務已啟動</h2>
<p>若 <code>static/index.html</code> 存在，後端會直接提供該前端頁面。</p>
<ul>
<li><a href="/docs" target="_blank">Swagger API 文件</a></li>
<li><a href="/health" target="_blank">健康檢查</a></li>
</ul>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    fp = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(fp):
        return FileResponse(fp)
    return HTMLResponse(INDEX_HTML)


# ---------------- 啟動前：建立/遷移資料表 ----------------
create_db()
migrate_sqlite_users_table()
