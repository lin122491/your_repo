# app.py
import os
import json
import math
from datetime import datetime, date, timedelta
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, EmailStr, field_validator
import jwt
from jwt import InvalidTokenError
from passlib.context import CryptContext

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import IntegrityError


# ---------------- 基本設定 ----------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")  # 正式請改成環境變數
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

pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


# ---------------- 小工具 ----------------
def _normalize_gender(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip()
    if not v:
        return None
    m = {
        "male": "male", "m": "male", "M": "male", "男": "male",
        "female": "female", "f": "female", "F": "female", "女": "female",
    }
    return m.get(v, v.lower())


def _gender_label(g: Optional[str]) -> str:
    return {"male": "男", "female": "女"}.get(g or "", "")


def _parse_birthday_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _calc_age(birthday_str: Optional[str]) -> Optional[int]:
    d = _parse_birthday_to_date(birthday_str)
    if not d:
        return None
    today = date.today()
    age = today.year - d.year - ((today.month, today.day) < (d.month, d.day))
    return max(age, 0)


def _norm_interests(v: Optional[str | List[str]]) -> List[str]:
    """
    接受字串（逗號或空白分隔）、JSON 字串、或字串陣列。
    英文自動轉大寫；中文保留；去除空字串與重複。
    """
    if v is None:
        return []
    items: List[str] = []
    if isinstance(v, list):
        items = [str(x) for x in v]
    else:
        s = str(v).strip()
        if not s:
            return []
        # 嘗試 JSON
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                items = [str(x) for x in arr]
            else:
                items = [s]
        except Exception:
            # 逗號或空白分隔
            parts = [x for x in s.replace("\n", " ").replace("，", ",").split(",")]
            if len(parts) == 1:
                parts = s.split()
            items = [p.strip() for p in parts if p.strip()]

    out: List[str] = []
    seen = set()
    for tag in items:
        # 英文數字全轉大寫；其它（含中文）不變
        if tag.encode("utf-8").isascii():
            t = tag.upper()
        else:
            t = tag
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)


def _verify_password(pw: str, pw_hash: str) -> bool:
    return pwd_ctx.verify(pw, pw_hash)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def _create_access_token(sub: str, ttl_minutes: int = ACCESS_TOKEN_TTL_MIN) -> str:
    payload = {"sub": sub, "exp": datetime.utcnow() + timedelta(minutes=ttl_minutes)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def _decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return str(payload["sub"])
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="無效或過期的授權")


# ---------------- 資料模型 ----------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(200), nullable=True)
    password_hash = Column(String(200), nullable=False)

    nickname = Column(String(100), nullable=True)
    gender = Column(String(20), nullable=True)        # male / female / None
    birthday = Column(String(10), nullable=True)      # YYYY-MM-DD
    bio = Column(Text, nullable=True)
    interests = Column(Text, nullable=True)           # JSON array (str)
    city = Column(String(100), nullable=True)

    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def create_db():
    Base.metadata.create_all(engine)


def migrate_sqlite_users_table():
    """輕量遷移：補缺欄位（lat / lng / updated_at / nickname / gender / birthday / bio / interests / city）"""
    if not DATABASE_URL.startswith("sqlite"):
        return
    with engine.connect() as conn:
        info = conn.exec_driver_sql("PRAGMA table_info('users')").fetchall()
        cols = {r[1] for r in info}
        stmts = []
        add = lambda c, t: stmts.append(f"ALTER TABLE users ADD COLUMN {c} {t}")
        if "lat" not in cols: add("lat", "REAL")
        if "lng" not in cols: add("lng", "REAL")
        if "updated_at" not in cols: add("updated_at", "DATETIME")
        if "nickname" not in cols: add("nickname", "TEXT")
        if "gender" not in cols: add("gender", "TEXT")
        if "birthday" not in cols: add("birthday", "TEXT")
        if "bio" not in cols: add("bio", "TEXT")
        if "interests" not in cols: add("interests", "TEXT")
        if "city" not in cols: add("city", "TEXT")
        for s in stmts:
            try:
                conn.exec_driver_sql(s)
            except Exception:
                pass


# ---------------- Pydantic Schemas ----------------
class SignupIn(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None


class LoginIn(BaseModel):
    username: str
    password: str


class ProfileIn(BaseModel):
    nickname: Optional[str] = None
    gender: Optional[str] = None
    birthday: Optional[str] = None  # YYYY-MM-DD
    bio: Optional[str] = None
    interests: Optional[str | List[str]] = None
    city: Optional[str] = None

    @field_validator("gender", mode="before")
    @classmethod
    def _norm_gender(cls, v):
        return _normalize_gender(v)


class LocationIn(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None


class NearbyIn(BaseModel):
    radius_km: float = 100.0
    gender: Optional[str] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    interest: Optional[str] = None

    @field_validator("gender", mode="before")
    @classmethod
    def _norm_gender(cls, v):
        return _normalize_gender(v)


# ---------------- FastAPI 初始化 ----------------
app = FastAPI(title="Dating Prototype", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(Exception)
async def all_exception_handler(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"server error: {str(exc)}"})


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_bearer_token(request: Request) -> str:
    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    raise HTTPException(status_code=401, detail="缺少 Bearer Token")


def current_user(db: Session = Depends(get_db), token: str = Depends(get_bearer_token)) -> User:
    username = _decode_token(token)
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="用戶不存在")
    return user


def user_to_dict(u: User) -> dict:
    interests = []
    try:
        interests = json.loads(u.interests or "[]")
        if not isinstance(interests, list):
            interests = []
    except Exception:
        interests = []
    return {
        "id": u.id, "username": u.username, "nickname": u.nickname,
        "gender": u.gender, "gender_label": _gender_label(u.gender),
        "birthday": u.birthday, "age": _calc_age(u.birthday),
        "bio": u.bio, "interests": interests,
        "city": u.city, "lat": u.lat, "lng": u.lng,
        "updated_at": u.updated_at.isoformat() if u.updated_at else None,
    }


# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/auth/signup")
def signup(payload: SignupIn, db: Session = Depends(get_db)):
    if db.query(User.id).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="username 已使用")
    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=_hash_password(payload.password),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="username 已使用")
    return {"ok": True, "user_id": user.id}


@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not _verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    token = _create_access_token(user.username)
    return {"access_token": token, "token_type": "bearer"}


@app.get("/me")
def me(u: User = Depends(current_user)):
    return user_to_dict(u)


@app.post("/me/profile")
def update_profile(payload: ProfileIn, db: Session = Depends(get_db), u: User = Depends(current_user)):
    if payload.nickname is not None:
        u.nickname = payload.nickname.strip()
    if payload.gender is not None:
        u.gender = _normalize_gender(payload.gender)
    if payload.birthday is not None:
        # 僅存 YYYY-MM-DD 字串
        b = _parse_birthday_to_date(payload.birthday)
        u.birthday = b.isoformat() if b else None
    if payload.bio is not None:
        u.bio = payload.bio.strip()
    if payload.interests is not None:
        arr = _norm_interests(payload.interests)
        u.interests = json.dumps(arr, ensure_ascii=False)
    if payload.city is not None:
        u.city = payload.city.strip()
    db.add(u)
    db.commit()
    db.refresh(u)
    return {"ok": True, "me": user_to_dict(u)}


@app.post("/me/location")
def update_location(payload: LocationIn, db: Session = Depends(get_db), u: User = Depends(current_user)):
    if payload.lat is not None:
        u.lat = payload.lat
    if payload.lng is not None:
        u.lng = payload.lng
    u.updated_at = datetime.utcnow()
    db.add(u)
    db.commit()
    return {"ok": True}


@app.post("/nearby")
def nearby(payload: NearbyIn, db: Session = Depends(get_db), u: User = Depends(current_user)):
    if u.lat is None or u.lng is None:
        raise HTTPException(status_code=400, detail="尚未上報定位")

    q = db.query(User).filter(User.id != u.id)
    if payload.gender:
        q = q.filter(User.gender == payload.gender)
    users = q.all()

    items = []
    for x in users:
        if x.lat is None or x.lng is None:
            continue

        age = _calc_age(x.birthday)
        if payload.min_age is not None and (age is None or age < payload.min_age):
            continue
        if payload.max_age is not None and (age is None or age > payload.max_age):
            continue

        if payload.interest:
            try:
                arr = json.loads(x.interests or "[]")
                arr = [str(i) for i in arr]
            except Exception:
                arr = []
            want = _norm_interests(payload.interest)
            if want and not any(w in arr for w in want):
                continue

        dist = _haversine_km(u.lat, u.lng, x.lat, x.lng)
        if dist <= (payload.radius_km or 100):
            d = user_to_dict(x)
            d["distance_km"] = round(dist, 3)
            items.append(d)

    items.sort(key=lambda r: r["distance_km"])
    return {"total": len(items), "users": items}


@app.get("/matches")
def matches(db: Session = Depends(get_db), u: User = Depends(current_user)):
    """簡單版本：直接用 100km 內的人當作『我的配對』"""
    dummy = NearbyIn(radius_km=100.0)
    return nearby(dummy, db, u)


# ---------- 靜態首頁 ----------
@app.get("/", response_class=HTMLResponse)
def index():
    fp = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(fp):
        return FileResponse(fp)
    # 備援簡單首頁
    html = f"""
    <!doctype html><html><meta charset="utf-8"/>
    <title>Dating Prototype</title>
    <h2>🚀 服務已啟動</h2>
    <p>找不到 <code>static/index.html</code>，請前往 <a href="/docs" target="_blank">Swagger API</a> 操作，
       或把前端頁面放到 <code>static/index.html</code>。</p>
    <ul>
      <li><a href="/docs" target="_blank">Swagger API 文件</a></li>
    </ul>
    """
    return HTMLResponse(html)


# ---------- 啟動前 ----------
create_db()
migrate_sqlite_users_table()
