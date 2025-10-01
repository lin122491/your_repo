import os, json, math, re
from datetime import datetime, timedelta, date
from typing import Optional, List, Tuple, Literal

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, EmailStr, field_validator
import jwt
from jwt import InvalidTokenError

from passlib.context import CryptContext

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.exc import IntegrityError

# ---------------- 基本設定 ----------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")  # 正式環境請改環境變數
JWT_ALG = "HS256"
ACCESS_TOKEN_TTL_MIN = 12 * 60

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dating.db")
engine_kwargs = {"echo": False, "future": True}
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# ---------------- 資料模型 ----------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(200), nullable=True)
    password_hash = Column(String(200), nullable=False)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Profile(Base):
    __tablename__ = "profiles"
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    display_name = Column(String(80), nullable=True)
    gender = Column(String(16), nullable=True)  # male / female / other
    birthday = Column(String(10), nullable=True)  # YYYY-MM-DD (簡單存字串)
    bio = Column(String(2000), nullable=True)
    interests = Column(String(4000), nullable=True)  # 存 JSON list[str]
    city = Column(String(80), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", lazy="joined")


class Like(Base):
    __tablename__ = "likes"
    id = Column(Integer, primary_key=True)
    from_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    to_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("from_user_id", "to_user_id", name="uq_like_from_to"),)


class Match(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True)
    user_a_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user_b_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("user_a_id", "user_b_id", name="uq_match_pair"),)


def create_db():
    Base.metadata.create_all(bind=engine)

def migrate_sqlite_users_table():
    # 補 users.lat/lng/updated_at 欄位（若從舊版升上來）
    if not DATABASE_URL.startswith("sqlite"):
        return
    with engine.connect() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(users);").fetchall()
        cols = {r[1] for r in rows}
        stmts = []
        if "lat" not in cols: stmts.append("ALTER TABLE users ADD COLUMN lat REAL;")
        if "lng" not in cols: stmts.append("ALTER TABLE users ADD COLUMN lng REAL;")
        if "updated_at" not in cols: stmts.append("ALTER TABLE users ADD COLUMN updated_at DATETIME;")
        for s in stmts:
            conn.exec_driver_sql(s)

create_db()
migrate_sqlite_users_table()

# ---------------- Schemas ----------------
class SignupIn(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None

    @field_validator("email", mode="before")
    @classmethod
    def empty_as_none(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


class LoginIn(BaseModel):
    username: str
    password: str


class MeOut(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    updated_at: Optional[datetime] = None


Gender = Literal["male", "female", "other", "MALE", "FEMALE", "OTHER"]  # 前端可能傳大寫
class ProfileIn(BaseModel):
    display_name: Optional[str] = None
    gender: Optional[str] = None
    birthday: Optional[str] = None       # YYYY-MM-DD
    bio: Optional[str] = None
    interests: Optional[List[str]] = None
    city: Optional[str] = None

class ProfileOut(ProfileIn):
    # 同 ProfileIn，多回 user 位置信息
    user_id: int
    lat: Optional[float] = None
    lng: Optional[float] = None


class LocationIn(BaseModel):
    lat: float
    lng: float


class NearbyIn(BaseModel):
    lat: float
    lng: float
    radius_km: float = 5.0
    # 可選基本篩選（之後可擴充）
    gender: Optional[str] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    interests: Optional[List[str]] = None  # 交集檢查（粗略）


class LikeIn(BaseModel):
    to_username: str


# ---------------- App 本體 ----------------
app = FastAPI(title="交友 APP 後端（表單配對＋精簡個人檔案）")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ----------- 共用工具 -----------
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

def current_user(db: Session, token: str) -> User:
    username = parse_token(token)
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="用戶不存在")
    return user

ASCII_RE = re.compile(r"[A-Za-z]")

def normalize_interests(items: Optional[List[str]]) -> List[str]:
    if not items:
        return []
    out = []
    seen = set()
    for raw in items:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        # 英文字轉大寫；中文保留原樣
        if ASCII_RE.search(s):
            s_norm = s.upper()
        else:
            s_norm = s  # 中文或無英文字：原樣
        key = s_norm  # 略做去重 key
        if key not in seen:
            seen.add(key)
            out.append(s_norm)
    return out

def age_from_birthday(birthday_str: Optional[str]) -> Optional[int]:
    if not birthday_str:
        return None
    try:
        y, m, d = [int(x) for x in birthday_str.split("-")]
        b = date(y, m, d)
        today = date.today()
        age = today.year - b.year - ((today.month, today.day) < (b.month, b.day))
        return max(age, 0)
    except Exception:
        return None

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float, R: float = 6371.0) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ----------- 例外處理 -----------
@app.exception_handler(Exception)
async def all_exception_handler(_c: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"server error: {str(exc)}"})


# ----------- Routes -----------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# Auth
@app.post("/auth/signup")
def signup(payload: SignupIn, db: Session = Depends(get_db)):
    if db.query(User.id).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="username 已使用")
    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=pwd_ctx.hash(payload.password),
    )
    db.add(user)
    db.flush()  # 拿到 user.id

    # 建立空白 Profile
    prof = Profile(user_id=user.id, updated_at=datetime.utcnow())
    db.add(prof)

    try:
        db.commit()
        return {"ok": True, "user_id": user.id}
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="註冊失敗（可能是重複）")

@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not pwd_ctx.verify(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    return {"access_token": make_token(user.username), "token_type": "bearer"}

# 我
@app.get("/me", response_model=MeOut)
def me(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    u = current_user(db, token)
    return MeOut(id=u.id, username=u.username, email=u.email, lat=u.lat, lng=u.lng, updated_at=u.updated_at)

@app.post("/me/location")
def update_location(payload: LocationIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    me.lat, me.lng = payload.lat, payload.lng
    me.updated_at = datetime.utcnow()
    db.add(me)
    db.commit()
    return {"ok": True}

# Profile
@app.get("/me/profile", response_model=ProfileOut)
def get_my_profile(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    prof = db.query(Profile).filter(Profile.user_id == me.id).first()
    if not prof:
        prof = Profile(user_id=me.id)
        db.add(prof)
        db.commit()
    interests = json.loads(prof.interests) if prof.interests else []
    return ProfileOut(
        user_id=me.id,
        display_name=prof.display_name,
        gender=prof.gender,
        birthday=prof.birthday,
        bio=prof.bio,
        interests=interests,
        city=prof.city,
        lat=me.lat, lng=me.lng
    )

@app.put("/me/profile", response_model=ProfileOut)
def update_my_profile(payload: ProfileIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    prof = db.query(Profile).filter(Profile.user_id == me.id).first()
    if not prof:
        prof = Profile(user_id=me.id)
        db.add(prof)

    # 正規化興趣：英文字轉大寫、中文原樣、去重
    interests_norm = None
    if payload.interests is not None:
        interests_norm = normalize_interests(payload.interests)

    for k in ["display_name", "gender", "birthday", "bio", "city"]:
        v = getattr(payload, k)
        if v is not None:
            setattr(prof, k, v)

    if interests_norm is not None:
        prof.interests = json.dumps(interests_norm, ensure_ascii=False)

    prof.updated_at = datetime.utcnow()
    db.add(prof)
    db.commit()

    interests = json.loads(prof.interests) if prof.interests else []
    return ProfileOut(
        user_id=me.id,
        display_name=prof.display_name,
        gender=prof.gender,
        birthday=prof.birthday,
        bio=prof.bio,
        interests=interests,
        city=prof.city,
        lat=me.lat, lng=me.lng
    )

@app.get("/users/{username}", response_model=ProfileOut)
def get_user_public(username: str, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.username == username).first()
    if not u:
        raise HTTPException(status_code=404, detail="用戶不存在")
    prof = db.query(Profile).filter(Profile.user_id == u.id).first()
    interests = json.loads(prof.interests) if prof and prof.interests else []
    return ProfileOut(
        user_id=u.id,
        display_name=prof.display_name if prof else None,
        gender=prof.gender if prof else None,
        birthday=prof.birthday if prof else None,
        bio=prof.bio if prof else None,
        interests=interests,
        city=prof.city if prof else None,
        lat=u.lat, lng=u.lng
    )

# Nearby（回傳列表，前端可以用「勾選表單」做配對）
@app.post("/nearby")
def nearby(payload: NearbyIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)

    candidates = (
        db.query(User, Profile)
        .join(Profile, Profile.user_id == User.id, isouter=True)
        .filter(User.id != me.id)
        .filter(User.lat.isnot(None), User.lng.isnot(None))
        .all()
    )

    out = []
    for u, prof in candidates:
        dist = haversine_km(payload.lat, payload.lng, u.lat or 0.0, u.lng or 0.0)
        if dist > payload.radius_km:
            continue

        # 篩選條件
        g = (prof.gender or "").lower() if prof else ""
        if payload.gender and payload.gender.lower() not in ("", g):
            continue

        age = age_from_birthday(prof.birthday if prof else None)
        if payload.min_age is not None and (age is None or age < payload.min_age):
            continue
        if payload.max_age is not None and (age is None or age > payload.max_age):
            continue

        ints = json.loads(prof.interests) if prof and prof.interests else []
        if payload.interests:
            want = set(normalize_interests(payload.interests))
            have = set(normalize_interests(ints))
            if not (want & have):
                continue

        out.append({
            "username": u.username,
            "display_name": prof.display_name if prof else None,
            "gender": prof.gender if prof else None,
            "age": age,
            "city": prof.city if prof else None,
            "interests": ints,
            "distance_km": round(dist, 3),
        })

    out.sort(key=lambda x: x["distance_km"])
    return {"users": out}

# 喜歡 / 建立配對
@app.post("/like")
def like_user(payload: LikeIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)

    target = db.query(User).filter(User.username == payload.to_username).first()
    if not target:
        raise HTTPException(status_code=404, detail="對方不存在")
    if target.id == me.id:
        raise HTTPException(status_code=400, detail="不可喜歡自己")

    # 寫入 like（忽略重複）
    try:
        db.add(Like(from_user_id=me.id, to_user_id=target.id))
        db.commit()
    except IntegrityError:
        db.rollback()
        # 已經點過喜歡就忽略

    # 檢查是否互相喜歡 → 建立 match
    existed = db.query(Like).filter(
        Like.from_user_id == target.id, Like.to_user_id == me.id
    ).first()

    matched = False
    match_id = None
    if existed:
        a, b = sorted([me.id, target.id])
        # 已存在就忽略
        m = db.query(Match).filter(Match.user_a_id == a, Match.user_b_id == b).first()
        if not m:
            try:
                m = Match(user_a_id=a, user_b_id=b)
                db.add(m)
                db.commit()
                match_id = m.id
                matched = True
            except IntegrityError:
                db.rollback()
        else:
            matched = True
            match_id = m.id

    return {"ok": True, "matched": matched, "match_id": match_id}

@app.get("/matches")
def list_matches(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)

    ms = db.query(Match).filter(
        (Match.user_a_id == me.id) | (Match.user_b_id == me.id)
    ).all()

    results = []
    for m in ms:
        other_id = m.user_b_id if m.user_a_id == me.id else m.user_a_id
        other = db.query(User).filter(User.id == other_id).first()
        prof = db.query(Profile).filter(Profile.user_id == other_id).first()
        results.append({
            "match_id": m.id,
            "username": other.username if other else None,
            "display_name": prof.display_name if prof else None,
            "gender": prof.gender if prof else None,
            "city": prof.city if prof else None,
            "created_at": m.created_at.isoformat() if m.created_at else None
        })
    return {"matches": results}


# ----------- 首頁（供靜態前端） -----------
FALLBACK_HTML = """<!doctype html>
<html lang="zh-Hant"><meta charset="utf-8">
<title>交友APP 後端啟動</title>
<body style="font-family:system-ui;padding:24px">
<h1>🚀 服務已啟動</h1>
<ul>
  <li><a href="/docs" target="_blank">Swagger API 文件</a></li>
  <li><a href="/static/index.html" target="_blank">開啟前端頁面</a></li>
</ul>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    fp = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(fp):
        return FileResponse(fp)
    return HTMLResponse(FALLBACK_HTML)
