import os
import math
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, EmailStr
from jose import jwt, JWTError
from passlib.context import CryptContext

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import IntegrityError

# ---------------- 設定 ----------------

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")  # 正式請改成環境變數
JWT_ALG = "HS256"
ACCESS_TOKEN_TTL_MIN = 60 * 24 * 7          # 7 天

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dating.db")

engine_kwargs = {"echo": False, "future": True}
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------- 資料表 ----------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(200), unique=False, nullable=True)  # email 可空（也可設 unique=True 視需求）
    password_hash = Column(String(200), nullable=False)

    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("username", name="uq_username"),
    )


def create_db():
    Base.metadata.create_all(bind=engine)

# ---------------- Schemas ----------------

class SignupIn(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None  # ✅ 允許空白/None

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

class LocationIn(BaseModel):
    lat: float
    lng: float

class NearbyIn(BaseModel):
    lat: float
    lng: float
    radius_km: float = 5.0

# ---------------- FastAPI ----------------

app = FastAPI(title="Dating Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# JSON 500 handler（避免回純文字）
@app.exception_handler(Exception)
async def all_exception_handler(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"server error: {str(exc)}"})

# DB Session
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Auth utils
def create_access_token(sub: str, ttl_minutes: int = ACCESS_TOKEN_TTL_MIN) -> str:
    payload = {"sub": sub, "exp": datetime.utcnow() + timedelta(minutes=ttl_minutes)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="無效或過期的憑證")

def current_user(db: Session, token: str) -> User:
    username = decode_token(token)
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="使用者不存在")
    return user

# 簡易 Bearer：從 Authorization: Bearer xxx 取 token
def get_bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    raise HTTPException(status_code=401, detail="缺少 Bearer Token")

# ---------------- Routes ----------------

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.post("/auth/signup")
def signup(payload: SignupIn, db: Session = Depends(get_db)):
    try:
        if db.query(User.id).filter(User.username == payload.username).first():
            raise HTTPException(status_code=400, detail="username 已被使用")

        user = User(
            username=payload.username,
            email=payload.email,
            password_hash=pwd_context.hash(payload.password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"ok": True, "user_id": user.id}
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError:
        db.rollback()
        # 可能是 unique 約束（若你把 email 設 unique=True）
        raise HTTPException(status_code=400, detail="帳號或 email 已存在")
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"detail": f"server error: {str(e)}"})

@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not pwd_context.verify(payload.password, user.password_hash):
        raise HTTPException(status_code=400, detail="帳號或密碼錯誤")
    token = create_access_token(user.username)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me", response_model=MeOut)
def me(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    user = current_user(db, token)
    return MeOut(
        id=user.id, username=user.username, email=user.email,
        lat=user.lat, lng=user.lng, updated_at=user.updated_at
    )

@app.post("/me/location")
def update_location(payload: LocationIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    user = current_user(db, token)
    user.lat = payload.lat
    user.lng = payload.lng
    user.updated_at = datetime.utcnow()
    db.add(user)
    db.commit()
    return {"ok": True}

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

@app.post("/nearby")
def nearby(payload: NearbyIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)

    # 撈出有回報定位的其他人
    users: List[User] = (
        db.query(User)
        .filter(User.id != me.id)
        .filter(User.lat.isnot(None), User.lng.isnot(None))
        .all()
    )
    out: List[Tuple[str, float]] = []
    for u in users:
        d = haversine_km(payload.lat, payload.lng, u.lat, u.lng)
        if d <= payload.radius_km:
            out.append({"username": u.username, "distance_km": round(d, 3)})

    out.sort(key=lambda x: x["distance_km"])
    return {"count": len(out), "users": out}

@app.get("/matches")
def matches(request: Request, db: Session = Depends(get_db)):
    # Demo：先回空清單（後續可改為你真實的配對邏輯）
    token = get_bearer_token(request)
    _ = current_user(db, token)
    reimport os
import math
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, EmailStr, field_validator
import jwt
from jwt import InvalidTokenError
from passlib.context import CryptContext

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import IntegrityError


# ---------------- 設定 ----------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")   # 正式請改成環境變數
JWT_ALG = "HS256"
ACCESS_TOKEN_TTL_MIN = 60 * 24 * 7   # 7 天

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dating.db")

engine_kwargs = {"echo": False, "future": True}
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ---------------- 資料表 ----------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(200), unique=False, nullable=True)  # email 可空
    password_hash = Column(String(200), nullable=False)

    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (UniqueConstraint("username", name="uq_username"),)


def create_db():
    Base.metadata.create_all(bind=engine)


# ---------------- Schemas ----------------
class SignupIn(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None  # ✅ 允許空白/None

    # 把空字串 "" 自動轉成 None，避免 EmailStr 驗證錯
    @field_validator("email", mode="before")
    @classmethod
    def empty_to_none(cls, v):
        if v is None:
            return None
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


class LocationIn(BaseModel):
    lat: float
    lng: float


class NearbyIn(BaseModel):
    lat: float
    lng: float
    radius_km: float = 5.0


# ---------------- FastAPI ----------------
app = FastAPI(title="Dating Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# 統一把未攔截例外變成 JSON（避免回純文字導致前端 JSON 解析錯）
@app.exception_handler(Exception)
async def all_exception_handler(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"server error: {str(exc)}"})


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------- Auth utils (PyJWT) ----------------
def create_access_token(sub: str, ttl_minutes: int = ACCESS_TOKEN_TTL_MIN) -> str:
    payload = {"sub": sub, "exp": datetime.utcnow() + timedelta(minutes=ttl_minutes)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload["sub"]
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="無效或過期的憑證")


def current_user(db: Session, token: str) -> User:
    username = decode_token(token)
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="使用者不存在")
    return user


# Bearer 擷取
def get_bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    raise HTTPException(status_code=401, detail="缺少 Bearer Token")


# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/auth/signup")
def signup(payload: SignupIn, db: Session = Depends(get_db)):
    try:
        if db.query(User.id).filter(User.username == payload.username).first():
            raise HTTPException(status_code=400, detail="username 已被使用")

        user = User(
            username=payload.username,
            email=payload.email,
            password_hash=pwd_context.hash(payload.password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"ok": True, "user_id": user.id}
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="帳號或 email 已存在")
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"detail": f"server error: {str(e)}"})


@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not pwd_context.verify(payload.password, user.password_hash):
        raise HTTPException(status_code=400, detail="帳號或密碼錯誤")
    token = create_access_token(user.username)
    return {"access_token": token, "token_type": "bearer"}


@app.get("/me", response_model=MeOut)
def me(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    user = current_user(db, token)
    return MeOut(
        id=user.id, username=user.username, email=user.email,
        lat=user.lat, lng=user.lng, updated_at=user.updated_at
    )


@app.post("/me/location")
def update_location(payload: LocationIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    user = current_user(db, token)
    user.lat = payload.lat
    user.lng = payload.lng
    user.updated_at = datetime.utcnow()
    db.add(user)
    db.commit()
    return {"ok": True}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))


@app.post("/nearby")
def nearby(payload: NearbyIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)

    users: List[User] = (
        db.query(User)
        .filter(User.id != me.id)
        .filter(User.lat.isnot(None), User.lng.isnot(None))
        .all()
    )
    out: List[Tuple[str, float]] = []
    for u in users:
        d = haversine_km(payload.lat, payload.lng, u.lat, u.lng)
        if d <= payload.radius_km:
            out.append({"username": u.username, "distance_km": round(d, 3)})

    out.sort(key=lambda x: x["distance_km"])
    return {"count": len(out), "users": out}


@app.get("/matches")
def matches(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    _ = current_user(db, token)
    return {"matches": []}


# ---------------- 靜態檔案 / 首頁 ----------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    fp = os.path.join("static", "index.html")
    if os.path.exists(fp):
        return FileResponse(fp)
    return {"service": "dating-prototype", "ok": True}


# ---------------- 啟動前建表 ----------------
create_db()
turn {"matches": []}

# ---------------- 靜態檔案 / 首頁 ----------------

# 提供 static/
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    # 有 static/index.html 就回它；否則回 API 狀態
    fp = os.path.join("static", "index.html")
    if os.path.exists(fp):
        return FileResponse(fp)
    return {"service": "dating-prototype", "ok": True}

# ---------------- 啟動前建表 ----------------
create_db()
