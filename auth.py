import bcrypt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import Request
from fastapi.responses import RedirectResponse

SECRET_KEY = "da-aims-dev-secret-key-2026"
COOKIE_NAME = "da_session"
_signer = URLSafeTimedSerializer(SECRET_KEY)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def make_session_token(user_id: int, role: str) -> str:
    return _signer.dumps({"id": user_id, "role": role})


def read_session(request: Request) -> dict | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    try:
        return _signer.loads(token, max_age=7 * 24 * 3600)
    except (BadSignature, SignatureExpired):
        return None


def set_session_cookie(response: RedirectResponse, user_id: int, role: str) -> None:
    token = make_session_token(user_id, role)
    response.set_cookie(COOKIE_NAME, token, httponly=True, max_age=7 * 24 * 3600, samesite="lax")


def clear_session_cookie(response: RedirectResponse) -> None:
    response.delete_cookie(COOKIE_NAME)
