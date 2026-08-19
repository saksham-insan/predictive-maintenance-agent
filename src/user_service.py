from sqlalchemy.orm import Session

from src.auth import hash_password, verify_password
from src.database_models import User


def create_user(db: Session, username: str, password: str) -> User:
    """Create a new user with a bcrypt-hashed password."""

    existing_user = (
        db.query(User)
        .filter(User.username == username)
        .first()
    )

    if existing_user:
        raise ValueError("Username already exists")

    user = User(
        username=username,
        password_hash=hash_password(password),
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def authenticate_user(
    db: Session,
    username: str,
    password: str,
) -> User | None:
    """Return the user if the username/password is valid."""

    user = (
        db.query(User)
        .filter(User.username == username)
        .first()
    )

    if user is None:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user