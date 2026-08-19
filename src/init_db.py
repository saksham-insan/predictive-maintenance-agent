from src.database import engine, Base
from src import database_models  # noqa: F401


def init_database():
    Base.metadata.create_all(bind=engine)
    print("DATABASE TABLES CREATED SUCCESSFULLY")


if __name__ == "__main__":
    init_database()