import os
import tempfile

# 必须在导入任何 app 模块之前指定独立的临时数据库，
# 测试只使用临时 SQLite 文件，不影响本地开发库
_TEST_DB_DIR = tempfile.mkdtemp(prefix="robot_data_test_")
TEST_DB_PATH = os.path.join(_TEST_DB_DIR, "test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, SessionLocal, engine
from main import app


@pytest.fixture(scope="session")
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def db_path():
    return TEST_DB_PATH


@pytest.fixture()
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def clean_tables():
    yield
    session = SessionLocal()
    try:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
    finally:
        session.close()
