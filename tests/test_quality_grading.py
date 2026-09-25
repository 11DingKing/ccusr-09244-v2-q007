"""质量分级策略校验与批量重算测试。

覆盖：
- 临界分数落在阈值边界时归入更高等级，且结果可重复；
- 权重浮点和的容差接受/拒绝、负权重与超区间取值；
- 阈值严格有序（颠倒、相等均拒绝），错误响应指出无效字段；
- 缺少标注的作业按标注零分参与加权；
- 指定不存在的作业编号时整体拒绝、不更新任何数据；
- 批量评分/提交中途异常时回滚，原等级保持不变；
- “重启”（换引擎重新打开同一数据库）后等级保持不变，重复重算结果一致。
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models import Annotation, OperationData, RobotModel, Scene, Skill
from app.routers import analytics
from app.services.grading import OperationNotFoundError, grade_operations
from app.services.scoring import (
    GradePolicyError,
    determine_grade,
    validate_grade_policy,
)


# ---------- 构造数据 ----------

FULL_TRAJECTORY = {"waypoints": [{"x": 1}], "joint_angles": [0.1, 0.2]}
FULL_PERCEPTION = {f"sensor_{i}": {"value": i} for i in range(5)}


def make_operation(*, complete: bool = True) -> OperationData:
    now = datetime.now(timezone.utc)
    if complete:
        return OperationData(
            robot_model_id=1,
            scene_id=1,
            skill_id=1,
            motion_trajectory=FULL_TRAJECTORY,
            perception_records=FULL_PERCEPTION,
            grasp_result={"success": True},
            timestamp_start=now,
            timestamp_end=now,
            duration_ms=100,
            environment_conditions={"temperature": 25},
            hardware_status={"battery": 0.9},
        )
    return OperationData(
        robot_model_id=1,
        scene_id=1,
        skill_id=1,
        motion_trajectory={},
        perception_records={},
        grasp_result=None,
        timestamp_start=now,
        timestamp_end=now,
        duration_ms=None,
        environment_conditions=None,
        hardware_status=None,
    )


def make_annotation(operation_id: int, quality_score: float) -> Annotation:
    return Annotation(
        operation_data_id=operation_id,
        is_success=True,
        review_status="approved",
        annotation_quality_score=quality_score,
    )


def seed_reference_data(db: Session) -> None:
    db.add(RobotModel(id=1, name="机型A", manufacturer="厂商"))
    db.add(Scene(id=1, name="场景A", category="制造"))
    db.add(Skill(id=1, name="技能A", category="抓取"))
    db.commit()


def make_memory_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture()
def db_session():
    engine = make_memory_engine()
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    seed_reference_data(session)
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


STANDARD_POLICY_KWARGS = dict(
    completeness_weight=0.5,
    annotation_weight=0.5,
    grade_a_threshold=0.9,
    grade_b_threshold=0.7,
    grade_c_threshold=0.5,
)


# ---------- 策略校验：区间、容差、字段级错误 ----------

def test_validate_policy_accepts_defaults():
    policy = validate_grade_policy(**STANDARD_POLICY_KWARGS)
    assert policy.completeness_weight == 0.5
    assert policy.threshold_map()["grade_a"] == 0.9


def test_validate_policy_accepts_float_sum_within_tolerance():
    # 0.1 + 0.9000000001 在二进制浮点下得到 1.0000000001，不等于 1.0 但仅差约 1e-10（容差内）
    w_c = 0.1
    w_a = 0.9000000001
    assert w_c + w_a != 1.0  # 前提：浮点和确实不精确
    policy = validate_grade_policy(w_c, w_a, 0.9, 0.7, 0.5)
    assert policy.completeness_weight == pytest.approx(0.1, abs=1e-12)


def test_validate_policy_rejects_float_sum_outside_tolerance():
    with pytest.raises(GradePolicyError) as exc_info:
        validate_grade_policy(0.5, 0.5 + 2e-6, 0.9, 0.7, 0.5)
    assert "weights" in exc_info.value.field_errors


def test_validate_policy_rejects_negative_and_out_of_range_weights():
    with pytest.raises(GradePolicyError) as exc_info:
        validate_grade_policy(-0.1, 1.2, 0.9, 0.7, 0.5)
    errors = exc_info.value.field_errors
    assert "completeness_weight" in errors
    assert "annotation_weight" in errors
    assert "weights" in errors  # 和（1.1）也不等于一，一并指出


def test_validate_policy_rejects_inverted_and_equal_thresholds():
    with pytest.raises(GradePolicyError) as exc_info:
        validate_grade_policy(0.5, 0.5, 0.5, 0.7, 0.9)
    errors = exc_info.value.field_errors
    assert "grade_a_threshold" in errors
    assert "grade_b_threshold" in errors
    assert "grade_c_threshold" in errors

    with pytest.raises(GradePolicyError):
        validate_grade_policy(0.5, 0.5, 0.8, 0.8, 0.5)


def test_validate_policy_rejects_non_finite_values():
    with pytest.raises(GradePolicyError) as exc_info:
        validate_grade_policy(float("nan"), float("inf"), 0.9, 0.7, 0.5)
    assert "completeness_weight" in exc_info.value.field_errors
    assert "annotation_weight" in exc_info.value.field_errors


# ---------- 临界分数与可重复边界等级 ----------

def test_determine_grade_boundaries_belong_to_higher_grade():
    thresholds = {"grade_a": 0.9, "grade_b": 0.7, "grade_c": 0.5}
    assert determine_grade(0.9, thresholds) == "A"
    assert determine_grade(0.7, thresholds) == "B"
    assert determine_grade(0.5, thresholds) == "C"
    assert determine_grade(0.4999, thresholds) == "D"


@pytest.mark.parametrize(
    "annotation_score,expected_grade",
    [
        (0.8, "A"),    # 1.0*0.5 + 0.8*0.5 = 0.9，恰好 A 边界
        (0.799, "B"),  # 0.8995，差一点也不能升入 A
        (0.4, "B"),    # 0.7，恰好 B 边界
        (0.0, "C"),    # 0.5，恰好 C 边界
    ],
)
def test_boundary_grades_are_deterministic(db_session, annotation_score, expected_grade):
    op = make_operation(complete=True)
    db_session.add(op)
    db_session.commit()
    db_session.add(make_annotation(op.id, annotation_score))
    db_session.commit()

    policy = validate_grade_policy(**STANDARD_POLICY_KWARGS)
    first = grade_operations(db_session, policy, [op.id])
    second = grade_operations(db_session, policy, [op.id])

    assert first.items[0].data_grade == expected_grade
    # 同一策略重复重算，分数与等级完全一致
    assert first.items == second.items


def test_weighted_score_uses_decimal_to_avoid_float_drift(db_session):
    # 完整度 1/6、无标注、权重各半时，二进制浮点会产生长尾误差；边界判定必须稳定
    op = make_operation(complete=False)
    op.grasp_result = {"success": True}  # 仅一个完整度字段 → 1/6
    db_session.add(op)
    db_session.commit()

    policy = validate_grade_policy(**STANDARD_POLICY_KWARGS)
    result = grade_operations(db_session, policy, [op.id])
    item = result.items[0]
    # round(1/6, 4) = 0.1667，加权后 0.0834（Decimal 量化，无浮点尾差）
    assert item.completeness_score == 0.1667
    assert item.quality_score == 0.0834
    assert item.data_grade == "D"
    assert Decimal(str(item.quality_score)) == Decimal("0.0834")


# ---------- 缺少标注 ----------

def test_missing_annotation_scores_zero(db_session):
    op = make_operation(complete=True)
    db_session.add(op)
    db_session.commit()

    policy = validate_grade_policy(
        completeness_weight=0.9,
        annotation_weight=0.1,
        grade_a_threshold=0.9,
        grade_b_threshold=0.7,
        grade_c_threshold=0.5,
    )
    result = grade_operations(db_session, policy, [op.id])
    item = result.items[0]
    # 无标注：标注质量按 0 处理，1.0*0.9 + 0*0.1 = 0.9，恰好 A 边界
    assert item.quality_score == 0.9
    assert item.data_grade == "A"


# ---------- 指定不存在的作业编号：整体不更新 ----------

def test_missing_operation_id_aborts_entire_batch(db_session):
    op = make_operation(complete=True)
    db_session.add(op)
    db_session.commit()
    db_session.add(make_annotation(op.id, 0.4))
    db_session.commit()
    # 预置一个旧等级
    op.data_grade = "A"
    op.quality_score = 0.99
    db_session.commit()

    policy = validate_grade_policy(**STANDARD_POLICY_KWARGS)
    with pytest.raises(OperationNotFoundError) as exc_info:
        grade_operations(db_session, policy, [op.id, 99999])
    assert exc_info.value.missing_ids == [99999]

    db_session.rollback()
    db_session.expire_all()
    refreshed = db_session.get(OperationData, op.id)
    # 存在的作业也不能被部分更新
    assert refreshed.data_grade == "A"
    assert refreshed.quality_score == 0.99


# ---------- 批量中途异常：回滚、原等级保持 ----------

def test_scoring_failure_mid_batch_changes_nothing(db_session, monkeypatch):
    ops = [make_operation(complete=True) for _ in range(3)]
    db_session.add_all(ops)
    db_session.commit()
    for op in ops:
        op.data_grade = "B"
        op.quality_score = 0.71
    db_session.commit()

    import app.services.grading as grading_module

    calls = {"n": 0}
    real_compute = grading_module.compute_operation_quality

    def flaky_compute(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("评分服务暂时不可用")
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(grading_module, "compute_operation_quality", flaky_compute)

    policy = validate_grade_policy(**STANDARD_POLICY_KWARGS)
    with pytest.raises(RuntimeError):
        grade_operations(db_session, policy)

    db_session.rollback()
    db_session.expire_all()
    for op in ops:
        refreshed = db_session.get(OperationData, op.id)
        assert refreshed.data_grade == "B"
        assert refreshed.quality_score == 0.71


def test_commit_failure_mid_batch_rolls_back(db_session, monkeypatch):
    ops = [make_operation(complete=True) for _ in range(3)]
    db_session.add_all(ops)
    db_session.commit()
    for op in ops:
        op.data_grade = "C"
        op.quality_score = 0.55
    db_session.commit()

    original_commit = Session.commit
    attempts = {"n": 0}

    def failing_commit(self, *args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("模拟提交失败：磁盘已满")
        return original_commit(self, *args, **kwargs)

    monkeypatch.setattr(Session, "commit", failing_commit)

    policy = validate_grade_policy(**STANDARD_POLICY_KWARGS)
    with pytest.raises(RuntimeError):
        grade_operations(db_session, policy)

    db_session.rollback()
    db_session.expire_all()
    for op in ops:
        refreshed = db_session.get(OperationData, op.id)
        assert refreshed.data_grade == "C"
        assert refreshed.quality_score == 0.55


# ---------- 重启后等级保持 + 重复重算一致 ----------

def test_grades_persist_across_restart(tmp_path):
    db_path = tmp_path / "restart_quality.db"
    url = f"sqlite:///{db_path}"

    engine = create_engine(url)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    seed_reference_data(session)

    ops = [
        make_operation(complete=True),
        make_operation(complete=True),
        make_operation(complete=False),
    ]
    session.add_all(ops)
    session.commit()
    session.add(make_annotation(ops[0].id, 0.8))  # → A
    session.add(make_annotation(ops[1].id, 0.4))  # → B
    session.commit()

    policy = validate_grade_policy(**STANDARD_POLICY_KWARGS)
    before = {item.operation_id: item for item in grade_operations(session, policy).items}
    session.close()
    engine.dispose()

    # “重启”：用全新引擎/会话打开同一个数据库文件
    engine2 = create_engine(url)
    SessionLocal2 = sessionmaker(bind=engine2)
    session2 = SessionLocal2()
    try:
        persisted = {
            op.id: (op.data_grade, op.quality_score, op.completeness_score)
            for op in session2.query(OperationData).all()
        }
        for op_id, item in before.items():
            assert persisted[op_id] == (
                item.data_grade,
                item.quality_score,
                item.completeness_score,
            )

        # 重启后用同一策略再算一遍：结果必须完全一致
        rerun = {item.operation_id: item for item in grade_operations(session2, policy).items}
        assert rerun == before
    finally:
        session2.close()
        engine2.dispose()


# ---------- HTTP 接口层 ----------

@pytest.fixture()
def client(db_session):
    fastapi_app = FastAPI()
    fastapi_app.include_router(analytics.router)
    fastapi_app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(fastapi_app)


def test_api_rejects_invalid_policy_with_field_errors(client):
    response = client.post(
        "/quality/grade-operations",
        json={
            "completeness_weight": -0.2,
            "annotation_weight": 1.2,
            "grade_a_threshold": 0.5,
            "grade_b_threshold": 0.7,
            "grade_c_threshold": 0.9,
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    field_errors = detail["field_errors"]
    for field in (
        "completeness_weight",
        "annotation_weight",
        "grade_a_threshold",
        "grade_b_threshold",
        "grade_c_threshold",
    ):
        assert field in field_errors


def test_api_rejects_nonexistent_operation_ids(client, db_session):
    op = make_operation(complete=True)
    db_session.add(op)
    db_session.commit()
    op.data_grade = "A"
    db_session.commit()

    response = client.post(
        "/quality/grade-operations?operation_ids=99999",
        json={
            "completeness_weight": 0.5,
            "annotation_weight": 0.5,
            "grade_a_threshold": 0.9,
            "grade_b_threshold": 0.7,
            "grade_c_threshold": 0.5,
        },
    )
    assert response.status_code == 404
    body = response.json()["detail"]
    assert body["missing_operation_ids"] == [99999]
    assert "operation_ids" in body["field_errors"]

    db_session.expire_all()
    assert db_session.get(OperationData, op.id).data_grade == "A"


def test_api_valid_policy_boundary_grading(client, db_session):
    op = make_operation(complete=True)
    db_session.add(op)
    db_session.commit()
    db_session.add(make_annotation(op.id, 0.8))  # 恰好 0.9 → A
    db_session.commit()

    response = client.post(
        "/quality/grade-operations",
        json={
            "completeness_weight": 0.5,
            "annotation_weight": 0.5,
            "grade_a_threshold": 0.9,
            "grade_b_threshold": 0.7,
            "grade_c_threshold": 0.5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["graded_count"] == 1
    assert body["grade_distribution"]["A"] == 1
    assert body["items"][0]["quality_score"] == 0.9
    assert body["items"][0]["data_grade"] == "A"
