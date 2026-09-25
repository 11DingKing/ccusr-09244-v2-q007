from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Annotation, OperationData, RobotModel, Scene, Skill
from app.services.scoring import (
    GradePolicyError,
    determine_grade,
    validate_grading_policy,
    compute_operation_quality,
)


DEFAULT_POLICY = {
    "completeness_weight": 0.5,
    "annotation_weight": 0.5,
    "grade_a_threshold": 0.9,
    "grade_b_threshold": 0.7,
    "grade_c_threshold": 0.5,
}

GRADE_URL = "/api/v1/quality/grade-operations"


@pytest.fixture()
def base_entities(db_session):
    robot_model = RobotModel(name="RM-Test", manufacturer="ACME")
    scene = Scene(name="装配场景", category="生产制造")
    skill = Skill(name="抓取", category="操作")
    db_session.add_all([robot_model, scene, skill])
    db_session.commit()
    return {
        "robot_model_id": robot_model.id,
        "scene_id": scene.id,
        "skill_id": skill.id,
    }


def make_operation(db, ids, full=True, annotation_quality=None, review_status="pending"):
    """完整数据完整度为 1.0；full=False 时完整度为 0.5。"""
    operation = OperationData(
        robot_model_id=ids["robot_model_id"],
        scene_id=ids["scene_id"],
        skill_id=ids["skill_id"],
        robot_serial="SN-TEST",
        motion_trajectory={"waypoints": [[0.0, 0.0, 0.0]], "joint_angles": [0.1]},
        perception_records={"rgb": {}, "depth": {}, "imu": {}, "force": {}, "audio": {}},
        grasp_result={"success": True} if full else None,
        timestamp_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        timestamp_end=datetime(2025, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
        duration_ms=30000,
        environment_conditions={"temperature": 25} if full else None,
        hardware_status={"servo": "ok"} if full else None,
    )
    db.add(operation)
    db.flush()
    if annotation_quality is not None:
        db.add(Annotation(
            operation_data_id=operation.id,
            is_success=True,
            review_status=review_status,
            annotation_quality_score=annotation_quality,
        ))
    db.commit()
    db.refresh(operation)
    return operation


def grade_all(client, policy=None):
    return client.post(GRADE_URL, json=policy or DEFAULT_POLICY)


def stored_grades(db_session):
    db_session.expire_all()
    return {
        op.id: (op.data_grade, op.quality_score)
        for op in db_session.query(OperationData).all()
    }


class TestValidateGradingPolicy:
    def test_valid_policy_returns_exact_decimal_policy(self):
        policy = validate_grading_policy(0.3, 0.7, 0.9, 0.7, 0.5)
        assert policy.completeness_weight == Decimal("0.3")
        assert policy.annotation_weight == Decimal("0.7")
        assert policy.thresholds() == {
            "grade_a": Decimal("0.9"),
            "grade_b": Decimal("0.7"),
            "grade_c": Decimal("0.5"),
        }

    def test_weight_sum_tolerance_boundary(self):
        # 偏差恰好等于容差 1e-6 时仍可接受
        policy = validate_grading_policy(0.5, 0.499999, 0.9, 0.7, 0.5)
        assert policy.annotation_weight == Decimal("0.499999")
        # 超过容差则拒绝，并指出 weight_sum 字段
        with pytest.raises(GradePolicyError) as exc_info:
            validate_grading_policy(0.5, 0.499998, 0.9, 0.7, 0.5)
        assert "weight_sum" in exc_info.value.errors

    def test_all_invalid_fields_reported_together(self):
        with pytest.raises(GradePolicyError) as exc_info:
            validate_grading_policy(-0.1, 1.1, 0.4, 0.8, 0.9)
        assert set(exc_info.value.errors) == {
            "completeness_weight",
            "annotation_weight",
            "grade_a_threshold",
            "grade_b_threshold",
            "grade_c_threshold",
        }

    def test_thresholds_must_be_strictly_decreasing(self):
        with pytest.raises(GradePolicyError):
            validate_grading_policy(0.5, 0.5, 0.7, 0.7, 0.5)
        with pytest.raises(GradePolicyError):
            validate_grading_policy(0.5, 0.5, 0.9, 0.5, 0.5)

    def test_non_numeric_and_out_of_range_values_rejected(self):
        with pytest.raises(GradePolicyError) as exc_info:
            validate_grading_policy("abc", 0.5, 0.9, 0.7, 0.5)
        assert "completeness_weight" in exc_info.value.errors
        with pytest.raises(GradePolicyError) as exc_info:
            validate_grading_policy(0.5, 0.5, 1.5, 0.7, 0.5)
        assert "grade_a_threshold" in exc_info.value.errors


class TestDetermineGrade:
    def test_exact_threshold_falls_into_higher_grade(self):
        thresholds = {"grade_a": 0.9, "grade_b": 0.7, "grade_c": 0.5}
        assert determine_grade(0.9, thresholds) == "A"
        assert determine_grade(0.7, thresholds) == "B"
        assert determine_grade(0.5, thresholds) == "C"

    def test_one_quantum_below_threshold_falls_into_lower_grade(self):
        thresholds = {"grade_a": 0.9, "grade_b": 0.7, "grade_c": 0.5}
        assert determine_grade(0.8999, thresholds) == "B"
        assert determine_grade(0.6999, thresholds) == "C"
        assert determine_grade(0.4999, thresholds) == "D"

    def test_decimal_and_string_inputs_compare_exactly(self):
        thresholds = {"grade_a": "0.9", "grade_b": "0.7", "grade_c": "0.5"}
        assert determine_grade("0.7", thresholds) == "B"
        assert determine_grade(Decimal("0.9"), thresholds) == "A"


class TestPolicyValidationApi:
    def test_float_weight_sum_within_tolerance_accepted(self, client, db_session, base_entities):
        make_operation(db_session, base_entities)
        # 0.3333333 + 0.6666666 = 0.9999999，与 1 的偏差在容差内
        policy = {**DEFAULT_POLICY, "completeness_weight": 0.3333333, "annotation_weight": 0.6666666}
        resp = grade_all(client, policy)
        assert resp.status_code == 200
        assert resp.json()["graded_count"] == 1

    def test_weight_sum_beyond_tolerance_rejected(self, client, db_session, base_entities):
        make_operation(db_session, base_entities)
        policy = {**DEFAULT_POLICY, "completeness_weight": 0.5, "annotation_weight": 0.4999}
        resp = grade_all(client, policy)
        assert resp.status_code == 400
        assert "weight_sum" in resp.json()["detail"]["invalid_fields"]

    def test_out_of_range_weights_rejected(self, client, db_session, base_entities):
        make_operation(db_session, base_entities)
        # -0.2 + 1.2 = 1.0，权重和合法但取值越界，两个字段都应被指出
        policy = {**DEFAULT_POLICY, "completeness_weight": -0.2, "annotation_weight": 1.2}
        resp = grade_all(client, policy)
        assert resp.status_code == 400
        fields = resp.json()["detail"]["invalid_fields"]
        assert "completeness_weight" in fields
        assert "annotation_weight" in fields

    def test_reversed_thresholds_rejected_and_nothing_updated(self, client, db_session, base_entities):
        operation = make_operation(db_session, base_entities, annotation_quality=0.8)
        assert grade_all(client).status_code == 200
        before = stored_grades(db_session)
        assert before[operation.id][0] == "A"

        # A 级阈值低于 B 级阈值：顺序颠倒，必须整体拒绝且不改写已有等级
        policy = {**DEFAULT_POLICY, "grade_a_threshold": 0.4, "grade_b_threshold": 0.8}
        resp = grade_all(client, policy)
        assert resp.status_code == 400
        fields = resp.json()["detail"]["invalid_fields"]
        assert "grade_a_threshold" in fields
        assert "grade_b_threshold" in fields
        assert stored_grades(db_session) == before

    def test_equal_thresholds_rejected(self, client, db_session, base_entities):
        make_operation(db_session, base_entities)
        policy = {**DEFAULT_POLICY, "grade_a_threshold": 0.8, "grade_b_threshold": 0.8}
        resp = grade_all(client, policy)
        assert resp.status_code == 400
        assert "grade_a_threshold" in resp.json()["detail"]["invalid_fields"]


class TestGradingProcess:
    def test_boundary_scores_map_to_defined_grades(self, client, db_session, base_entities):
        op_a = make_operation(db_session, base_entities, annotation_quality=0.8)      # 0.9000 -> A
        op_b = make_operation(db_session, base_entities, annotation_quality=0.4)      # 0.7000 -> B
        op_round = make_operation(db_session, base_entities, annotation_quality=0.3999)  # 0.69995 -> 0.7000 -> B
        op_c = make_operation(db_session, base_entities, annotation_quality=0.3998)   # 0.6999 -> C
        op_c2 = make_operation(db_session, base_entities, annotation_quality=0.0)     # 0.5000 -> C
        op_d = make_operation(db_session, base_entities, full=False)                  # 0.2500 -> D

        resp = grade_all(client)
        assert resp.status_code == 200
        assert resp.json()["grade_distribution"] == {"A": 1, "B": 2, "C": 2, "D": 1}

        expected = {
            op_a.id: "A",
            op_b.id: "B",
            op_round.id: "B",
            op_c.id: "C",
            op_c2.id: "C",
            op_d.id: "D",
        }
        db_session.expire_all()
        for op_id, grade in expected.items():
            assert db_session.get(OperationData, op_id).data_grade == grade

        # 同一策略再算一次，边界等级完全可重复
        resp2 = grade_all(client)
        assert resp2.json()["grade_distribution"] == {"A": 1, "B": 2, "C": 2, "D": 1}
        db_session.expire_all()
        for op_id, grade in expected.items():
            assert db_session.get(OperationData, op_id).data_grade == grade

    def test_missing_annotation_counts_as_zero(self, client, db_session, base_entities):
        operation = make_operation(db_session, base_entities)  # 完整度 1.0，无标注
        resp = grade_all(client)
        assert resp.status_code == 200

        info = client.get(f"/api/v1/quality/operation/{operation.id}").json()
        assert info["has_annotation"] is False
        assert info["quality_score"] == 0.5  # 0.5 * 1.0 + 0.5 * 0.0
        assert info["data_grade"] == "C"

    def test_nonexistent_operation_ids_rejected_before_any_update(self, client, db_session, base_entities):
        operation = make_operation(db_session, base_entities, annotation_quality=0.8)
        assert grade_all(client).status_code == 200
        before = stored_grades(db_session)
        assert before[operation.id][0] == "A"

        # 请求混入不存在的编号，并附带会把 A 改成 C 的策略：必须整体拒绝
        aggressive = {**DEFAULT_POLICY, "grade_a_threshold": 0.95}
        resp = client.post(
            GRADE_URL,
            params={"operation_ids": [operation.id, 999999]},
            json=aggressive,
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["invalid_operation_ids"] == [999999]
        assert stored_grades(db_session) == before

        # 全部编号存在时正常执行
        resp2 = client.post(GRADE_URL, params={"operation_ids": [operation.id]}, json=aggressive)
        assert resp2.status_code == 200
        assert resp2.json()["graded_count"] == 1

    def test_mid_batch_failure_rolls_back_everything(self, client, db_session, base_entities, monkeypatch):
        for quality in (0.8, 0.4, 0.0):
            make_operation(db_session, base_entities, annotation_quality=quality)
        assert grade_all(client).status_code == 200
        before = stored_grades(db_session)

        from app.services.scoring import compute_operation_quality as real_compute

        calls = {"count": 0}

        def failing_compute(operation, annotation, policy):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("模拟评分中途失败")
            return real_compute(operation, annotation, policy)

        monkeypatch.setattr("app.routers.analytics.compute_operation_quality", failing_compute)

        # 会把全部数据改判为 A 的策略，中途异常后任何一条都不应被改写
        aggressive = {**DEFAULT_POLICY, "grade_a_threshold": 0.3, "grade_b_threshold": 0.2, "grade_c_threshold": 0.1}
        resp = grade_all(client, aggressive)
        assert resp.status_code == 500
        assert calls["count"] == 2
        assert stored_grades(db_session) == before

        # 恢复后同一策略可以完整重算
        resp2 = grade_all(client, aggressive)
        assert resp2.status_code == 200
        assert resp2.json()["grade_distribution"] == {"A": 3, "B": 0, "C": 0, "D": 0}

    def test_grades_persist_across_restart(self, client, db_session, base_entities, db_path):
        op_a = make_operation(db_session, base_entities, annotation_quality=0.8)  # 0.9000 -> A
        op_d = make_operation(db_session, base_entities, full=False)              # 0.2500 -> D
        assert grade_all(client).status_code == 200

        # 模拟服务重启：用全新的引擎与会话读取同一个数据库文件
        engine2 = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        session2 = sessionmaker(bind=engine2)()
        try:
            restarted_a = session2.get(OperationData, op_a.id)
            restarted_d = session2.get(OperationData, op_d.id)
            assert restarted_a.data_grade == "A"
            assert restarted_a.quality_score == 0.9
            assert restarted_d.data_grade == "D"
        finally:
            session2.close()
            engine2.dispose()

        # 重启后用同一策略重算，等级保持不变
        resp = grade_all(client)
        assert resp.json()["grade_distribution"] == {"A": 1, "B": 0, "C": 0, "D": 1}
        assert stored_grades(db_session)[op_a.id][0] == "A"
