from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, Mapping, Optional

from app.models import OperationData, Annotation


# 质量分数统一量化为四位小数，保证同一分数在任何一次重算中落入同一等级
SCORE_QUANTUM = Decimal("0.0001")
# 完整度权重与标注权重之和允许的浮点容差
WEIGHT_SUM_TOLERANCE = Decimal("0.000001")

DEFAULT_THRESHOLDS: Dict[str, Decimal] = {
    "grade_a": Decimal("0.9"),
    "grade_b": Decimal("0.7"),
    "grade_c": Decimal("0.5"),
}


class GradePolicyError(ValueError):
    """分级策略校验失败。errors 以字段名为键，说明每个无效字段的原因。"""

    def __init__(self, errors: Mapping[str, str]):
        self.errors = dict(errors)
        message = "; ".join(f"{field}: {reason}" for field, reason in self.errors.items())
        super().__init__(message or "分级策略无效")


@dataclass(frozen=True)
class GradingPolicy:
    """校验通过、可直接用于分级的策略，数值全部为精确的十进制。"""

    completeness_weight: Decimal
    annotation_weight: Decimal
    grade_a_threshold: Decimal
    grade_b_threshold: Decimal
    grade_c_threshold: Decimal

    def thresholds(self) -> Dict[str, Decimal]:
        return {
            "grade_a": self.grade_a_threshold,
            "grade_b": self.grade_b_threshold,
            "grade_c": self.grade_c_threshold,
        }


@dataclass
class QualityScores:
    completeness_score: float
    annotation_quality_score: float
    quality_score: float
    data_grade: str


def _as_decimal(value: Any, field: str, errors: Dict[str, str]) -> Optional[Decimal]:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        errors[field] = "必须是有效数字"
        return None
    if not result.is_finite():
        errors[field] = "必须是有限数字"
        return None
    return result


def validate_grading_policy(
    completeness_weight: Any,
    annotation_weight: Any,
    grade_a_threshold: Any,
    grade_b_threshold: Any,
    grade_c_threshold: Any,
) -> GradingPolicy:
    """校验分级策略。

    权重必须位于 [0, 1] 区间，且两者之和等于 1（容差 WEIGHT_SUM_TOLERANCE）；
    三个等级阈值必须位于 [0, 1] 区间且严格递减（A > B > C），
    否则同一分数会落入错误等级。所有字段的错误一次性收集，
    通过 GradePolicyError.errors 按字段名返回。
    """
    errors: Dict[str, str] = {}
    cw = _as_decimal(completeness_weight, "completeness_weight", errors)
    aw = _as_decimal(annotation_weight, "annotation_weight", errors)
    ta = _as_decimal(grade_a_threshold, "grade_a_threshold", errors)
    tb = _as_decimal(grade_b_threshold, "grade_b_threshold", errors)
    tc = _as_decimal(grade_c_threshold, "grade_c_threshold", errors)

    zero = Decimal("0")
    one = Decimal("1")

    for field, value in (("completeness_weight", cw), ("annotation_weight", aw)):
        if value is not None and not zero <= value <= one:
            errors[field] = "权重必须位于 [0, 1] 区间"

    if (
        cw is not None
        and aw is not None
        and "completeness_weight" not in errors
        and "annotation_weight" not in errors
    ):
        weight_sum = cw + aw
        if abs(weight_sum - one) > WEIGHT_SUM_TOLERANCE:
            errors["weight_sum"] = (
                f"完整度权重与标注权重之和必须等于 1"
                f"（容差 {WEIGHT_SUM_TOLERANCE}），当前为 {weight_sum}"
            )

    for field, value in (
        ("grade_a_threshold", ta),
        ("grade_b_threshold", tb),
        ("grade_c_threshold", tc),
    ):
        if value is not None and not zero <= value <= one:
            errors[field] = "阈值必须位于 [0, 1] 区间"

    # 阈值必须严格递减，否则同一分数会落入错误等级
    if ta is not None and tb is not None and not ta > tb:
        errors.setdefault("grade_a_threshold", "A 级阈值必须严格大于 B 级阈值")
        errors.setdefault("grade_b_threshold", "B 级阈值必须严格小于 A 级阈值")
    if tb is not None and tc is not None and not tb > tc:
        errors.setdefault("grade_b_threshold", "B 级阈值必须严格大于 C 级阈值")
        errors.setdefault("grade_c_threshold", "C 级阈值必须严格小于 B 级阈值")

    if errors:
        raise GradePolicyError(errors)

    return GradingPolicy(
        completeness_weight=cw,
        annotation_weight=aw,
        grade_a_threshold=ta,
        grade_b_threshold=tb,
        grade_c_threshold=tc,
    )


def calculate_completeness_score(operation: OperationData) -> float:
    total_fields = 6
    score = 0.0

    if operation.motion_trajectory:
        traj = operation.motion_trajectory
        if isinstance(traj, dict):
            if traj.get("waypoints") and len(traj["waypoints"]) > 0:
                score += 0.5
            if traj.get("joint_angles"):
                score += 0.5
        else:
            score += 1.0

    if operation.perception_records:
        percep = operation.perception_records
        if isinstance(percep, dict):
            keys_count = len(percep.keys())
            score += min(1.0, keys_count / 5)
        else:
            score += 1.0

    if operation.grasp_result:
        score += 1.0
    if operation.environment_conditions:
        score += 1.0
    if operation.hardware_status:
        score += 1.0
    if operation.duration_ms:
        score += 1.0

    return round(score / total_fields, 4)


def calculate_annotation_quality_score(annotation: Optional[Annotation]) -> float:
    if not annotation:
        return 0.0

    base = 0.6
    if annotation.review_status == "approved":
        base += 0.2
    if annotation.annotation_quality_score is not None:
        base = annotation.annotation_quality_score
    elif annotation.failure_category and annotation.failure_description:
        base += 0.2
    return min(1.0, base)


def _to_decimal_score(value: Any) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("分数必须是有限数字")
    return result


def determine_grade(quality_score: Any, thresholds: Mapping[str, Any]) -> str:
    """按阈值判定等级，分数恰好等于阈值时归入较高等级。

    比较基于十进制精确值，避免浮点舍入让同一边界分数落入不同等级。
    """
    score = _to_decimal_score(quality_score)
    grade_a = _to_decimal_score(thresholds.get("grade_a", DEFAULT_THRESHOLDS["grade_a"]))
    grade_b = _to_decimal_score(thresholds.get("grade_b", DEFAULT_THRESHOLDS["grade_b"]))
    grade_c = _to_decimal_score(thresholds.get("grade_c", DEFAULT_THRESHOLDS["grade_c"]))
    if score >= grade_a:
        return "A"
    if score >= grade_b:
        return "B"
    if score >= grade_c:
        return "C"
    return "D"


def compute_operation_quality(
    operation: OperationData,
    annotation: Optional[Annotation],
    policy: GradingPolicy,
) -> QualityScores:
    """按校验过的策略计算质量分数与等级。

    全程使用十进制运算并按 SCORE_QUANTUM 做 ROUND_HALF_UP 量化，
    相同输入在任何一次重算中都得到相同的分数与边界等级。
    """
    completeness = _to_decimal_score(calculate_completeness_score(operation)).quantize(
        SCORE_QUANTUM, rounding=ROUND_HALF_UP
    )
    annotation_quality = _to_decimal_score(calculate_annotation_quality_score(annotation)).quantize(
        SCORE_QUANTUM, rounding=ROUND_HALF_UP
    )

    quality_score = (
        completeness * policy.completeness_weight + annotation_quality * policy.annotation_weight
    ).quantize(SCORE_QUANTUM, rounding=ROUND_HALF_UP)
    grade = determine_grade(quality_score, policy.thresholds())

    return QualityScores(
        completeness_score=float(completeness),
        annotation_quality_score=float(annotation_quality),
        quality_score=float(quality_score),
        data_grade=grade
    )
