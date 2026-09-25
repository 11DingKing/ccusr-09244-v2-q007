from typing import Optional, Dict, Mapping
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import math

from app.models import OperationData, Annotation


# ---- 质量策略的取值区间与容差（在 API 层校验之外，领域层同样强制） ----
WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.0
WEIGHT_SUM_TARGET = 1.0
# 完整度权重与标注权重之和允许的浮点舍入误差
WEIGHT_SUM_TOLERANCE = 1e-6
# 分级阈值的取值区间
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0
# 相邻阈值之间必须保留的最小间距：小于该值视为“未严格有序”
THRESHOLD_MIN_GAP = 1e-9
# 质量分数保留的小数位，边界比较也按该精度进行
SCORE_PLACES = 4
_SCORE_QUANTUM = Decimal("0.0001")


@dataclass
class QualityScores:
    completeness_score: float
    annotation_quality_score: float
    quality_score: float
    data_grade: str


@dataclass(frozen=True)
class GradePolicy:
    """经验证的质量分级策略，保证权重合法、阈值严格有序。"""

    completeness_weight: float
    annotation_weight: float
    grade_a_threshold: float
    grade_b_threshold: float
    grade_c_threshold: float

    def threshold_map(self) -> Dict[str, float]:
        return {
            "grade_a": self.grade_a_threshold,
            "grade_b": self.grade_b_threshold,
            "grade_c": self.grade_c_threshold,
        }


class GradePolicyError(ValueError):
    """质量分级策略不合法，field_errors 按字段名给出具体原因。"""

    def __init__(self, field_errors: Mapping[str, str]):
        self.field_errors: Dict[str, list[str]] = {
            key: [value] for key, value in field_errors.items()
        }
        message = "; ".join(f"{key}: {value}" for key, value in field_errors.items())
        super().__init__(message)


_FIELD_LABELS = {
    "completeness_weight": "完整度权重",
    "annotation_weight": "标注质量权重",
    "grade_a_threshold": "A级阈值",
    "grade_b_threshold": "B级阈值",
    "grade_c_threshold": "C级阈值",
}


def validate_grade_policy(
    completeness_weight: float,
    annotation_weight: float,
    grade_a_threshold: float,
    grade_b_threshold: float,
    grade_c_threshold: float,
) -> GradePolicy:
    """校验分级策略，收集全部字段错误后一次性抛出。

    规则：
    - 权重与阈值都必须是有限数字；
    - 权重与阈值都必须位于闭区间 [0, 1]；
    - 完整度权重与标注权重之和必须等于 1（允许 WEIGHT_SUM_TOLERANCE 的浮点误差）；
    - grade_a > grade_b > grade_c，相邻阈值严格有序（间距大于 THRESHOLD_MIN_GAP）。
    """
    raw = {
        "completeness_weight": completeness_weight,
        "annotation_weight": annotation_weight,
        "grade_a_threshold": grade_a_threshold,
        "grade_b_threshold": grade_b_threshold,
        "grade_c_threshold": grade_c_threshold,
    }
    numeric: Dict[str, float] = {}
    field_errors: Dict[str, str] = {}

    for key, value in raw.items():
        label = _FIELD_LABELS[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            field_errors[key] = f"{label}必须是数字"
            continue
        converted = float(value)
        if not math.isfinite(converted):
            field_errors[key] = f"{label}必须是有限数字，不能为无穷或NaN"
            continue
        numeric[key] = converted

    if not field_errors:
        for key in ("completeness_weight", "annotation_weight"):
            value = numeric[key]
            if not WEIGHT_MIN <= value <= WEIGHT_MAX:
                field_errors[key] = (
                    f"{_FIELD_LABELS[key]}必须位于区间[{WEIGHT_MIN:g}, {WEIGHT_MAX:g}]内，"
                    f"当前为 {value:g}"
                )

        for key in ("grade_a_threshold", "grade_b_threshold", "grade_c_threshold"):
            value = numeric[key]
            if not THRESHOLD_MIN <= value <= THRESHOLD_MAX:
                field_errors[key] = (
                    f"{_FIELD_LABELS[key]}必须位于区间[{THRESHOLD_MIN:g}, {THRESHOLD_MAX:g}]内，"
                    f"当前为 {value:g}"
                )

        weight_total = numeric["completeness_weight"] + numeric["annotation_weight"]
        if abs(weight_total - WEIGHT_SUM_TARGET) > WEIGHT_SUM_TOLERANCE:
            field_errors["weights"] = (
                f"完整度权重与标注质量权重之和必须等于 {WEIGHT_SUM_TARGET:g}"
                f"（容差 ±{WEIGHT_SUM_TOLERANCE:g}），当前和为 {weight_total!r}"
            )

        # 阈值必须严格有序：A > B > C
        ordered_pairs = (
            ("grade_a_threshold", "grade_b_threshold"),
            ("grade_b_threshold", "grade_c_threshold"),
        )
        for higher_key, lower_key in ordered_pairs:
            gap = numeric[higher_key] - numeric[lower_key]
            if gap <= THRESHOLD_MIN_GAP:
                message = (
                    f"{_FIELD_LABELS[higher_key]}必须严格大于{_FIELD_LABELS[lower_key]}"
                )
                field_errors.setdefault(higher_key, message)
                field_errors.setdefault(lower_key, message)

    if field_errors:
        raise GradePolicyError(field_errors)

    return GradePolicy(
        completeness_weight=numeric["completeness_weight"],
        annotation_weight=numeric["annotation_weight"],
        grade_a_threshold=numeric["grade_a_threshold"],
        grade_b_threshold=numeric["grade_b_threshold"],
        grade_c_threshold=numeric["grade_c_threshold"],
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
    return min(1.0, max(0.0, base))


def _score_decimal(value: float) -> Decimal:
    """把分数转成十进制表示，消除二进制浮点的边界舍入误差。"""
    return Decimal(str(value)).quantize(_SCORE_QUANTUM, rounding=ROUND_HALF_UP)


def determine_grade(quality_score: float, thresholds: Mapping[str, float]) -> str:
    """按阈值判定等级，分数落在阈值边界上时归入更高等级（边界可重复）。"""
    score = Decimal(str(quality_score))
    grade_a = Decimal(str(thresholds.get("grade_a", 0.9)))
    grade_b = Decimal(str(thresholds.get("grade_b", 0.7)))
    grade_c = Decimal(str(thresholds.get("grade_c", 0.5)))

    if score >= grade_a:
        return "A"
    elif score >= grade_b:
        return "B"
    elif score >= grade_c:
        return "C"
    else:
        return "D"


def compute_operation_quality(
    operation: OperationData,
    annotation: Optional[Annotation],
    completeness_weight: float,
    annotation_weight: float,
    thresholds: Mapping[str, float]
) -> QualityScores:
    completeness = calculate_completeness_score(operation)
    annotation_quality = calculate_annotation_quality_score(annotation)

    # 使用 Decimal 计算加权和，避免 0.1+0.2 这类二进制浮点误差使边界分数错级
    quality_decimal = (
        Decimal(str(completeness)) * Decimal(str(completeness_weight))
        + Decimal(str(annotation_quality)) * Decimal(str(annotation_weight))
    ).quantize(_SCORE_QUANTUM, rounding=ROUND_HALF_UP)
    quality_score = float(quality_decimal)
    grade = determine_grade(quality_score, thresholds)

    return QualityScores(
        completeness_score=round(completeness, 4),
        annotation_quality_score=round(annotation_quality, 4),
        quality_score=quality_score,
        data_grade=grade
    )
