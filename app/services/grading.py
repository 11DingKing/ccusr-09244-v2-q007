"""质量分级批量重算：先验证全部输入、再一次性更新并提交。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from sqlalchemy.orm import Session

from app.models import Annotation, OperationData
from app.services.scoring import (
    GradePolicy,
    QualityScores,
    compute_operation_quality,
)


class OperationNotFoundError(LookupError):
    """指定的作业编号不存在，missing_ids 给出全部缺失编号。"""

    def __init__(self, missing_ids: Sequence[int]):
        self.missing_ids = sorted(set(missing_ids))
        super().__init__(f"作业数据不存在: {self.missing_ids}")


@dataclass(frozen=True)
class GradedItem:
    operation_id: int
    completeness_score: float
    annotation_quality_score: float
    quality_score: float
    data_grade: str


@dataclass(frozen=True)
class GradeBatchResult:
    total: int
    graded_count: int
    grade_distribution: dict[str, int]
    items: tuple[GradedItem, ...]


def grade_operations(
    db: Session,
    policy: GradePolicy,
    operation_ids: Optional[Sequence[int]] = None,
) -> GradeBatchResult:
    """按策略批量重算作业质量等级。

    执行顺序保证原子性：
    1. 策略已由 validate_grade_policy 校验（调用方负责）；
    2. 若指定作业编号，先确认全部存在，任何一个缺失都不更新任何数据；
    3. 读入全部目标作业及标注，先在内存中完成全部评分（任何异常都不会触碰原数据）；
    4. 全部评分成功后才统一写回，单次 commit；提交失败则回滚，原等级保持不变。
    """
    query = db.query(OperationData)
    requested_ids: Optional[list[int]] = None
    if operation_ids is not None:
        requested_ids = list(dict.fromkeys(operation_ids))  # 去重且保序
        if requested_ids:
            query = query.filter(OperationData.id.in_(requested_ids))

    operations = query.order_by(OperationData.id.asc()).all()

    if requested_ids is not None:
        found_ids = {op.id for op in operations}
        missing_ids = [op_id for op_id in requested_ids if op_id not in found_ids]
        if missing_ids:
            raise OperationNotFoundError(missing_ids)

    # 一次性取出相关标注，避免逐条查询；缺少标注的作业按零分处理
    annotations = (
        db.query(Annotation)
        .filter(Annotation.operation_data_id.in_([op.id for op in operations]))
        .all()
        if operations
        else []
    )
    annotation_map = {item.operation_data_id: item for item in annotations}
    thresholds = policy.threshold_map()

    # 第一阶段：纯计算，全部成功后才允许写库
    planned: list[tuple[OperationData, QualityScores]] = []
    for op in operations:
        scores = compute_operation_quality(
            operation=op,
            annotation=annotation_map.get(op.id),
            completeness_weight=policy.completeness_weight,
            annotation_weight=policy.annotation_weight,
            thresholds=thresholds,
        )
        planned.append((op, scores))

    # 第二阶段：统一更新、单次提交
    items: list[GradedItem] = []
    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    try:
        for op, scores in planned:
            op.completeness_score = scores.completeness_score
            op.quality_score = scores.quality_score
            op.data_grade = scores.data_grade

            grade_counts[scores.data_grade] += 1
            items.append(
                GradedItem(
                    operation_id=op.id,
                    completeness_score=scores.completeness_score,
                    annotation_quality_score=scores.annotation_quality_score,
                    quality_score=scores.quality_score,
                    data_grade=scores.data_grade,
                )
            )
        db.commit()
    except Exception:
        db.rollback()
        raise

    return GradeBatchResult(
        total=len(planned),
        graded_count=len(planned),
        grade_distribution=grade_counts,
        items=tuple(items),
    )
