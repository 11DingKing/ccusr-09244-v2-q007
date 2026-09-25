from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models import (
    OperationData, Annotation, Dataset,
    RobotModel, Scene, Skill
)
from app.schemas.dataset import (
    OverallStats, RobotModelStats, SceneStats,
    DataGradeStats, QualityGradeRequest, ReviewStatusStats
)
from app.services.scoring import (
    GradePolicyError,
    calculate_completeness_score,
    compute_operation_quality,
    validate_grading_policy
)
from app.services.aggregation import compute_group_stats

router = APIRouter()


@router.post("/quality/grade-operations", tags=["数据质量分级"])
def grade_operation_data(
    req: QualityGradeRequest,
    operation_ids: Optional[List[int]] = Query(None, description="指定作业ID列表，空则处理全部"),
    db: Session = Depends(get_db)
):
    # 1) 先完整校验分级策略：权重取值区间、权重和容差、阈值严格递减，
    #    所有无效字段一次性返回，策略不合法时不修改任何数据
    try:
        policy = validate_grading_policy(
            completeness_weight=req.completeness_weight,
            annotation_weight=req.annotation_weight,
            grade_a_threshold=req.grade_a_threshold,
            grade_b_threshold=req.grade_b_threshold,
            grade_c_threshold=req.grade_c_threshold
        )
    except GradePolicyError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": "质量分级策略无效", "invalid_fields": exc.errors}
        )

    # 2) 再校验目标作业：指定的编号必须全部存在，否则不执行任何分级
    if operation_ids:
        operations = db.query(OperationData).filter(OperationData.id.in_(operation_ids)).all()
        found_ids = {op.id for op in operations}
        missing_ids = [op_id for op_id in operation_ids if op_id not in found_ids]
        if missing_ids:
            raise HTTPException(
                status_code=400,
                detail={"message": "部分作业数据不存在，未执行任何分级", "invalid_operation_ids": missing_ids}
            )
    else:
        operations = db.query(OperationData).all()

    # 3) 全部输入合法后，先在内存中算完全部分数（纯计算，不写库），
    # 4) 再一次提交全部更新；中途任何异常都回滚，不会留下只更新了一半的数据
    try:
        annotations = {}
        if operations:
            annotation_rows = db.query(Annotation).filter(
                Annotation.operation_data_id.in_([op.id for op in operations])
            ).all()
            annotations = {row.operation_data_id: row for row in annotation_rows}

        computed = [
            (op, compute_operation_quality(op, annotations.get(op.id), policy))
            for op in operations
        ]

        for op, scores in computed:
            op.completeness_score = scores.completeness_score
            op.quality_score = scores.quality_score
            op.data_grade = scores.data_grade
        db.commit()
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={"message": "质量分级过程中发生异常，已回滚全部修改", "error": str(exc)}
        )

    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for _, scores in computed:
        grade_counts[scores.data_grade] += 1

    return {
        "message": f"已完成 {len(computed)} 条数据的质量分级",
        "graded_count": len(computed),
        "grade_distribution": grade_counts
    }


@router.get("/quality/operation/{operation_id}", tags=["数据质量分级"])
def get_operation_quality(operation_id: int, db: Session = Depends(get_db)):
    op = db.query(OperationData).filter(OperationData.id == operation_id).first()
    if not op:
        raise HTTPException(status_code=404, detail="作业数据不存在")

    completeness = op.completeness_score if op.completeness_score is not None else calculate_completeness_score(op)
    annotation = db.query(Annotation).filter(Annotation.operation_data_id == operation_id).first()

    annotation_info = None
    if annotation:
        annotation_info = {
            "is_success": annotation.is_success,
            "failure_category": annotation.failure_category,
            "review_status": annotation.review_status,
            "quality_score": annotation.annotation_quality_score
        }

    return {
        "operation_id": operation_id,
        "completeness_score": completeness,
        "quality_score": op.quality_score,
        "data_grade": op.data_grade,
        "has_annotation": annotation is not None,
        "annotation": annotation_info
    }


@router.get("/stats/overview", response_model=OverallStats, tags=["统计分析"])
def get_overall_stats(db: Session = Depends(get_db)):
    total_operation_data = db.query(func.count(OperationData.id)).scalar() or 0
    total_annotated = db.query(func.count(Annotation.id)).scalar() or 0

    success_count = db.query(func.count(Annotation.id)).filter(Annotation.is_success == True).scalar() or 0
    failure_count = db.query(func.count(Annotation.id)).filter(Annotation.is_success == False).scalar() or 0

    total_datasets = db.query(func.count(Dataset.id)).scalar() or 0
    total_published_datasets = db.query(func.count(Dataset.id)).filter(Dataset.is_published == True).scalar() or 0
    total_reuse_count = db.query(func.sum(Dataset.reuse_count)).scalar() or 0

    robot_model_count = db.query(func.count(RobotModel.id)).scalar() or 0
    scene_count = db.query(func.count(Scene.id)).scalar() or 0
    skill_count = db.query(func.count(Skill.id)).scalar() or 0

    return OverallStats(
        total_operation_data=total_operation_data,
        total_annotated=total_annotated,
        overall_annotation_rate=round(total_annotated / total_operation_data, 4) if total_operation_data > 0 else 0.0,
        total_datasets=total_datasets,
        total_published_datasets=total_published_datasets,
        total_reuse_count=total_reuse_count,
        total_success_count=success_count,
        total_failure_count=failure_count,
        robot_model_count=robot_model_count,
        scene_count=scene_count,
        skill_count=skill_count
    )


@router.get("/stats/by-robot-model", response_model=List[RobotModelStats], tags=["统计分析"])
def get_stats_by_robot_model(
    robot_model_id: Optional[int] = Query(None, description="指定机型ID"),
    db: Session = Depends(get_db)
):
    query = db.query(RobotModel)
    if robot_model_id:
        query = query.filter(RobotModel.id == robot_model_id)
    robot_models = query.all()

    results = []
    for rm in robot_models:
        ops = db.query(OperationData).filter(OperationData.robot_model_id == rm.id).all()
        datasets = db.query(Dataset).filter(Dataset.robot_model_id == rm.id).all()

        stats = compute_group_stats(db, ops, datasets, include_grade_distribution=True)

        results.append(RobotModelStats(
            robot_model_id=rm.id,
            robot_model_name=rm.name,
            manufacturer=rm.manufacturer,
            total_data_count=stats.total_data_count,
            annotated_count=stats.annotation.annotated_count,
            annotation_complete_rate=stats.annotation_complete_rate,
            success_count=stats.annotation.success_count,
            failure_count=stats.annotation.failure_count,
            dataset_count=stats.datasets.dataset_count,
            total_reuse_count=stats.datasets.total_reuse_count,
            reuse_rate=stats.datasets.reuse_rate,
            grade_distribution=stats.grade_distribution
        ))

    return results


@router.get("/stats/by-scene", response_model=List[SceneStats], tags=["统计分析"])
def get_stats_by_scene(
    scene_id: Optional[int] = Query(None, description="指定场景ID"),
    db: Session = Depends(get_db)
):
    query = db.query(Scene)
    if scene_id:
        query = query.filter(Scene.id == scene_id)
    scenes = query.all()

    results = []
    for s in scenes:
        ops = db.query(OperationData).filter(OperationData.scene_id == s.id).all()
        datasets = db.query(Dataset).filter(Dataset.scene_id == s.id).all()

        stats = compute_group_stats(db, ops, datasets, include_grade_distribution=False)

        results.append(SceneStats(
            scene_id=s.id,
            scene_name=s.name,
            scene_category=s.category,
            total_data_count=stats.total_data_count,
            annotated_count=stats.annotation.annotated_count,
            annotation_complete_rate=stats.annotation_complete_rate,
            success_count=stats.annotation.success_count,
            failure_count=stats.annotation.failure_count,
            dataset_count=stats.datasets.dataset_count,
            total_reuse_count=stats.datasets.total_reuse_count,
            reuse_rate=stats.datasets.reuse_rate
        ))

    return results


@router.get("/stats/grade-distribution", tags=["统计分析"])
def get_grade_distribution(
    robot_model_id: Optional[int] = Query(None),
    scene_id: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(OperationData)
    if robot_model_id:
        query = query.filter(OperationData.robot_model_id == robot_model_id)
    if scene_id:
        query = query.filter(OperationData.scene_id == scene_id)

    total = query.count()
    grade_counts = {}
    for grade in ["A", "B", "C", "D", None]:
        if grade is None:
            count = query.filter(OperationData.data_grade.is_(None)).count()
            label = "未分级"
        else:
            count = query.filter(OperationData.data_grade == grade).count()
            label = grade
        grade_counts[label] = {
            "count": count,
            "percentage": round(count / total, 4) if total > 0 else 0.0
        }

    return {
        "total": total,
        "distribution": grade_counts
    }


@router.get("/stats/failure-analysis", tags=["统计分析"])
def get_failure_analysis(
    robot_model_id: Optional[int] = Query(None),
    scene_id: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(Annotation).filter(Annotation.is_success == False)
    if robot_model_id or scene_id:
        query = query.join(OperationData, Annotation.operation_data_id == OperationData.id)
        if robot_model_id:
            query = query.filter(OperationData.robot_model_id == robot_model_id)
        if scene_id:
            query = query.filter(OperationData.scene_id == scene_id)

    failures = query.all()
    total_failures = len(failures)

    category_counts = {}
    subcategory_counts = {}
    for f in failures:
        cat = f.failure_category or "未分类"
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if f.failure_subcategory:
            if cat not in subcategory_counts:
                subcategory_counts[cat] = {}
            subcategory_counts[cat][f.failure_subcategory] = subcategory_counts[cat].get(f.failure_subcategory, 0) + 1

    category_distribution = []
    for cat, count in category_counts.items():
        category_distribution.append({
            "category": cat,
            "count": count,
            "percentage": round(count / total_failures, 4) if total_failures > 0 else 0.0,
            "subcategories": subcategory_counts.get(cat, {})
        })

    return {
        "total_failures": total_failures,
        "category_distribution": sorted(category_distribution, key=lambda x: x["count"], reverse=True)
    }


@router.get("/stats/review-status", response_model=ReviewStatusStats, tags=["统计分析"])
def get_review_status_stats(db: Session = Depends(get_db)):
    draft = db.query(func.count(Dataset.id)).filter(Dataset.review_status == "draft").scalar() or 0
    pending_review = db.query(func.count(Dataset.id)).filter(Dataset.review_status == "pending_review").scalar() or 0
    approved = db.query(func.count(Dataset.id)).filter(Dataset.review_status == "approved").scalar() or 0
    rejected = db.query(func.count(Dataset.id)).filter(Dataset.review_status == "rejected").scalar() or 0
    published = db.query(func.count(Dataset.id)).filter(Dataset.is_published == True).scalar() or 0

    return ReviewStatusStats(
        draft=draft,
        pending_review=pending_review,
        approved=approved,
        rejected=rejected,
        published=published
    )
