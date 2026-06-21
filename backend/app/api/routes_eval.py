"""
Evaluation endpoints — Phase 6.

    POST   /eval/run
    GET    /eval/runs
    GET    /eval/runs/{run_id}

`POST /eval/run` runs every item in the dataset through the same
retrieval + generation pipeline used by /rag/ask, then persists the
EvaluationRun + per-question EvaluationResult rows. Summary stats are
computed on the fly (we don't store them) so they're always accurate
even if the metric formulas change later.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import EvaluationResult, EvaluationRun
from app.db.session import get_db
from app.schemas.eval import (
    EvalResultRead,
    EvalRunList,
    EvalRunRead,
    EvalRunRequest,
    EvalRunSummaryRow,
    EvalSummary,
)
from app.services.evaluation import run_evaluation, summarize_run

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# POST /eval/run
# =============================================================================
@router.post(
    "/run",
    response_model=EvalRunRead,
    status_code=status.HTTP_201_CREATED,
    summary="Run the evaluation harness and persist results.",
)
def run_eval(
    payload: EvalRunRequest,
    db: Session = Depends(get_db),
) -> EvalRunRead:
    try:
        run = run_evaluation(
            db,
            name=payload.name,
            description=payload.description,
            dataset_path=payload.dataset_path,
            top_k=payload.top_k,
            mode=payload.mode,
            alpha=payload.alpha,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    db.commit()
    db.refresh(run)
    # Load results so the response body includes them.
    db.refresh(run, attribute_names=["results"])

    return _serialize_run(run)


# =============================================================================
# GET /eval/runs
# =============================================================================
@router.get(
    "/runs",
    response_model=EvalRunList,
    summary="List evaluation runs (most recent first).",
)
def list_runs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> EvalRunList:
    # One aggregated query to avoid the N+1 you'd get from
    # `len(run.results)` in a loop.
    counts_subq = (
        select(
            EvaluationResult.evaluation_run_id.label("run_id"),
            func.count(EvaluationResult.id).label("num_questions"),
        )
        .group_by(EvaluationResult.evaluation_run_id)
        .subquery()
    )

    stmt = (
        select(EvaluationRun, func.coalesce(counts_subq.c.num_questions, 0))
        .outerjoin(counts_subq, EvaluationRun.id == counts_subq.c.run_id)
        .order_by(EvaluationRun.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = db.execute(stmt).all()

    total = db.execute(select(func.count(EvaluationRun.id))).scalar_one()

    return EvalRunList(
        items=[
            EvalRunSummaryRow(
                id=run.id,
                name=run.name,
                description=run.description,
                created_at=run.created_at,
                num_questions=int(n),
            )
            for run, n in rows
        ],
        total=int(total),
    )


# =============================================================================
# GET /eval/runs/{run_id}
# =============================================================================
@router.get(
    "/runs/{run_id}",
    response_model=EvalRunRead,
    summary="Get one evaluation run with its per-question results and a fresh summary.",
)
def get_run(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> EvalRunRead:
    stmt = (
        select(EvaluationRun)
        .where(EvaluationRun.id == run_id)
        .options(selectinload(EvaluationRun.results))
    )
    run = db.execute(stmt).scalar_one_or_none()
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "evaluation run not found")
    return _serialize_run(run)


# =============================================================================
# Serialization helper
# =============================================================================
def _serialize_run(run: EvaluationRun) -> EvalRunRead:
    summary_dict = summarize_run(run)
    return EvalRunRead(
        id=run.id,
        name=run.name,
        description=run.description,
        created_at=run.created_at,
        summary=EvalSummary(**summary_dict),
        results=[
            EvalResultRead(
                id=r.id,
                question=r.question,
                expected_answer=r.expected_answer,
                generated_answer=r.generated_answer,
                context_precision=r.context_precision,
                context_recall=r.context_recall,
                faithfulness=r.faithfulness,
                answer_relevancy=r.answer_relevancy,
                hallucination_flag=r.hallucination_flag,
                latency_ms=r.latency_ms,
                notes=r.notes,
                details=r.details or {},
            )
            for r in sorted(run.results, key=lambda x: x.question)
        ],
    )
