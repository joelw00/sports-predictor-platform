"""raw ingestion payloads and runs

Revision ID: 0002_raw_ingestion
Revises: 0001_initial
Create Date: 2026-04-22

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_raw_ingestion"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def _ts() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "ingestion_payloads",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("endpoint", sa.String(256), nullable=False),
        sa.Column("params", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("status_code", sa.Integer),
        sa.Column("payload", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ok", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("error", sa.String(512)),
        *_ts(),
    )
    op.create_index(
        "ix_ingestion_payloads_source_endpoint",
        "ingestion_payloads",
        ["source", "endpoint", "fetched_at"],
    )

    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("trigger", sa.String(32), nullable=False, server_default="manual"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("matches_upserted", sa.Integer, nullable=False, server_default="0"),
        sa.Column("stats_upserted", sa.Integer, nullable=False, server_default="0"),
        sa.Column("odds_upserted", sa.Integer, nullable=False, server_default="0"),
        sa.Column("ok", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("error", sa.String(1024)),
        sa.Column("meta", sa.JSON, nullable=False, server_default="{}"),
        *_ts(),
    )
    op.create_index(
        "ix_ingestion_runs_source_started",
        "ingestion_runs",
        ["source", "started_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_ingestion_runs_source_started", table_name="ingestion_runs")
    op.drop_table("ingestion_runs")
    op.drop_index("ix_ingestion_payloads_source_endpoint", table_name="ingestion_payloads")
    op.drop_table("ingestion_payloads")
