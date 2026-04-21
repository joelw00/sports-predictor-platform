"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-21

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def _ts() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "sports",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("code", sa.String(32), unique=True, nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        *_ts(),
    )

    op.create_table(
        "competitions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("sport_id", sa.Integer, sa.ForeignKey("sports.id"), nullable=False),
        sa.Column("code", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("country", sa.String(64)),
        sa.Column("tier", sa.Integer),
        *_ts(),
        sa.UniqueConstraint("sport_id", "code", name="uq_competition_sport_code"),
    )

    op.create_table(
        "teams",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("competition_id", sa.Integer, sa.ForeignKey("competitions.id")),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("short_name", sa.String(32)),
        sa.Column("country", sa.String(64)),
        *_ts(),
    )

    op.create_table(
        "players",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id")),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("position", sa.String(16)),
        sa.Column("rating", sa.Float),
        *_ts(),
    )

    op.create_table(
        "matches",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("sport_id", sa.Integer, sa.ForeignKey("sports.id"), nullable=False),
        sa.Column("competition_id", sa.Integer, sa.ForeignKey("competitions.id")),
        sa.Column("home_team_id", sa.Integer, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("away_team_id", sa.Integer, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("kickoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="scheduled"),
        sa.Column("home_score", sa.Integer),
        sa.Column("away_score", sa.Integer),
        sa.Column("season", sa.String(16)),
        sa.Column("stage", sa.String(64)),
        sa.Column("venue", sa.String(128)),
        *_ts(),
    )
    op.create_index("ix_matches_kickoff", "matches", ["kickoff"])

    op.create_table(
        "match_stats",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("match_id", sa.Integer, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("shots", sa.Integer),
        sa.Column("shots_on_target", sa.Integer),
        sa.Column("corners", sa.Integer),
        sa.Column("yellow_cards", sa.Integer),
        sa.Column("red_cards", sa.Integer),
        sa.Column("fouls", sa.Integer),
        sa.Column("possession", sa.Float),
        sa.Column("xg", sa.Float),
        sa.Column("xga", sa.Float),
        sa.Column("passes", sa.Integer),
        sa.Column("pass_accuracy", sa.Float),
        *_ts(),
    )

    op.create_table(
        "odds_snapshots",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("match_id", sa.Integer, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("bookmaker", sa.String(64), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("selection", sa.String(64), nullable=False),
        sa.Column("line", sa.Float),
        sa.Column("price", sa.Float, nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_closing", sa.Boolean, server_default=sa.false(), nullable=False),
        sa.Column("is_live", sa.Boolean, server_default=sa.false(), nullable=False),
        *_ts(),
    )
    op.create_index("ix_odds_match", "odds_snapshots", ["match_id", "market", "selection"])

    op.create_table(
        "model_registry",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("sport_code", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("family", sa.String(32), nullable=False),
        sa.Column("version", sa.String(32), nullable=False),
        sa.Column("artifact_path", sa.String(512), nullable=False),
        sa.Column("metrics", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("is_active", sa.Boolean, server_default=sa.true(), nullable=False),
        *_ts(),
        sa.UniqueConstraint("sport_code", "market", "version", name="uq_model_unique_version"),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("match_id", sa.Integer, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("selection", sa.String(64), nullable=False),
        sa.Column("line", sa.Float),
        sa.Column("probability", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("drivers", sa.JSON, nullable=False, server_default="{}"),
        *_ts(),
        sa.UniqueConstraint("match_id", "market", "selection", "line", name="uq_pred_match_market_selection"),
    )

    op.create_table(
        "value_bets",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("match_id", sa.Integer, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("selection", sa.String(64), nullable=False),
        sa.Column("line", sa.Float),
        sa.Column("bookmaker", sa.String(64), nullable=False),
        sa.Column("price", sa.Float, nullable=False),
        sa.Column("p_model", sa.Float, nullable=False),
        sa.Column("p_fair", sa.Float, nullable=False),
        sa.Column("edge", sa.Float, nullable=False),
        sa.Column("expected_value", sa.Float, nullable=False),
        sa.Column("kelly_fraction", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("rationale", sa.JSON, nullable=False, server_default="{}"),
        *_ts(),
    )
    op.create_index("ix_value_bets_edge", "value_bets", ["edge"])

    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("label", sa.String(128), nullable=False),
        sa.Column("sport_code", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("start_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("strategy", sa.String(32), nullable=False, server_default="flat"),
        sa.Column("min_edge", sa.Float, nullable=False, server_default="0.03"),
        sa.Column("stake", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("n_bets", sa.Integer, nullable=False, server_default="0"),
        sa.Column("n_wins", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_staked", sa.Float, nullable=False, server_default="0"),
        sa.Column("total_return", sa.Float, nullable=False, server_default="0"),
        sa.Column("roi", sa.Float, nullable=False, server_default="0"),
        sa.Column("yield_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("max_drawdown", sa.Float, nullable=False, server_default="0"),
        sa.Column("profit_factor", sa.Float, nullable=False, server_default="0"),
        sa.Column("equity_curve", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("breakdown", sa.JSON, nullable=False, server_default="{}"),
        *_ts(),
    )


def downgrade() -> None:
    for table in [
        "backtest_runs",
        "value_bets",
        "predictions",
        "model_registry",
        "odds_snapshots",
        "match_stats",
        "matches",
        "players",
        "teams",
        "competitions",
        "sports",
    ]:
        op.drop_table(table)
