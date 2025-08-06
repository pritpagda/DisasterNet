"""Create users, predictions, feedback tables

Revision ID: 61d518c8bd76
Revises:
Create Date: 2025-07-15 23:41:37.393228

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '61d518c8bd76'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False, unique=True),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("text", sa.Text, nullable=True),
        sa.Column("image_path", sa.String(length=255), nullable=True),
        sa.Column("informative", sa.String(length=50), nullable=True),
        sa.Column("humanitarian", sa.String(length=100), nullable=True),
        sa.Column("damage", sa.String(length=50), nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("prediction_id", sa.Integer, sa.ForeignKey("predictions.id"), nullable=False),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("correct", sa.String(length=10), nullable=True),  # 'yes' or 'no'
        sa.Column("comments", sa.Text, nullable=True),
        sa.Column("submitted_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_table("predictions")
    op.drop_table("users")
