"""message created_at -> clock_timestamp

Revision ID: b7f2c0a1d9e3
Revises: e21b99ab6e0a
Create Date: 2026-10-08 16:00:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b7f2c0a1d9e3'
down_revision: Union[str, Sequence[str], None] = 'e21b99ab6e0a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1) Новые сообщения: created_at = clock_timestamp() — время выполнения
    #    INSERT, а не начала транзакции (now() = transaction_timestamp()).
    #    Иначе user и assistant, записанные в одной транзакции (run_turn),
    #    получают одинаковый created_at, и сортировка падает на случайный
    #    UUID id — в UI ответ ассистента оказывается выше вопроса.
    op.alter_column(
        'messages', 'created_at',
        server_default=sa.text('clock_timestamp()'))

    # 2) Починить уже записанные диалоги: внутри группы с одинаковым
    #    (conversation_id, created_at) развести строки на 1 мкс. Внутри
    #    группы всегда сначала user, потом assistant (ORDER BY это
    #    гарантирует), поэтому user остаётся на месте, assistant сдвигается
    #    на +1 мкс — в UI такой сдвиг незаметен.
    op.execute("""
        UPDATE messages AS m
        SET created_at = ranked.new_ts
        FROM (
            SELECT
                id,
                created_at + (row_number() OVER (
                    PARTITION BY conversation_id, created_at
                    ORDER BY CASE role WHEN 'user' THEN 0 ELSE 1 END, id
                ) - 1) * interval '1 microsecond' AS new_ts
            FROM messages
            WHERE (conversation_id, created_at) IN (
                SELECT conversation_id, created_at
                FROM messages
                GROUP BY conversation_id, created_at
                HAVING count(*) > 1
            )
        ) ranked
        WHERE m.id = ranked.id
          AND m.created_at <> ranked.new_ts
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        'messages', 'created_at',
        server_default=sa.text('now()'))
    # Бэкфилл данных однонаправленный: вернуть прежние (склеенные)
    # created_at на downgrade нельзя — это нормально для data-миграции.
