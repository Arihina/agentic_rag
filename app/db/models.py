from __future__ import annotations

"""Модели персистентного слоя agentic_rag."""

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PgUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, TypeDecorator, CHAR


class Base(DeclarativeBase):
    ...


class UUIDType(TypeDecorator):
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PgUUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(value)
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value if isinstance(value, uuid.UUID) else uuid.UUID(value)


class JSONBType(TypeDecorator):
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUIDType(), nullable=False)
    rag_id: Mapped[uuid.UUID] = mapped_column(UUIDType(), nullable=False)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="ok")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation: Mapped["Conversation"] = relationship(
        back_populates="messages")
    sources: Mapped[list["MessageSource"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="MessageSource.order",
    )
    usage: Mapped["MessageUsage | None"] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        uselist=False,
    )
    feedback: Mapped["MessageFeedback | None"] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        uselist=False,
    )


class MessageSource(Base):
    __tablename__ = "message_sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(), primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False, index=True)
    chunk_id: Mapped[str] = mapped_column(String(64), nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUIDType(), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    rag_id: Mapped[uuid.UUID] = mapped_column(UUIDType(), nullable=False)
    order: Mapped[int] = mapped_column(Integer, nullable=False)

    message: Mapped["Message"] = relationship(back_populates="sources")

    __table_args__ = (
        UniqueConstraint("message_id", "order",
                         name="uq_source_message_order"),
    )


class MessageUsage(Base):
    __tablename__ = "message_usage"

    id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(), primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False, unique=True)
    prompt_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0)
    model: Mapped[str] = mapped_column(String(128), nullable=False)

    message: Mapped["Message"] = relationship(back_populates="usage")


class MessageFeedback(Base):
    __tablename__ = "message_feedback"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True, autoincrement=True)
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False, unique=True)
    data: Mapped[dict] = mapped_column(
        JSONBType(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False)

    message: Mapped["Message"] = relationship(back_populates="feedback")
