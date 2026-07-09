from __future__ import annotations

from pydantic import BaseModel, field_validator

from app.models.enums import ConditionType


class FalsificationConditionCreate(BaseModel):
    """Schema for creating or fully replacing a falsification condition."""

    description: str
    condition_type: ConditionType
    trigger_type: str | None = None
    measurable_proxy: str
    evaluation_logic: str

    @field_validator("description", "measurable_proxy", "evaluation_logic")
    @classmethod
    def string_fields_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("This field is required.")
        return v.strip()

    @field_validator("trigger_type", mode="before")
    @classmethod
    def blank_trigger_type_is_none(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            return None
        return v


class FalsificationConditionUpdate(FalsificationConditionCreate):
    """Schema for updating an existing falsification condition — same shape
    as create; a condition is always fully replaced, not patched."""
