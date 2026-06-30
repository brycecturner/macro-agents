from __future__ import annotations

from pydantic import BaseModel, field_validator

from app.models.enums import Direction


class ThesisCreate(BaseModel):
    """Schema for creating a new thesis via the JSON API."""

    thesis_title: str
    time_horizon: str
    direction: Direction
    notes: str

    @field_validator("thesis_title", "time_horizon", "notes")
    @classmethod
    def string_fields_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("This field is required.")
        return v.strip()

    @field_validator("direction", mode="before")
    @classmethod
    def direction_must_be_provided(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            raise ValueError("Direction is required.")
        return v
