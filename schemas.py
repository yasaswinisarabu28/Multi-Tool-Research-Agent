"""
Pydantic schemas for each tool's input arguments.

Why this matters: these are the exact same schemas you'll expose as
MCP tool definitions in Project 4. Get comfortable with this pattern now.
"""

from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query to look up on the web")


class CalculatorInput(BaseModel):
    expression: str = Field(
        ..., description="A math expression to evaluate, e.g. '12 * (4 + 3)'"
    )


class WikipediaInput(BaseModel):
    topic: str = Field(..., description="Topic or entity to look up on Wikipedia")

