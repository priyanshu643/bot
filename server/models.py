# models.py
from pydantic import BaseModel, Field
from typing import Dict

class ServerState(BaseModel):
    """Defines the exact shape of a single server's telemetry."""
    load: int = Field(ge=0, le=100, description="CPU/Power load percentage")
    temp: int = Field(ge=0, le=120, description="Temperature in Celsius")

class Observation(BaseModel):
    """The full state of the data center passed to the AI."""
    servers: Dict[str, ServerState]

class Action(BaseModel):
    """The strict format the AI must use to issue commands."""
    action: str = Field(description="Must be: cool_server, transfer_load, warn_critical, or do_nothing")
    source: str = Field(default="", description="The server experiencing the issue")
    target: str = Field(default="", description="The destination server for load transfers")
    reason: str = Field(default="", description="AI's justification for the action")
