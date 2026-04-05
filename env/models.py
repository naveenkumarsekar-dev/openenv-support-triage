from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal, Dict
from datetime import datetime


class Ticket(BaseModel):
    """Support ticket with urgency and customer tier information."""
    id: str
    text: str
    customer_tier: Literal["free", "pro", "enterprise"]
    timestamp: int


class Observation(BaseModel):
    """Current environment state observation."""
    current_ticket: Optional[Ticket] = None
    remaining_tickets: int = 0
    history: List[dict] = Field(default_factory=list)
    episode_step: int = 0


class Action(BaseModel):
    """Agent action with strict validation."""
    action_type: Literal["respond", "escalate", "ignore"]
    response_text: str = Field(default="", min_length=0)
    priority: Optional[Literal["low", "medium", "high"]] = None
    
    @validator("response_text")
    def validate_response_text(cls, v):
        if v is None:
            return ""
        return str(v).strip()


class Reward(BaseModel):
    """Detailed reward feedback with decomposition."""
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    components: Dict[str, float] = Field(default_factory=dict)