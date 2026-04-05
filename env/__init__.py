"""OpenEnv Support Triage - AI-powered customer support ticket classification.

Core Components:
- SupportEnv: Main RL environment
- Models: Pydantic-based data structures
- Reward: Multi-signal reward computation
- Graders: Deterministic evaluation with partial credit
"""

from .environment import SupportEnv
from .models import Ticket, Observation, Action, Reward

__version__ = "1.0.0"
__all__ = ["SupportEnv", "Ticket", "Observation", "Action", "Reward"]
