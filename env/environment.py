import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from env.models import Ticket, Observation, Action, Reward
from env.reward import compute_reward
from env.graders import grade_episode


class SupportEnv:
    """Robust support ticket triage environment with clean state management."""
    
    def __init__(self, task_name: str):
        """
        Initialize environment with task.
        
        Args:
            task_name: Name of task file (e.g., 'tickets_easy', 'tickets_medium', 'tickets_hard')
        """
        self.task_name = task_name
        self.tickets: List[Ticket] = []
        self.current_idx = 0
        self.history = []
        self.episode_rewards = []
        self._load_task(task_name)
    
    def _load_task(self, task_name: str) -> None:
        """Load tickets from JSON file."""
        data_path = Path(__file__).parent.parent / "data" / f"{task_name}.json"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Task file not found: {data_path}")
        
        with open(data_path, "r") as f:
            data = json.load(f)
        
        # Validate and parse tickets
        self.tickets = []
        for item in data:
            try:
                ticket = Ticket(
                    id=item.get("id", ""),
                    text=item.get("text", ""),
                    customer_tier=item.get("customer_tier", "free"),
                    timestamp=item.get("timestamp", 0)
                )
                self.tickets.append(ticket)
            except Exception as e:
                print(f"Warning: Failed to parse ticket {item}: {e}")
    
    def reset(self) -> Observation:
        """
        Reset environment to clean initial state.
        
        Returns:
            Observation: Initial observation with first ticket
        """
        self.current_idx = 0
        self.history = []
        self.episode_rewards = []
        
        if not self.tickets:
            # Handle empty task
            return Observation(
                current_ticket=None,
                remaining_tickets=0,
                history=[],
                episode_step=0
            )
        
        return self.state()
    
    def state(self) -> Observation:
        """
        Get current observation state.
        
        Returns:
            Observation: Current state including ticket, remaining count, and history
        """
        current_ticket = None
        if self.current_idx < len(self.tickets):
            current_ticket = self.tickets[self.current_idx]
        
        return Observation(
            current_ticket=current_ticket,
            remaining_tickets=len(self.tickets) - self.current_idx,
            history=self.history.copy(),
            episode_step=len(self.history)
        )
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute action on current ticket.
        
        Args:
            action: Action to take (respond, escalate, ignore)
        
        Returns:
            (observation, reward_score, done, info_dict)
        """
        if self.current_idx >= len(self.tickets):
            # Episode already finished
            obs = Observation(
                current_ticket=None, 
                remaining_tickets=0, 
                history=self.history.copy(), 
                episode_step=len(self.history)
            )
            return obs, 0.0, True, {"error": "Episode already finished"}
        
        current_ticket = self.tickets[self.current_idx]
        
        # Validate and sanitize action
        if action.action_type not in ["respond", "escalate", "ignore"]:
            action.action_type = "respond"  # Fallback
        
        # Compute reward
        reward = compute_reward(current_ticket, action)
        
        # Record action in history
        self.history.append({
            "ticket_id": current_ticket.id,
            "action": action.action_type,
            "response_text": action.response_text,
            "priority": action.priority,
            "reward": reward.score
        })
        self.episode_rewards.append(reward.score)
        
        # Move to next ticket
        self.current_idx += 1
        done = self.current_idx >= len(self.tickets)
        
        # Get next observation
        next_obs = self.state()
        
        info = {
            "ticket_id": current_ticket.id,
            "reward_components": reward.components,
            "reward_reason": reward.reason
        }
        
        return next_obs, reward.score, done, info
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for completed episode."""
        if not self.episode_rewards:
            return {"total_reward": 0.0, "mean_reward": 0.0, "actions_taken": 0}
        
        return {
            "total_reward": sum(self.episode_rewards),
            "mean_reward": sum(self.episode_rewards) / len(self.episode_rewards),
            "actions_taken": len(self.episode_rewards),
            "history": self.history
        }
