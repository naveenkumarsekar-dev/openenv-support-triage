"""Enhanced grading with deterministic partial correctness scoring."""

from typing import List, Dict, Any
from env.models import Ticket


def is_urgent(ticket: Ticket) -> bool:
    """Detect urgent tickets."""
    urgent_keywords = ["urgent", "asap", "critical", "down", "failed", "broken", "error"]
    return any(keyword in ticket.text.lower() for keyword in urgent_keywords)


def is_spam(ticket: Ticket) -> bool:
    """Detect spam tickets."""
    spam_keywords = ["buy now", "limited offer", "click here", "spam", "viagra"]
    return any(keyword in ticket.text.lower() for keyword in spam_keywords)


def is_refund_related(ticket: Ticket) -> bool:
    """Detect refund requests."""
    refund_keywords = ["refund", "money back", "reimbursement", "charge dispute"]
    return any(keyword in ticket.text.lower() for keyword in refund_keywords)


def get_ideal_action(ticket: Ticket) -> str:
    """
    Deterministic ideal action mapping based on ticket properties.
    
    Rules (applied in priority order):
    1. Spam → ignore
    2. Refund request → escalate
    3. Urgent issue → escalate
    4. Normal → respond
    """
    if is_spam(ticket):
        return "ignore"
    if is_refund_related(ticket):
        return "escalate"
    if is_urgent(ticket):
        return "escalate"
    return "respond"


def score_action(ticket: Ticket, actual_action: str, ideal_action: str) -> float:
    """
    Score an action on a partial correctness scale.
    
    - Perfect match: 1.0
    - Reasonable alternative: 0.5
    - Wrong action: 0.0
    
    Reasonable alternatives:
    - Escalate vs Respond for urgent (both attempt to help)
    - Ignore spam is ideal, but escalating spam is at least not responding helpfully (-0.2)
    """
    if actual_action == ideal_action:
        return 1.0
    
    # Define reasonable alternatives
    if ideal_action == "escalate":
        # For urgent/refund, responding is better than ignoring
        if actual_action == "respond":
            return 0.5
        # Ignoring urgent/refund is worst
        return 0.0
    
    if ideal_action == "respond":
        # For normal tickets, escalating is acceptable (wastes resources but helps)
        if actual_action == "escalate":
            return 0.5
        # Ignoring normal tickets is bad
        return 0.0
    
    if ideal_action == "ignore":
        # For spam, responding/escalating is worse than ignoring
        if actual_action == "respond":
            return 0.3
        if actual_action == "escalate":
            return 0.2
        return 0.0
    
    return 0.0


def grade_episode(history: List[Dict[str, Any]], tickets: List[Ticket]) -> Dict[str, Any]:
    """
    Deterministic episode grading with partial credit.
    
    Returns metrics:
    - accuracy: Perfect action match rate
    - partial_accuracy: Including partial credit
    - mistake_count: Number of wrong decisions
    - category_scores: Performance per ticket category
    """
    if not history or not tickets:
        return {
            "accuracy": 0.0,
            "partial_accuracy": 0.0,
            "mistake_count": 0,
            "category_scores": {},
            "details": []
        }
    
    perfect_matches = 0
    partial_score = 0.0
    mistake_count = 0
    category_results = {
        "spam": [],
        "refund": [],
        "urgent": [],
        "normal": []
    }
    
    details = []
    
    # Match history entries to tickets
    for i, action_record in enumerate(history):
        if i >= len(tickets):
            break
        
        ticket = tickets[i]
        actual_action = action_record.get("action", "respond")
        ideal_action = get_ideal_action(ticket)
        
        # Score this action
        score = score_action(ticket, actual_action, ideal_action)
        partial_score += score
        
        if score == 1.0:
            perfect_matches += 1
        elif score == 0.0:
            mistake_count += 1
        
        # Categorize result
        if is_spam(ticket):
            category_results["spam"].append(score)
        elif is_refund_related(ticket):
            category_results["refund"].append(score)
        elif is_urgent(ticket):
            category_results["urgent"].append(score)
        else:
            category_results["normal"].append(score)
        
        details.append({
            "ticket_id": ticket.id,
            "ideal_action": ideal_action,
            "actual_action": actual_action,
            "score": score,
            "category": "spam" if is_spam(ticket) else "refund" if is_refund_related(ticket) else "urgent" if is_urgent(ticket) else "normal"
        })
    
    # Compute category-wise metrics
    category_scores = {}
    for category, scores in category_results.items():
        if scores:
            category_scores[category] = {
                "mean_score": sum(scores) / len(scores),
                "count": len(scores),
                "perfect": sum(1 for s in scores if s == 1.0)
            }
    
    accuracy = perfect_matches / len(history) if history else 0.0
    partial_accuracy = partial_score / len(history) if history else 0.0
    
    return {
        "accuracy": accuracy,
        "partial_accuracy": partial_accuracy,
        "perfect_matches": perfect_matches,
        "mistake_count": mistake_count,
        "total_actions": len(history),
        "category_scores": category_scores,
        "details": details
    }
