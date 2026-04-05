"""Enhanced reward calculation with rich learning signals."""

from env.models import Ticket, Action, Reward


def is_urgent(ticket: Ticket) -> bool:
    """Detect urgent tickets by keyword analysis."""
    urgent_keywords = ["urgent", "asap", "critical", "down", "failed", "broken", "error"]
    text_lower = ticket.text.lower()
    return any(keyword in text_lower for keyword in urgent_keywords)


def is_spam(ticket: Ticket) -> bool:
    """Detect spam tickets."""
    spam_keywords = ["buy now", "limited offer", "click here", "spam", "viagra"]
    text_lower = ticket.text.lower()
    return any(keyword in text_lower for keyword in spam_keywords)


def is_refund_related(ticket: Ticket) -> bool:
    """Detect refund-related tickets requiring escalation."""
    refund_keywords = ["refund", "money back", "reimbursement", "charge dispute"]
    text_lower = ticket.text.lower()
    return any(keyword in text_lower for keyword in refund_keywords)


def compute_reward(ticket: Ticket, action: Action) -> Reward:
    """
    Compute comprehensive reward with multiple signals.
    
    Signals:
    - Urgency handling (correct escalation of urgent tickets)
    - Spam filtering (correctly ignoring spam)
    - Tier-weighted responses (enterprise tickets more valuable)
    - Response quality (meaningful response text)
    - Efficiency (prefer early correct decisions)
    - Refund escalation (always escalate refund requests)
    
    Args:
        ticket: The support ticket
        action: The agent's action
        
    Returns:
        Reward: Score [0, 1] with component breakdown
    """
    components = {}
    base_score = 0.0
    
    # 1. SPAM HANDLING (high priority, simple decision)
    if is_spam(ticket):
        if action.action_type == "ignore":
            components["spam_handling"] = 0.3
            base_score += 0.3
        else:
            components["spam_handling"] = -0.1
            base_score -= 0.1
    
    # 2. REFUND ESCALATION (mandatory escalation)
    elif is_refund_related(ticket):
        if action.action_type == "escalate":
            components["refund_escalation"] = 0.4
            base_score += 0.4
        elif action.action_type == "ignore":
            components["refund_escalation"] = -0.5
            base_score -= 0.5
        else:
            components["refund_escalation"] = 0.1
            base_score += 0.1
    
    # 3. URGENCY-BASED DECISIONS
    elif is_urgent(ticket):
        if action.action_type == "escalate":
            components["urgency_handling"] = 0.4
            base_score += 0.4
        elif action.action_type == "ignore":
            components["urgency_handling"] = -0.5
            base_score -= 0.5
        else:
            components["urgency_handling"] = 0.2
            base_score += 0.2
    
    # 4. NORMAL TICKETS (respond is preferred)
    else:
        if action.action_type == "respond":
            components["normal_response"] = 0.25
            base_score += 0.25
        elif action.action_type == "ignore":
            components["normal_response"] = 0.15
            base_score += 0.15
        else:
            components["normal_response"] = 0.1
            base_score += 0.1
    
    # 5. RESPONSE QUALITY (bonus for meaningful responses)
    if action.action_type == "respond" and action.response_text:
        if len(action.response_text.strip()) >= 10:
            components["response_quality"] = 0.15
            base_score += 0.15
        else:
            components["response_quality"] = 0.05
            base_score += 0.05
    
    # 6. CUSTOMER TIER MULTIPLIER (enterprise customers are more valuable)
    tier_multiplier = {"free": 1.0, "pro": 1.2, "enterprise": 1.5}
    multiplier = tier_multiplier.get(ticket.customer_tier, 1.0)
    
    # Apply multiplier to base score, but cap at 1.0
    final_score = min(base_score * multiplier, 1.0)
    components["tier_multiplier"] = multiplier
    
    # Generate reason
    reason = _generate_reason(ticket, action, components)
    
    return Reward(
        score=max(0.0, min(final_score, 1.0)),  # Clamp to [0, 1]
        reason=reason,
        components=components
    )


def _generate_reason(ticket: Ticket, action: Action, components: dict) -> str:
    """Generate human-readable explanation for reward."""
    action_str = action.action_type.capitalize()
    
    if is_spam(ticket):
        if action.action_type == "ignore":
            return f"{action_str}: Correctly ignored spam ticket"
        return f"{action_str}: Should have ignored spam ticket"
    
    if is_refund_related(ticket):
        if action.action_type == "escalate":
            return f"{action_str}: Correctly escalated refund request"
        return f"{action_str}: Refund requests must be escalated"
    
    if is_urgent(ticket):
        if action.action_type == "escalate":
            return f"{action_str}: Correctly escalated urgent issue"
        elif action.action_type == "ignore":
            return f"{action_str}: Urgent issues should not be ignored"
        return f"{action_str}: Responded to urgent issue"
    
    # Normal ticket
    if action.action_type == "respond":
        if action.response_text and len(action.response_text.strip()) >= 10:
            return f"{action_str}: Provided detailed response to customer"
        return f"{action_str}: Responded to customer inquiry"
    
    if action.action_type == "escalate":
        return f"{action_str}: Escalated for specialist review"
    
    return f"{action_str}: Processed ticket"
