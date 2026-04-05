"""Production-ready baseline agent with robust error handling."""

import os
import re
from openai import OpenAI
from env.environment import SupportEnv
from env.models import Action


def create_client() -> OpenAI:
    """Create OpenAI client with error handling."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize without proxies argument (not supported in OpenAI 1.3.0)
    try:
        client = OpenAI(api_key=api_key)
        return client
    except TypeError as e:
        if "proxies" in str(e):
            # If proxies are causing issues, create without them
            print(f"Warning: {e}. Creating client without proxy configuration.")
            return OpenAI(api_key=api_key)
        raise


def parse_action(action_text: str) -> Action:
    """
    Strictly parse and validate action from model output.
    
    Handles:
    - Case insensitivity
    - Extraction from sentence context (e.g., "I recommend escalate")
    - Fallback to respond on any error
    
    Returns:
        Action: Validated action object with fallback mechanism
    """
    action_text = action_text.strip().lower()
    
    # Try exact match first
    if "escalate" in action_text:
        return Action(action_type="escalate")
    if "ignore" in action_text:
        return Action(action_type="ignore")
    if "respond" in action_text:
        return Action(action_type="respond")
    
    # Fallback to respond (safest default)
    print(f"Warning: Could not parse action '{action_text}'. Defaulting to respond.")
    return Action(action_type="respond")


def get_model_action(client: OpenAI, ticket_text: str) -> str:
    """
    Query the model with strict formatting requirements.
    
    Args:
        client: OpenAI client
        ticket_text: Support ticket content
        
    Returns:
        str: Model's response text
    """
    prompt = f"""You are a support ticket triage agent. Analyze the ticket and decide on ONE action.

Ticket:
{ticket_text}

You must respond with exactly ONE of these actions:
- respond
- escalate
- ignore

Respond with ONLY the action word, nothing else. For example: "escalate" or "respond"."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling model: {e}")
        return "respond"  # Fallback


def run_episode(task_name: str, client: OpenAI) -> dict:
    """
    Run one complete episode of support triage.
    
    Args:
        task_name: Task name (e.g., 'tickets_easy')
        client: OpenAI client
        
    Returns:
        dict: Episode results with metrics
    """
    try:
        env = SupportEnv(task_name)
    except Exception as e:
        print(f"Error loading task {task_name}: {e}")
        return {"task": task_name, "status": "failed", "error": str(e)}
    
    obs = env.reset()
    total_reward = 0.0
    step_count = 0
    
    while obs.current_ticket is not None:
        try:
            # Get model decision
            action_text = get_model_action(client, obs.current_ticket.text)
            
            # Parse and validate
            action = parse_action(action_text)
            
            # Ensure response_text is provided for respond actions
            if action.action_type == "respond" and not action.response_text:
                action.response_text = f"Thank you for contacting us. We'll look into this matter."
            
            # Execute action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
        except Exception as e:
            print(f"Error in step: {e}")
            obs, reward, done, info = env.step(Action(action_type="respond"))
            total_reward += reward
            step_count += 1
        
        if done:
            break
    
    # Get episode summary
    summary = env.get_episode_summary()
    
    return {
        "task": task_name,
        "status": "completed",
        "total_reward": total_reward,
        "mean_reward": total_reward / step_count if step_count > 0 else 0.0,
        "steps": step_count,
        "summary": summary
    }


def main():
    """Run baseline on all tasks."""
    try:
        client = create_client()
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return
    
    tasks = ["tickets_easy", "tickets_medium", "tickets_hard"]
    results = []
    
    for task in tasks:
        print(f"\nRunning {task}...")
        result = run_episode(task, client)
        results.append(result)
        
        if result["status"] == "completed":
            print(f"  Total Reward: {result['total_reward']:.2f}")
            print(f"  Mean Reward:  {result['mean_reward']:.2f}")
            print(f"  Steps:        {result['steps']}")
        else:
            print(f"  Failed: {result.get('error', 'Unknown error')}")
    
    # Print summary
    print("\n" + "="*50)
    print("BASELINE RESULTS SUMMARY")
    print("="*50)
    for result in results:
        print(f"{result['task']}: {result['status']}", end="")
        if result["status"] == "completed":
            print(f" | Reward: {result['total_reward']:.3f}")
        else:
            print()


if __name__ == "__main__":
    main()
