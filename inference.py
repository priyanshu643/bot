import os
import json
import time
import re
from openai import OpenAI
from env import DataCenterEnv
from models import Action  # We must import our strict Action model!

# ==========================================
# MANDATORY HACKATHON VARIABLES
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

FALLBACK_ACTION = {"action": "do_nothing", "source": "", "target": "", "reason": "Parsing failed"}


def get_ai_action(client, current_state):
    # FIX 1: Use Pydantic's built-in `.model_dump_json()` instead of standard json.dumps()
    prompt = f"""
    You are an AI managing a data center.
    Current Data: {current_state.model_dump_json()}

    RULES:
    1. If a server has high temp (> 80) AND high load (> 70): action is "transfer_load" to a server with low load (< 30).
    2. If a server has high temp (> 80) AND low load (< 70): action is "cool_server" (apply fans).
    3. If a server has critical temp (>= 90): action is "warn_critical".
    4. Otherwise, action is "do_nothing".

    CRITICAL INSTRUCTION: You can only take ONE action per cycle. Find the SINGLE most urgent server problem and fix that one. 
    You MUST reply ONLY with ONE valid JSON object in this exact format, with no extra text:
    {{"action": "...", "source": "server_X", "target": "server_Y", "reason": "..."}}
    (Note: If action is cool_server or warn_critical, target can be empty "").
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You output a single strict JSON object only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.1
        )

        content = completion.choices[0].message.content.strip()

        # Robust Parsing
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            return json.loads(content)

    except Exception as exc:
        print(f"Model parsing failed ({exc}).")
        return FALLBACK_ACTION


def main() -> None:
    if not API_KEY:
        print("ERROR: HF_TOKEN environment variable is not set!")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DataCenterEnv()

    print(f"Starting Data Center AI Agent using model: {MODEL_NAME}")
    state = env.reset()

    while True:
        try:
            if not state or not state.servers:
                print("Waiting for valid Firebase data...")
                time.sleep(2)
                state = env.reset()
                continue

            print("\n--- New Cycle ---")

            # The AI returns a dictionary
            decision_dict = get_ai_action(client, state)

            # FIX 2: Convert the raw dictionary into our strict Pydantic Action model
            try:
                action_obj = Action(**decision_dict)
            except Exception as e:
                print(f"Failed to build Action model: {e}")
                action_obj = Action(**FALLBACK_ACTION)

            # FIX 3: Unpack all 4 variables returned by the new OpenEnv spec (State, Reward, Done, Info)
            next_state, reward, done, info = env.step(action_obj)
            print(f"Score: {reward} | Info: {info}")

            if done:
                state = env.reset()  # This will automatically move to the next task if programmed, or restart
            else:
                state = next_state

            time.sleep(6)

        except Exception as e:
            print(f"Error occurred in loop: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()

