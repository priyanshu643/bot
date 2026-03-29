# env.py
import requests
import json
from datetime import datetime
from models import Observation, Action, ServerState


class DataCenterEnv:
    def __init__(self):
        self.base_url = "https://buzzer-alarm--system-default-rtdb.asia-southeast1.firebasedatabase.app"
        self.datacenter_url = f"{self.base_url}/datacenter.json"
        self.action_url = f"{self.base_url}/ai_action.json"

        self.current_state = None
        self.current_task = None
        self.step_count = 0
        self.max_steps = 10  # Fails the task if it takes too long

    def _get_firebase_data(self) -> Observation:
        """Fetches live data and forces it into the strict Pydantic Observation model."""
        try:
            data = requests.get(self.datacenter_url).json()
            servers = {}
            if data:
                for s_id, stats in data.items():
                    if stats:
                        servers[s_id] = ServerState(load=stats.get('load', 0), temp=stats.get('temp', 0))
            return Observation(servers=servers)
        except Exception as e:
            print(f"Firebase read error: {e}")
            return Observation(servers={})

    def state(self) -> Observation:
        """Required by OpenEnv spec to return the current state."""
        return self._get_firebase_data()

    def reset(self, task_name="easy_cooling") -> Observation:
        """
        Required by OpenEnv spec. 
        Sets up the specific task scenarios (Easy, Medium, Hard) and resets the step counter.
        """
        self.current_task = task_name
        self.step_count = 0
        print(f"\n[ENVIRONMENT RESET] Initiating Task: {task_name}")

        # Baseline: everything is normal
        initial_setup = {f"server_{i}": {"load": 30, "temp": 45} for i in range(1, 11)}

        # Inject the specific problems based on the task difficulty
        if task_name == "easy_cooling":
            # EASY: One server is overheating but has low load. Requires 'cool_server'.
            initial_setup["server_1"] = {"load": 10, "temp": 95}

        elif task_name == "medium_balancing":
            # MEDIUM: Server 1 is overloaded and hot. Server 2 is completely empty. Requires 'transfer_load'.
            initial_setup["server_1"] = {"load": 90, "temp": 90}
            initial_setup["server_2"] = {"load": 10, "temp": 40}

        elif task_name == "hard_cascade":
            # HARD: Multiple servers overloaded. Requires triage and multiple transfers.
            initial_setup["server_1"] = {"load": 95, "temp": 98}
            initial_setup["server_2"] = {"load": 85, "temp": 89}
            initial_setup["server_3"] = {"load": 80, "temp": 85}

        # Push the scenario to Firebase (This will instantly update your web dashboard!)
        requests.put(self.datacenter_url, json=initial_setup)
        self.current_state = self.state()
        return self.current_state

    def step(self, action: Action):
        """
        Required by OpenEnv spec. 
        Takes a Pydantic Action, applies physics, and returns (Observation, Reward, Done, Info).
        """
        self.step_count += 1
        print(f">> Step {self.step_count}/{self.max_steps} | Action: {action.action} on {action.source}")

        # 1. Update UI
        requests.put(self.action_url, json=action.model_dump())

        # 2. Apply Physics
        state_dict = self.current_state.servers
        if action.source in state_dict:
            source_data = state_dict[action.source]

            if action.action == "transfer_load" and action.target in state_dict:
                target_data = state_dict[action.target]

                # Smart Load Balancing Math
                balanced_load = (source_data.load + target_data.load) // 2
                load_shifted = source_data.load - balanced_load

                new_source = {"load": balanced_load, "temp": max(40, source_data.temp - (load_shifted // 2))}
                new_target = {"load": balanced_load, "temp": min(100, target_data.temp + (load_shifted // 2))}

                requests.patch(f"{self.base_url}/datacenter/{action.source}.json", json=new_source)
                requests.patch(f"{self.base_url}/datacenter/{action.target}.json", json=new_target)

            elif action.action == "cool_server":
                new_source = {"load": source_data.load, "temp": 45}
                requests.patch(f"{self.base_url}/datacenter/{action.source}.json", json=new_source)

        # 3. Get new state
        self.current_state = self.state()

        # 4. Grader Logic (Calculates the 0.0 to 1.0 score required by the hackathon)
        score = self._calculate_grader_score()

        # Check if the episode is finished (Success or ran out of time)
        done = False
        if score == 1.0 or self.step_count >= self.max_steps:
            done = True
            print(f"[TASK COMPLETE] Final Score: {score}")

        info = {"task": self.current_task, "steps_taken": self.step_count}

        return self.current_state, score, done, info

    def _calculate_grader_score(self) -> float:
        """Evaluates if the AI successfully solved the specific task's problem (Returns 0.0 - 1.0)."""
        servers = self.current_state.servers

        if self.current_task == "easy_cooling":
            # Success: Server 1 is cooled down
            if "server_1" in servers and servers["server_1"].temp < 75:
                return 1.0

        elif self.current_task == "medium_balancing":
            # Success: Server 1 and 2 are balanced and cool
            if "server_1" in servers and "server_2" in servers:
                s1, s2 = servers["server_1"], servers["server_2"]
                if s1.temp < 80 and s2.temp < 80 and abs(s1.load - s2.load) < 10:
                    return 1.0

        elif self.current_task == "hard_cascade":
            # Success: All 3 critical servers are stabilized
            safe_count = 0
            for sid in ["server_1", "server_2", "server_3"]:
                if sid in servers and servers[sid].temp < 85 and servers[sid].load < 80:
                    safe_count += 1
            if safe_count == 3:
                return 1.0
            return safe_count * 0.33  # Partial credit for saving some servers

        return 0.0  # Task is not yet solved
