# env.py
from models import Observation, Action, ServerState

class DataCenterEnv:
    def __init__(self):
        self.local_db = {}
        self.current_state = None
        self.current_task = None
        self.step_count = 0
        self.max_steps = 10

    def state(self) -> Observation:
        """Returns the current state from local memory."""
        servers = {}
        for s_id, stats in self.local_db.items():
            servers[s_id] = ServerState(load=int(stats.get('load', 0)), temp=int(stats.get('temp', 0)))
        return Observation(servers=servers)

    def reset(self, task_name="easy_cooling") -> Observation:
        """Sets up the specific task scenarios in local memory."""
        self.current_task = task_name
        self.step_count = 0
        print(f"\n[ENVIRONMENT RESET] Initiating Task: {task_name}")

        self.local_db = {f"server_{i}": {"load": 30, "temp": 45} for i in range(1, 11)}

        if task_name == "easy_cooling":
            self.local_db["server_1"] = {"load": 10, "temp": 95}
        elif task_name == "medium_balancing":
            self.local_db["server_1"] = {"load": 90, "temp": 90}
            self.local_db["server_2"] = {"load": 10, "temp": 40}
        elif task_name == "hard_cascade":
            self.local_db["server_1"] = {"load": 95, "temp": 98}
            self.local_db["server_2"] = {"load": 85, "temp": 89}
            self.local_db["server_3"] = {"load": 80, "temp": 85}

        self.current_state = self.state()
        return self.current_state

    def step(self, action: Action):
        """Applies physics to the local memory state with strict error catching."""
        self.step_count += 1
        print(f">> Step {self.step_count}/{self.max_steps} | Action: {action.action} on {action.source}")
        
        try:
            # Safe Physics Engine
            if action.source and action.source in self.local_db:
                source_data = self.local_db[action.source]
                
                if action.action == "transfer_load" and action.target and action.target in self.local_db:
                    target_data = self.local_db[action.target]
                    
                    balanced_load = (source_data["load"] + target_data["load"]) // 2
                    load_shifted = source_data["load"] - balanced_load
                    
                    self.local_db[action.source] = {"load": balanced_load, "temp": max(40, source_data["temp"] - (load_shifted // 2))}
                    self.local_db[action.target] = {"load": balanced_load, "temp": min(100, target_data["temp"] + (load_shifted // 2))}

                elif action.action == "cool_server":
                    self.local_db[action.source]["temp"] = 45
        except Exception as physics_err:
            print(f"[PHYSICS ERROR BYPASSED]: {physics_err}")

        self.current_state = self.state()

        # Safe Grader Logic
        try:
            score = self._calculate_grader_score()
        except Exception as e:
            print(f"[GRADER ERROR]: {e}")
            score = 0.0
        
        done = False
        if score >= 1.0 or self.step_count >= self.max_steps:
            done = True
            print(f"[TASK COMPLETE] Final Score: {score}")

        info = {"task": self.current_task, "steps_taken": self.step_count}
        
        return self.current_state, score, done, info

    def _calculate_grader_score(self) -> float:
        """Evaluates if the AI successfully solved the specific task's problem."""
        servers = self.current_state.servers
        
        if self.current_task == "easy_cooling":
            if "server_1" in servers and servers["server_1"].temp < 75:
                return 1.0
                
        elif self.current_task == "medium_balancing":
            if "server_1" in servers and "server_2" in servers:
                s1, s2 = servers["server_1"], servers["server_2"]
                if s1.temp < 80 and s2.temp < 80 and abs(s1.load - s2.load) < 10:
                    return 1.0
                    
        elif self.current_task == "hard_cascade":
            safe_count = 0
            for sid in ["server_1", "server_2", "server_3"]:
                if sid in servers and servers[sid].temp < 85 and servers[sid].load < 80:
                    safe_count += 1
            if safe_count == 3:
                return 1.0
            return safe_count * 0.33 

        return 0.0
