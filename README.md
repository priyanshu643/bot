# Autonomous Data Center Thermal OpenEnv

## Environment Description & Motivation
Managing thermal loads in a modern data center is a highly complex cyber-physical problem. Traditional static load balancing often ignores the physical thermal realities of server racks, leading to hardware degradation or inefficient cooling energy spikes. 

This environment simulates a real-world IoT Data Center. An autonomous agent must monitor live server telemetry (CPU load and temperature) and issue precise commands to either transfer computational loads between servers or activate physical cooling systems. It bridges the gap between software load balancing and physical thermal management.

## Observation Space
The environment state is a dictionary mapping server IDs to their current telemetry, using strict Pydantic models:
* `load`: CPU/Power load percentage (0 - 100)
* `temp`: Temperature in Celsius (0 - 120)

## Action Space
The agent outputs a strict JSON object (validated via Pydantic) with the following parameters:
* `action`: The command to execute (`transfer_load`, `cool_server`, `warn_critical`, or `do_nothing`).
* `source`: The ID of the server experiencing the anomaly.
* `target`: The ID of the destination server (used only for `transfer_load`).
* `reason`: The agent's logical justification for the action.

## Graded Tasks
The environment features 3 distinct tasks to evaluate agent logic, each returning a score from `0.0` (Failure) to `1.0` (Success):
1. **`easy_cooling`**: A single server is overheating but has low load. The agent must identify that shifting load is impossible and successfully apply physical cooling.
2. **`medium_balancing`**: One server is overloaded and overheating, while another is idle. The agent must calculate and execute a load transfer to perfectly balance the two machines.
3. **`hard_cascade`**: Multiple servers are simultaneously reaching critical thresholds. The agent must triage the environment and execute multiple sequential actions within the step limit to prevent a thermal cascade.

## Setup and Usage Instructions
### Local Execution
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your environment variables:
   * `HF_TOKEN` = Your Hugging Face API Key
   * `MODEL_NAME` = `meta-llama/Meta-Llama-3-8B-Instruct`
   * `API_BASE_URL` = `https://router.huggingface.co/v1`
4. Run the baseline evaluation: `python inference.py`

### Docker Execution
```bash
docker run -it -p 7860:7860 --platform=linux/amd64 \
	-e HF_TOKEN="YOUR_VALUE_HERE" \
	registry.hf.space/priyanshukumarsinha-bot:latest
  ```
