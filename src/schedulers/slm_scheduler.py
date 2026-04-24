import collections
import os
import json
import re
import time
import requests
import urllib3
from dotenv import load_dotenv

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY        = os.environ.get("GEMINI_API_KEY")
SLM_MODEL      = os.environ.get("SLM_MODEL", "gemma-3n-e4b-it")
NPU_LATENCY_MS = float(os.environ.get("SLM_NPU_LATENCY_MS", "50.0"))
RPM_LIMIT      = int(os.environ.get("SLM_RPM_LIMIT", "14"))  # conservative under the 15 RPM cap

_PROMPT_TEMPLATE = """\
CONTEXT: You are an AI scheduler embedded in a Low-Earth-Orbit satellite (edge node). \
You have strict resource constraints and must make fast routing decisions. \
Output ONLY valid JSON — no markdown, no explanation.

TASK:
  id: {task_id}
  region: {region}
  ram_required: {ram} MB
  anomaly: "{anomaly}"

AVAILABLE SATELLITES:
{fleet_lines}

ROUTING RULES (apply in order, use semantic reasoning on the anomaly field):
  1. Data privacy regulations (e.g. GDPR, EU privacy law, or similar): route exclusively to a satellite
     in the geographically relevant region (e.g. EUROPE for EU regulations). Drop if none available.
  2. Data sovereignty constraints (e.g. national laws requiring data to stay within a country's jurisdiction):
     route exclusively to a satellite in the required country's region. Drop if none available.
  3. Critical hardware failure that makes the task unexecutable (e.g. a sensor, camera, or component
     required to process this task is broken): drop the task entirely.
  4. No anomaly or unknown anomaly: route to any satellite in the task's region with link_quality >= 20%
     and sufficient RAM. Prefer the satellite with the highest link_quality.
  5. Use action='process' when routing to the task's own region, action='route' when forwarding to a
     different region, action='drop' only when no compliant satellite exists or a fatal hardware failure applies.

Output ONLY valid JSON:
{{"action": "process|route|drop", "target_region": "USA|BRAZIL|EUROPE|null", "reason": "one short sentence"}}
"""


class SLMScheduler:
    def __init__(self):
        if not API_KEY:
            print(">>> [SLM] WARNING: GEMINI_API_KEY not found. API calls will return None (drop).")
        else:
            print(f">>> [ENGINE] SLM Scheduler Initialized ({SLM_MODEL} via Gemini API | NPU latency={NPU_LATENCY_MS}ms)")
        self.npu_busy_until = 0.0
        self.pending_tasks = []
        self._call_timestamps: collections.deque = collections.deque()

    # ------------------------------------------------------------------ #
    # Async state machine                                                  #
    # ------------------------------------------------------------------ #

    def receive_task(self, task_dict, current_time, fleet):
        ready_time = max(current_time, self.npu_busy_until) + (NPU_LATENCY_MS / 1000.0)
        self.npu_busy_until = ready_time
        task_dict["slm_ready_time"] = ready_time
        task_dict["slm_fleet_snapshot"] = list(fleet)
        self.pending_tasks.append(task_dict)
        return ready_time

    def check_completed_inferences(self, current_time):
        completed, remaining = [], []
        for task in self.pending_tasks:
            if current_time >= task.get("slm_ready_time", 0):
                completed.append((task, self._decide_route(task)))
            else:
                remaining.append(task)
        self.pending_tasks = remaining
        return completed

    # ------------------------------------------------------------------ #
    # Decision logic                                                       #
    # ------------------------------------------------------------------ #

    def _decide_route(self, task_dict):
        fleet = task_dict.get("slm_fleet_snapshot", [])
        t0 = time.perf_counter()
        api_result = self._call_gemini(task_dict, fleet)
        result = self._resolve_action(api_result, fleet, task_dict) if api_result is not None else None
        task_dict["slm_wall_latency_ms"] = (time.perf_counter() - t0) * 1000
        return result

    def _build_prompt(self, task_dict, fleet):
        fleet_lines = "\n".join(
            f"  SAT {s['id']} | region={s['region']} | link_quality={s['link_quality']:.0f}%"
            f" | ram_free={s['ram_free']} MB"
            for s in fleet
        )
        return _PROMPT_TEMPLATE.format(
            task_id=task_dict.get("id"),
            region=task_dict.get("region", "?"),
            ram=task_dict.get("ram", 0),
            anomaly=task_dict.get("semantic_anomaly", "none"),
            fleet_lines=fleet_lines,
        )

    def _rate_limit_wait(self):
        """Block until making another API call stays within RPM_LIMIT calls/minute."""
        now = time.monotonic()
        while self._call_timestamps and now - self._call_timestamps[0] > 60.0:
            self._call_timestamps.popleft()
        if len(self._call_timestamps) >= RPM_LIMIT:
            wait = 61.0 - (now - self._call_timestamps[0])
            if wait > 0:
                print(f"   [SLM Rate Limiter] {len(self._call_timestamps)} calls/min — waiting {wait:.1f}s")
                time.sleep(wait)
            now = time.monotonic()
            while self._call_timestamps and now - self._call_timestamps[0] > 60.0:
                self._call_timestamps.popleft()
        self._call_timestamps.append(time.monotonic())

    def _call_gemini(self, task_dict, fleet):
        if not API_KEY:
            return None
        self._rate_limit_wait()
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{SLM_MODEL}:generateContent?key={API_KEY}")
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": self._build_prompt(task_dict, fleet)}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 128,
            },
        }
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                r = requests.post(url, headers=headers, json=data, verify=False, timeout=30)
                latencia_ms = (time.perf_counter() - t0) * 1000
                if r.status_code == 200:
                    raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    match = re.search(r'\{.*?\}', raw, re.DOTALL)
                    if not match:
                        print(f"   [SLM Parse Error] No JSON object found in: {raw[:80]}")
                        break
                    result = json.loads(match.group())
                    print(f"   [SLM {SLM_MODEL} {latencia_ms:.0f}ms] action={result.get('action')}"
                          f" target={result.get('target_region')} reason={result.get('reason', '')}")
                    return result
                elif r.status_code == 429:
                    print(f"   [SLM Rate Limit] Tentativa {attempt+1}/3")
                    continue
                else:
                    print(f"   [SLM Error] HTTP {r.status_code} — {r.text[:120]}")
                    break
            except Exception as e:
                print(f"   [SLM Connection Error] {e}")
                continue
        return None

    def _resolve_action(self, api_result, fleet, task_dict):
        action = api_result.get("action", "drop")
        target_region = api_result.get("target_region")
        if action == "drop":
            return None
        if action == "process":
            target_region = task_dict.get("region")
        if target_region:
            for s in fleet:
                if (s.get("region") == target_region
                        and s.get("link_quality", 0) >= 20.0
                        and s.get("ram_free", 0) >= task_dict.get("ram", 0)):
                    return s.get("id")
        return None
