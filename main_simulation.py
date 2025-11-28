# main_simulation.py
'''
@desc
    Main simulation script for NTN-MEC scheduling with Real-World constraints.
    
    UPDATES V3 (Complex Scenarios):
    1. Geolocation Logic: Tasks can require specific regions (Data Sovereignty).
    2. Battery Logic: Satellites can start with depleted batteries.
    3. Advanced CoT Prompt: LLM evaluates Region -> RAM -> Battery -> CPU.
'''

import simpy
import random
import os
import re
import json
import google.generativeai as genai
from types import SimpleNamespace
from typing import TYPE_CHECKING

# --- Imports from CosmicBeats /src directory ---
try:
    from src.nodes.satellitebasic import SatelliteBasic, init_SatelliteBasic
    from src.utils import Time
    from src.simlogging.ilogger import ILogger, ELogType
except ImportError:
    print("Error: Could not import from 'src'.")
    print("Please run this script from the root directory of the CosmicBeats-Simulator.")
    exit(1)

# --- GLOBAL CONFIG ---
SIM_START_TIME_STR = "2025-11-10 20:30:00"

def get_current_sim_time(seconds_elapsed):
    t = Time()
    t.from_str(SIM_START_TIME_STR) 
    t.add_seconds(seconds_elapsed) 
    return t

class MockLogger(ILogger):
    def __init__(self, sim_env):
        self.env = sim_env
        self.__logTypeLevel = {}
    
    @property
    def logTypeLevel(self):
        return self.__logTypeLevel

    def write_Log(self, msg, log_type, timestamp):
        time_str = f"{self.env.now:.2f}"
        if isinstance(timestamp, Time):
            time_str = timestamp.to_str()
        print(f"[Time: {time_str}] {log_type}: {msg}")

# =============================================================================
# [UPDATED] Task Class (Now with Region Requirements)
# =============================================================================
class Task:
    def __init__(self, task_id: int, task_type: str, mips: float, 
                 data_in_mb: float, data_out_mb: float, deadline: float, 
                 creation_time: float, required_region: str = None):
        
        self.id = task_id
        self.type = task_type
        self.mips_required = mips
        self.data_input_size = data_in_mb
        self.data_output_size = data_out_mb
        self.deadline = deadline
        self.creation_time = creation_time
        
        # New Constraint: Data Sovereignty
        # If set (e.g., "BRAZIL"), task can ONLY be processed by sats in that region.
        self.required_region = required_region 

    def is_missed_deadline(self, current_time: float) -> bool:
        if self.deadline is None: return False
        return current_time > self.deadline

    def __str__(self):
        region_str = f"Region={self.required_region}" if self.required_region else "Global"
        return (f"Task(id={self.id}, Type={self.type}, Data={self.data_input_size}MB, "
                f"{region_str})")

# =============================================================================
# [UPDATED] SatelliteMEC Class (Region + Initial Battery)
# =============================================================================
class SatelliteMEC(SatelliteBasic):
    def __init__(self, _env: simpy.Environment, _mec_details, _logger, **kwargs) -> None:
        super().__init__(
            _nodeID=kwargs['_nodeID'],
            _topologyID=kwargs['_topologyID'],
            _tleline1=kwargs['_tleline1'],
            _tleline2=kwargs['_tleline2'],
            _timeDelta=kwargs['_timeDelta'],
            _timeStamp=kwargs['_timeStamp'],
            _endtime=kwargs['_endtime'],
            _Logger=_logger
        )
        
        self.env = _env
        self.logger = _logger

        # --- COMPUTATION ---
        self.cpu_capacity_mips = float(_mec_details.cpu_capacity_mips)
        self.cpu_resource = simpy.Resource(self.env, capacity=1)

        # --- MEMORY ---
        self.ram_capacity_mb = float(getattr(_mec_details, 'ram_capacity_mb', 4096.0))
        self.ram_container = simpy.Container(self.env, capacity=self.ram_capacity_mb, init=self.ram_capacity_mb)

        # --- I/O ---
        self.io_throughput_mbs = float(getattr(_mec_details, 'io_throughput_mbs', 500.0))

        # --- POWER (UPDATED) ---
        self.battery_capacity = float(getattr(_mec_details, 'battery_capacity_joules', 10000.0))
        
        # New: Allow starting with partial battery (to simulate dying satellites)
        initial_pct = getattr(_mec_details, 'initial_battery_pct', 100.0)
        self.current_battery = self.battery_capacity * (initial_pct / 100.0)
        
        self.power_idle = 5.0
        self.power_active = 20.0
        self.total_energy_consumed = 0.0
        self.battery_depleted = False
        
        # --- LOCATION (UPDATED) ---
        # For simulation simplicity, we assign a region string.
        # In full production, this would be calculated from TLE + Lat/Lon.
        self.current_region = getattr(_mec_details, 'region', 'GLOBAL')

        self.tasks_completed_count = 0
        self.tasks_dropped_count = 0

        self.env.process(self.battery_drain_process())

    # ... (Getters unchanged)
    def get_current_load_percentage(self) -> float:
        if self.cpu_resource.capacity == 0: return 0.0
        return (self.cpu_resource.count / self.cpu_resource.capacity) * 100.0

    def get_ram_usage_percentage(self) -> float:
        used = self.ram_capacity_mb - self.ram_container.level
        return (used / self.ram_capacity_mb) * 100.0
    
    def get_queue_length(self) -> int:
        return len(self.cpu_resource.queue)

    def process_task(self, task: Task):
        current_time_obj = get_current_sim_time(self.env.now)

        # 1. Battery Check
        if self.battery_depleted:
            self.logger.write_Log(f"Task {task.id} DROPPED. Sat {self.nodeID} DEAD (No Battery).", "LOGWARN", current_time_obj)
            self.tasks_dropped_count += 1
            return

        # 2. Region Check (Simulating Access Control enforcement)
        if task.required_region and task.required_region != self.current_region:
            self.logger.write_Log(f"Task {task.id} REJECTED. Region Mismatch (Task: {task.required_region}, Sat: {self.current_region}).", "LOGWARN", current_time_obj)
            self.tasks_dropped_count += 1
            return

        # 3. RAM Check
        if self.ram_container.level < task.data_input_size:
            self.logger.write_Log(f"Task {task.id} DROPPED. Sat {self.nodeID} OOM (Req: {task.data_input_size}MB, Free: {self.ram_container.level}MB).", "LOGWARN", current_time_obj)
            self.tasks_dropped_count += 1
            return

        yield self.ram_container.get(task.data_input_size)
        
        try:
            current_time_obj = get_current_sim_time(self.env.now)
            self.logger.write_Log(f"Task {task.id} accepted by Sat {self.nodeID} ({self.current_region}).", "LOGINFO", current_time_obj)

            io_time = task.data_input_size / self.io_throughput_mbs
            yield self.env.timeout(io_time)

            with self.cpu_resource.request() as req:
                yield req
                current_time_obj = get_current_sim_time(self.env.now)

                if task.is_missed_deadline(self.env.now):
                     self.logger.write_Log(f"Task {task.id} missed deadline in queue!", "LOGWARN", current_time_obj)

                processing_time = task.mips_required / self.cpu_capacity_mips
                yield self.env.timeout(processing_time)
                
                self.tasks_completed_count += 1
                total_time = self.env.now - task.creation_time
                current_time_obj = get_current_sim_time(self.env.now)
                self.logger.write_Log(f"Task {task.id} FINISHED. Total Time: {total_time:.2f}s", "LOGINFO", current_time_obj)

        finally:
            yield self.ram_container.put(task.data_input_size)

    def battery_drain_process(self):
        while True:
            yield self.env.timeout(1.0)
            is_active = (self.cpu_resource.count > 0)
            consumption = self.power_active if is_active else self.power_idle
            
            self.current_battery -= consumption
            self.total_energy_consumed += consumption
            
            if self.current_battery <= 0:
                self.battery_depleted = True
                self.current_battery = 0
                self.logger.write_Log(f"CRITICAL: Satellite {self.nodeID} BATTERY DEPLETED.", "LOGERROR", get_current_sim_time(self.env.now))
                break

# --- Helper to Init SatelliteMEC ---
def init_SatelliteMEC(_env, _mec_details, _nodeDetails, _timeDetails, _topologyID, _logger):
    t_start = Time()
    t_start.from_str(_timeDetails.starttime)
    t_end = Time()
    t_end.from_str(_timeDetails.endtime)
    
    satellite_args = {
        '_nodeID': _nodeDetails.nodeid,
        '_topologyID': _topologyID,
        '_tleline1': _nodeDetails.tle_1,
        '_tleline2': _nodeDetails.tle_2,
        '_timeDelta': _timeDetails.delta,
        '_timeStamp': t_start,
        '_endtime': t_end,
    }
    return SatelliteMEC(_env, _mec_details, _logger, **satellite_args)

# --- BRAIN 1: BaselineScheduler ---
class BaselineScheduler:
    def __init__(self, satellites, logger):
        self.satellites = satellites
        self.logger = logger
    
    def schedule_task(self, task, current_time):
        valid_sats = [s for s in self.satellites if not s.battery_depleted]
        if not valid_sats: return None
        # Heuristic: Just picks shortest queue, ignores Region and RAM!
        return min(valid_sats, key=lambda s: s.get_queue_length())

# =============================================================================
# [UPDATED] LLMScheduler with Advanced Multi-Constraint Logic
# =============================================================================
API_KEY = os.environ.get("GEMINI_API_KEY")
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class LLMScheduler:
    def __init__(self, satellites, logger):
        self.satellites = satellites
        self.logger = logger
        start_time = get_current_sim_time(0)
        
        if not API_KEY:
            self.logger.write_Log("CRITICAL: GEMINI_API_KEY not found!", "LOGERROR", start_time)
        else:
            self.logger.write_Log("LLMScheduler initialized (Multi-Constraint Mode).", "LOGINFO", start_time)

    def _state_to_prompt(self, task, current_time):
        # --- PROMPT V4: Region + Battery + RAM ---
        prompt = "You are an Autonomous Satellite Scheduler.\n"
        prompt += "Evaluate candidates based on these STRICT rules (in order of priority):\n"
        prompt += "1. REGION: If Task requires a region, Satellite MUST be in that region.\n"
        prompt += "2. RAM: Satellite MUST have enough Free RAM.\n"
        prompt += "3. ENERGY: Satellite MUST NOT have Critical Battery (<10%).\n"
        prompt += "4. SPEED: Choose fastest CPU among valid candidates.\n\n"
        
        prompt += f"--- NEW TASK ---\n"
        prompt += f"ID: {task.id} | Type: {task.type}\n"
        prompt += f"REQUIREMENTS: RAM {task.data_input_size} MB | Region: {task.required_region if task.required_region else 'Any'}\n\n"
        
        prompt += "--- SATELLITE STATUS ---\n"
        for sat in self.satellites:
            ram_free = sat.ram_container.level
            batt_pct = (sat.current_battery / sat.battery_capacity) * 100
            
            prompt += f"SAT {sat.nodeID}:\n"
            prompt += f"  - Location: {sat.current_region}\n"
            prompt += f"  - Battery: {batt_pct:.1f}%"
            
            if batt_pct < 10: prompt += " [CRITICAL!]"
            else: prompt += " [OK]"
            
            prompt += f"\n  - RAM Free: {ram_free:.1f} MB\n"
            prompt += f"  - CPU: {sat.cpu_capacity_mips} MIPS\n\n"
        
        prompt += "INSTRUCTION: Select the best satellite. Explain reasoning.\n"
        prompt += "JSON FORMAT: {\"satellite_id\": 101}"
        
        return prompt

    def _call_gemini_api(self, prompt):
        if not API_KEY: return None
        model_name = "gemini-2.0-flash" 
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
        headers = {'Content-Type': 'application/json'}
        data = { "contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000} }

        try:
            response = requests.post(url, headers=headers, json=data, verify=False)
            if response.status_code == 200:
                try: return response.json()['candidates'][0]['content']['parts'][0]['text']
                except: return None
            elif response.status_code == 429:
                print("DEBUG: Quota 429.")
                return None
            else:
                return None
        except: return None

    def _response_to_action(self, response_text):
        if not response_text: return None
        try:
            match = re.search(r'"satellite_id"\s*:\s*(\d+)', response_text)
            if match:
                chosen_id = int(match.group(1))
                for sat in self.satellites:
                    if sat.nodeID == chosen_id: return sat
            else:
                # Fallback regex
                match = re.search(r'JSON:.*?(\d{3})', response_text, re.DOTALL)
                if match:
                    chosen_id = int(match.group(1))
                    for sat in self.satellites:
                        if sat.nodeID == chosen_id: return sat
        except: pass
        return None

    def schedule_task(self, task, current_time):
        prompt = self._state_to_prompt(task, current_time)
        resp_text = self._call_gemini_api(prompt)
        
        print(f"\n--- AI REASONING (Task {task.id}) ---\n{resp_text}\n-------------------")
        
        decision = self._response_to_action(resp_text)
        if decision: return decision
        
        # Fallback
        valid_sats = [s for s in self.satellites if not s.battery_depleted]
        if not valid_sats: return None
        return min(valid_sats, key=lambda s: s.get_queue_length())

# =============================================================================
# [UPDATED] Scenario Generator: Geo + Battery + RAM
# =============================================================================
def task_generator(env, scheduler, num_tasks, logger):
    '''
    Complex Scenario:
    - Sat 101: BRAZIL, Fast CPU, LOW BATTERY (Risk!)
    - Sat 102: USA, Slow CPU, Full Battery, Huge RAM.
    '''
    for i in range(num_tasks):
        current_time_obj = get_current_sim_time(env.now)
        
        rand_val = random.random()
        
        if rand_val < 0.33:
            # TYPE A: Critical Gov Task (Must be in USA)
            # Challenge: Sat 101 is faster, but is in Brazil. Must pick Sat 102.
            t_type = "GOV_DATA"
            mips = 1000
            data = 500
            deadline = env.now + 20.0
            region = "USA"
            
        elif rand_val < 0.66:
            # TYPE B: IoT Brazil (Must be in Brazil)
            # Challenge: Sat 101 is in Brazil, BUT has Low Battery. 
            # If battery is critical, AI might need to reject or take risk (depending on prompt).
            t_type = "IOT_BRAZIL"
            mips = 200
            data = 10
            deadline = env.now + 10.0
            region = "BRAZIL"
            
        else:
            # TYPE C: Global Image Proc (No Region)
            # Challenge: Heavy RAM. Sat 101 has no RAM. Must pick Sat 102.
            t_type = "GLOBAL_IMG"
            mips = 3000
            data = 1500
            deadline = env.now + 60.0
            region = None # Global
        
        new_task = Task(i, t_type, mips, data, 1.0, deadline, env.now, region)
        logger.write_Log(f"GENERATOR: Created {new_task}", "LOGINFO", current_time_obj)
        
        sat = scheduler.schedule_task(new_task, current_time_obj)
        if sat:
            logger.write_Log(f"Scheduler assigned Task {new_task.id} to Sat {sat.nodeID}", "LOGDEBUG", current_time_obj)
            env.process(sat.process_task(new_task))
        else:
            logger.write_Log(f"Task {new_task.id} NOT SCHEDULED (No Valid Sat).", "LOGWARN", current_time_obj)
        
        yield env.timeout(2.0)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- NTN-MEC Sim: Multi-Constraint (Region/Battery/RAM) ---")
    USE_LLM_BRAIN = True 

    env = simpy.Environment()
    mock_logger = MockLogger(env)
    
    # --- SETUP SATELLITES ---
    
    # Sat 101: The "Fragile Speedster" (Brazil)
    # Fast CPU, but Low RAM and CRITICAL BATTERY START
    sat1_details = SimpleNamespace(
        cpu_capacity_mips=2000.0, 
        ram_capacity_mb=512.0,    
        io_throughput_mbs=600.0,
        battery_capacity_joules=5000.0,
        initial_battery_pct=8.0, # <--- DANGER! Starts at 8%
        region="BRAZIL"          # <--- Location
    )
    
    # Sat 102: The "Heavy Lifter" (USA)
    # Slow CPU, Huge RAM, Full Battery
    sat2_details = SimpleNamespace(
        cpu_capacity_mips=1000.0, 
        ram_capacity_mb=4096.0,   
        io_throughput_mbs=300.0,
        battery_capacity_joules=10000.0,
        initial_battery_pct=100.0,
        region="USA"             # <--- Location
    )

    node_details = json.loads('{"nodeid": 1, "tle_1": "...", "tle_2": "...", "additionalargs": ""}', object_hook=lambda d: SimpleNamespace(**d))
    time_details = json.loads(f'{{"starttime": "{SIM_START_TIME_STR}", "endtime": "2025-11-10 21:30:00", "delta": 1.0}}', object_hook=lambda d: SimpleNamespace(**d))

    satellites = []
    
    node_details.nodeid = 101
    s1 = init_SatelliteMEC(env, sat1_details, node_details, time_details, 1, mock_logger)
    satellites.append(s1)
    
    node_details.nodeid = 102
    s2 = init_SatelliteMEC(env, sat2_details, node_details, time_details, 1, mock_logger)
    satellites.append(s2)

    if USE_LLM_BRAIN:
        print(">>> Using LLM Scheduler")
        scheduler = LLMScheduler(satellites, mock_logger)
    else:
        print(">>> Using Baseline Scheduler")
        scheduler = BaselineScheduler(satellites, mock_logger)

    if 'scheduler' in locals():
        env.process(task_generator(env, scheduler, 15, mock_logger))
        env.run(until=100)
        
        print("\n--- RESULTS ---")
        for s in satellites:
            print(f"Sat {s.nodeID} ({s.current_region}): Completed {s.tasks_completed_count} | Dropped {s.tasks_dropped_count} | Battery End: {(s.current_battery/s.battery_capacity)*100:.1f}%")