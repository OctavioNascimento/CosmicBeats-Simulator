# main_simulation.py
'''
@desc
    Main simulation script for NTN-MEC scheduling with Real-World constraints.
    
    CORRECTIONS IN THIS VERSION:
    1. Fixed 'Time' object error by calculating time from start string + add_seconds.
    2. Fixed MockLogger abstract class instantiation error.
    3. Integrated Gemini API logic.
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
# Define simulation start time globally so we can calculate current time anywhere
SIM_START_TIME_STR = "2025-11-10 20:30:00"

# --- HELPER: Get Current Time ---
def get_current_sim_time(seconds_elapsed):
    '''
    Creates a Time object representing Start Time + Seconds Elapsed.
    Fixes the "AttributeError: 'Time' object has no attribute 'from_seconds'"
    '''
    t = Time()
    t.from_str(SIM_START_TIME_STR) # Initialize with start time
    t.add_seconds(seconds_elapsed) # Add SimPy time
    return t

# --- Mock Logger ---
class MockLogger(ILogger):
    def __init__(self, sim_env):
        self.env = sim_env
        self.__logTypeLevel = {} # Fix for abstract property
    
    @property
    def logTypeLevel(self):
        return self.__logTypeLevel

    def write_Log(self, msg, log_type, timestamp):
        time_str = f"{self.env.now:.2f}"
        if isinstance(timestamp, Time):
            time_str = timestamp.to_str()
        
        # Simple print for debug
        print(f"[Time: {time_str}] {log_type}: {msg}")

# =============================================================================
# [ITEM 2] REAL CHALLENGES: Updated Task Class
# =============================================================================
class Task:
    '''
    @desc
        Represents a computational job with real-world constraints.
    '''
    def __init__(self, task_id: int, task_type: str, mips: float, 
                 data_in_mb: float, data_out_mb: float, deadline: float, creation_time: float):
        self.id = task_id
        self.type = task_type          # 'IOT_ALERT' or 'IMAGE_PROC'
        self.mips_required = mips      # CPU Load
        self.data_input_size = data_in_mb   # RAM usage & I/O time
        self.data_output_size = data_out_mb # Downlink time (simplified)
        self.deadline = deadline       # QoS constraint
        self.creation_time = creation_time

    def is_missed_deadline(self, current_time: float) -> bool:
        if self.deadline is None: return False
        return current_time > self.deadline

    def __str__(self):
        return (f"Task(id={self.id}, Type={self.type}, MIPS={self.mips_required}, "
                f"Data={self.data_input_size}MB, Deadline={self.deadline:.1f}s)")

# src/nodes/satellite_mec.py

import simpy
from src.nodes.satellitebasic import SatelliteBasic
from src.simlogging.ilogger import ELogType

# =============================================================================
# [ITEM 1] MEC ATTRIBUTES: Updated SatelliteMEC Class (CORREÇÃO FINAL)
# =============================================================================
class SatelliteMEC(SatelliteBasic):
    '''
    A Satellite node with CPU, RAM, I/O limits, and Battery consumption.
    '''
    def __init__(self, _env: simpy.Environment, _mec_details, _logger, **kwargs) -> None:
        
        # CORREÇÃO: Removemos _additionalArgs da chamada nomeada.
        # Como *_additionalArgs em SatelliteBasic captura "restos", 
        # não podemos chamá-lo por nome.
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

        # --- COMPUTATION (Processing) ---
        self.cpu_capacity_mips = float(_mec_details.cpu_capacity_mips)
        self.cpu_resource = simpy.Resource(self.env, capacity=1)

        # --- MEMORY (RAM/Storage) ---
        self.ram_capacity_mb = float(getattr(_mec_details, 'ram_capacity_mb', 4096.0))
        self.ram_container = simpy.Container(self.env, capacity=self.ram_capacity_mb, init=self.ram_capacity_mb)

        # --- I/O (Throughput) ---
        self.io_throughput_mbs = float(getattr(_mec_details, 'io_throughput_mbs', 500.0))

        # --- POWER (Energy) ---
        self.battery_capacity = float(getattr(_mec_details, 'battery_capacity_joules', 10000.0))
        self.current_battery = self.battery_capacity
        self.power_idle = 5.0
        self.power_active = 20.0
        self.total_energy_consumed = 0.0
        self.battery_depleted = False

        self.tasks_completed_count = 0
        self.tasks_dropped_count = 0

        self.env.process(self.battery_drain_process())

    # --- Métodos de processamento (MEC) ---
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

        if self.battery_depleted:
            self.logger.write_Log(f"Task {task.id} DROPPED. Sat {self.nodeID} dead.", "LOGWARN", current_time_obj)
            self.tasks_dropped_count += 1
            return

        if self.ram_container.level < task.data_input_size:
            self.logger.write_Log(f"Task {task.id} DROPPED. Sat {self.nodeID} OOM (Req: {task.data_input_size}MB, Free: {self.ram_container.level}MB).", "LOGWARN", current_time_obj)
            self.tasks_dropped_count += 1
            return

        yield self.ram_container.get(task.data_input_size)
        
        try:
            current_time_obj = get_current_sim_time(self.env.now)
            self.logger.write_Log(f"Task {task.id} accepted. Loading data...", "LOGINFO", current_time_obj)

            io_time = task.data_input_size / self.io_throughput_mbs
            yield self.env.timeout(io_time)

            with self.cpu_resource.request() as req:
                yield req
                current_time_obj = get_current_sim_time(self.env.now)

                if task.is_missed_deadline(self.env.now):
                     self.logger.write_Log(f"Task {task.id} missed deadline in queue! Processing anyway.", "LOGWARN", current_time_obj)

                self.logger.write_Log(f"Task {task.id} processing...", "LOGINFO", current_time_obj)
                processing_time = task.mips_required / self.cpu_capacity_mips
                yield self.env.timeout(processing_time)
                
                self.tasks_completed_count += 1
                total_time = self.env.now - task.creation_time
                current_time_obj = get_current_sim_time(self.env.now)
                self.logger.write_Log(f"Task {task.id} ({task.type}) FINISHED. Total Time: {total_time:.2f}s", "LOGINFO", current_time_obj)

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

# --- Helper to Init SatelliteMEC (CORREÇÃO FINAL) ---
def init_SatelliteMEC(_env, _mec_details, _nodeDetails, _timeDetails, _topologyID, _logger):
    
    t_start = Time()
    t_start.from_str(_timeDetails.starttime)
    
    t_end = Time()
    t_end.from_str(_timeDetails.endtime)
    
    # Dicionário de argumentos corrigido
    satellite_args = {
        '_nodeID': _nodeDetails.nodeid,
        '_topologyID': _topologyID,
        '_tleline1': _nodeDetails.tle_1,
        '_tleline2': _nodeDetails.tle_2,
        '_timeDelta': _timeDetails.delta,
        '_timeStamp': t_start,
        '_endtime': t_end,
        # Removemos _additionalArgs daqui também para limpar
    }
    
    return SatelliteMEC(_env, _mec_details, _logger, **satellite_args)

# --- BRAIN 1: BaselineScheduler ---
class BaselineScheduler:
    def __init__(self, satellites, logger):
        self.satellites = satellites
        self.logger = logger
    
    def schedule_task(self, task, current_time):
        # Naive logic: Shortest CPU Queue
        # Does NOT consider RAM or Battery!
        alive_sats = [s for s in self.satellites if not s.battery_depleted]
        if not alive_sats: return None
        return min(alive_sats, key=lambda s: s.get_queue_length())

# --- BRAIN 2: LLMScheduler (Versão "Anti-Bloqueio" / REST API - CORRIGIDA) ---
API_KEY = os.environ.get("GEMINI_API_KEY")

# Import extra necessário para essa versão
import requests
import urllib3
# Desabilita o aviso chato de "InsecureRequestWarning" no terminal
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class LLMScheduler:
    def __init__(self, satellites, logger):
        self.satellites = satellites
        self.logger = logger
        
        # Correção do Tempo: Usamos get_current_sim_time(0) em vez de Time() vazio
        start_time = get_current_sim_time(0) 

        if not API_KEY:
            self.logger.write_Log("CRITICAL: GEMINI_API_KEY not found!", ELogType.LOGERROR, start_time)
        else:
            self.logger.write_Log("LLMScheduler initialized (REST Mode - SSL Verification OFF).", ELogType.LOGINFO, start_time)

    def _state_to_prompt(self, task, current_time):
        prompt = "You are the Master Scheduler for a Satellite Network.\n"
        prompt += "CRITICAL RULE: DO NOT assign tasks to satellites that do not have enough FREE RAM. The task will crash immediately.\n\n"
        
        prompt += f"--- NEW TASK ---\n"
        prompt += f"ID: {task.id} | Type: {task.type}\n"
        prompt += f"REQUIRED RAM: {task.data_input_size} MB\n"
        prompt += f"REQUIRED CPU: {task.mips_required} MIPS\n\n"
        
        prompt += "--- SATELLITE STATUS ---\n"
        for sat in self.satellites:
            ram_free = sat.ram_container.level
            cpu_load = sat.get_current_load_percentage()
            
            prompt += f"SAT {sat.nodeID}:\n"
            prompt += f"  - RAM FREE: {ram_free:.1f} MB"
            
            # Helper logic for the LLM
            if ram_free < task.data_input_size:
                 prompt += f" (NOT ENOUGH! Need {task.data_input_size - ram_free:.1f} more)"
            else:
                 prompt += " (OK)"
            
            prompt += f"\n  - CPU Speed: {sat.cpu_capacity_mips} MIPS | Load: {cpu_load:.0f}%\n\n"
        
        prompt += "STEP-BY-STEP REASONING:\n"
        prompt += "1. List satellites with enough RAM.\n"
        prompt += "2. Eliminate satellites with critical battery.\n"
        prompt += "3. Pick the fastest one among the survivors.\n"
        prompt += "4. Final Answer JSON.\n\n"
        prompt += "RESPONSE FORMAT:\n"
        prompt += "{\"reason\": \"Sat 101 has low RAM, Sat 102 is valid\", \"satellite_id\": 102}"
        
        return prompt

    def _call_gemini_api(self, prompt):
        if not API_KEY: return None

        # URL direta da API REST do Gemini
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
        
        headers = {'Content-Type': 'application/json'}
        
        # Corpo da requisição
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 100
            }
        }

        try:
            # O SEGREDO ESTÁ AQUI: verify=False ignora o certificado do Netskope
            response = requests.post(url, headers=headers, json=data, verify=False)
            
            if response.status_code == 200:
                # O JSON de resposta da REST API é um pouco diferente da lib python
                result = response.json()
                try:
                    return result['candidates'][0]['content']['parts'][0]['text']
                except KeyError:
                    return None
            else:
                # Se der erro (ex: 400, 500), mostramos o motivo
                # print(f"API Error {response.status_code}: {response.text}") 
                return None
                
        except Exception as e:
            # print(f"Request failed: {e}")
            return None

    def _response_to_action(self, response_text):
        if not response_text: return None
        try:
            match = re.search(r'[:"]?\s*(\d{3})\s*["}]?', response_text) 
            if match:
                chosen_id = int(match.group(1))
                for sat in self.satellites:
                    if sat.nodeID == chosen_id: return sat
        except: pass
        return None

    def schedule_task(self, task, current_time):
        # 1. Tenta usar o Gemini via REST
        prompt = self._state_to_prompt(task, current_time)
        resp_text = self._call_gemini_api(prompt)
        decision = self._response_to_action(resp_text)
        
        if decision: return decision
        
        # 2. Fallback
        valid_sats = [s for s in self.satellites if not s.battery_depleted]
        if not valid_sats: return None
        return min(valid_sats, key=lambda s: s.get_queue_length())
    
# =============================================================================
# [ITEM 3] LIST OF TESTS: Updated Task Generator
# =============================================================================
def task_generator(env, scheduler, num_tasks, logger):
    '''
    Simulates "Scenario B: Heterogeneity Test"
    '''
    for i in range(num_tasks):
        current_time_obj = get_current_sim_time(env.now) # USANDO O HELPER CORRIGIDO
        
        # Randomly choose task type
        if random.random() < 0.3:
            t_type = "IOT_ALERT"
            mips = random.randint(100, 300)
            data = random.randint(1, 5) # MB
            deadline = env.now + 5.0    
        else:
            t_type = "IMAGE_PROC"
            mips = random.randint(1500, 3000)
            data = random.randint(500, 1500) 
            deadline = env.now + 60.0        
        
        new_task = Task(i, t_type, mips, data, 1.0, deadline, env.now)
        logger.write_Log(f"GENERATOR: Created {new_task}", "LOGINFO", current_time_obj)
        
        sat = scheduler.schedule_task(new_task, current_time_obj)
        if sat:
            logger.write_Log(f"Scheduler assigned Task {new_task.id} to Satellite {sat.nodeID}", "LOGDEBUG", current_time_obj)
            env.process(sat.process_task(new_task))
        else:
            logger.write_Log(f"Task {new_task.id} could not be scheduled.", "LOGWARN", current_time_obj)
        
        yield env.timeout(random.uniform(1.0, 3.0))

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- NTN-MEC Sim: Heterogeneity Test Scenario ---")
    
    # --- CONFIG ---
    USE_LLM_BRAIN = True  # <--- Change this to False to test baseline
    # --------------

    env = simpy.Environment()
    mock_logger = MockLogger(env)
    
    # Sat 1: Powerful CPU, Low RAM
    sat1_details = SimpleNamespace(
        cpu_capacity_mips=2000.0, 
        ram_capacity_mb=512.0,    
        io_throughput_mbs=600.0,
        battery_capacity_joules=5000.0 
    )
    
    # Sat 2: High RAM, Average CPU
    sat2_details = SimpleNamespace(
        cpu_capacity_mips=1000.0, 
        ram_capacity_mb=4096.0,   
        io_throughput_mbs=300.0,
        battery_capacity_joules=10000.0
    )

    node_details = json.loads('{"nodeid": 1, "tle_1": "1 50985U...", "tle_2": "2 50985...", "additionalargs": ""}', object_hook=lambda d: SimpleNamespace(**d))
    time_details = json.loads(f'{{"starttime": "{SIM_START_TIME_STR}", "endtime": "2025-11-10 21:30:00", "delta": 1.0}}', object_hook=lambda d: SimpleNamespace(**d))

    satellites = []
    
    node_details.nodeid = 101
    s1 = init_SatelliteMEC(env, sat1_details, node_details, time_details, 1, mock_logger)
    satellites.append(s1)
    
    node_details.nodeid = 102
    s2 = init_SatelliteMEC(env, sat2_details, node_details, time_details, 1, mock_logger)
    satellites.append(s2)

    if USE_LLM_BRAIN:
        print(">>> Using LLM Scheduler (Gemini)")
        scheduler = LLMScheduler(satellites, mock_logger)
    else:
        print(">>> Using Baseline Scheduler (Shortest Queue)")
        scheduler = BaselineScheduler(satellites, mock_logger)

    if 'scheduler' in locals():
        env.process(task_generator(env, scheduler, 15, mock_logger))
        env.run(until=100)
        
        print("\n--- RESULTS ---")
        for s in satellites:
            print(f"Sat {s.nodeID}: Completed {s.tasks_completed_count} | Dropped {s.tasks_dropped_count} | Energy Used {s.total_energy_consumed:.1f}J")

USE_LLM_BRAIN = False