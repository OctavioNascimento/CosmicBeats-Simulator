# main_simulation.py
'''
@desc
    Phase 2.5: Logging, Visualization & Safe Mode.
    
    UPDATES:
    - Auto-Logging: Saves metrics to logs/sim_run_<timestamp>.jsonl
    - Safe Mode: Satellites reject tasks if battery < 20% (Professor's Request).
    - Data Sovereignty: Strict Region checks.
'''

import simpy
import random
import os
import re
import json
import requests
import urllib3
import time as pytime
from datetime import datetime
from types import SimpleNamespace

# Desabilita avisos de segurança
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- GLOBAL CONFIG ---
SIM_START_TIME_STR = "2025-11-10 20:30:00"
SIM_DURATION = 20 * 60 
LAMBDA_REQUESTS_PER_MIN = 4.0 
SAFE_MODE_THRESHOLD = 20.0 # % Battery limit (Professor's Rule)

API_KEY = os.environ.get("GEMINI_API_KEY")

# --- LOGGING SETUP (MODO SOBRESCREVER) ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
# Nome fixo para sempre sobrescrever o anterior
LOG_FILE = os.path.join(LOG_DIR, "sim_run_latest.jsonl")

# Limpa o arquivo no início da execução
with open(LOG_FILE, "w") as f:
    f.write("")

def log_event(event_type, time_val, details):
    """Escreve um evento estruturado no arquivo de log JSONL"""
    entry = {
        "timestamp": time_val,
        "type": event_type,
        "details": details
    }
    with open(LOG_FILE, "a") as f: # Usa 'a' para adicionar linha a linha
        f.write(json.dumps(entry) + "\n")
    
    print(f"[{time_val:.2f}] {event_type}: {details}")

# --- MOCK HELPERS ---
class Time:
    def from_str(self, s): pass
    def add_seconds(self, s): pass

def get_current_sim_time(env_now): return env_now

# =============================================================================
# [CORE] Task Class
# =============================================================================
class Task:
    def __init__(self, task_id, origin_gs_id, task_type, mips, data_mb, deadline, created_at):
        self.id = task_id
        self.type = task_type
        self.mips = mips
        self.data = data_mb
        self.deadline = deadline
        self.origin = origin_gs_id
        # Regra de Soberania Simples
        self.region = "BRAZIL" if 3 <= origin_gs_id <= 8 else "USA"

    def to_dict(self):
        return {"id": self.id, "type": self.type, "region": self.region, "data_mb": self.data}

# =============================================================================
# [NODE] SatelliteMEC (With Safe Mode)
# =============================================================================
class SatelliteMEC:
    def __init__(self, env, node_id, name, region, battery_pct, ram_mb, cpu_mips):
        self.env = env
        self.node_id = node_id
        self.name = name
        self.region = region
        
        self.cpu = simpy.Resource(env, capacity=1)
        self.cpu_mips = cpu_mips
        self.ram = simpy.Container(env, capacity=ram_mb, init=ram_mb)
        self.ram_total = ram_mb
        
        self.battery_capacity = 10000.0
        self.battery = battery_pct * 100.0 
        
        self.alive = True
        self.env.process(self.life_cycle())

    def life_cycle(self):
        while self.alive:
            yield self.env.timeout(1.0)
            # Consumo: 1 unidade basal, +5 se processando
            drain = 1.0 + (5.0 if self.cpu.count > 0 else 0)
            self.battery -= drain
            
            # Log de telemetria a cada 10s para não lotar o disco
            if self.env.now % 10 == 0:
                pct = (self.battery / self.battery_capacity) * 100
                log_event("TELEMETRY", self.env.now, {
                    "sat_id": self.node_id, 
                    "battery": pct, 
                    "ram_free": self.ram.level
                })

            if self.battery <= 0:
                self.battery = 0
                self.alive = False
                log_event("CRITICAL_FAILURE", self.env.now, f"{self.name} Battery Depleted.")

    def get_telemetry(self):
        pct = (self.battery / self.battery_capacity) * 100
        return {
            "id": self.node_id,
            "region": self.region,
            "battery_pct": pct,
            "ram_free": self.ram.level,
            "alive": self.alive,
            "safe_mode": pct < SAFE_MODE_THRESHOLD
        }

    def process_task(self, task):
        # 1. Verifica se está vivo
        if not self.alive: return

        # 2. SAFE MODE CHECK (Professor's Rule)
        batt_pct = (self.battery / self.battery_capacity) * 100
        if batt_pct < SAFE_MODE_THRESHOLD:
            log_event("TASK_REJECTED", self.env.now, f"{self.name} in SAFE MODE ({batt_pct:.1f}%). Task {task.id} refused.")
            return

        # 3. RAM Check
        if self.ram.level < task.data:
            log_event("TASK_DROPPED", self.env.now, f"{self.name} OOM for Task {task.id}")
            return
        
        # Aloca RAM
        yield self.ram.get(task.data)
        
        try:
            # Simula processamento
            with self.cpu.request() as req:
                yield req
                duration = task.mips / self.cpu_mips
                yield self.env.timeout(duration)
                
                log_event("TASK_COMPLETED", self.env.now, {
                    "task_id": task.id, 
                    "sat_id": self.node_id, 
                    "duration": duration
                })
        finally:
            yield self.ram.put(task.data)

# =============================================================================
# [BRAIN] CentralBrainGS
# =============================================================================
class CentralBrainGS:
    def __init__(self, env, satellites):
        self.env = env
        self.satellites = satellites
        self.telemetry = {}
        self.env.process(self.heartbeat())

    def heartbeat(self):
        while True:
            for s in self.satellites:
                self.telemetry[s.node_id] = s.get_telemetry()
            yield self.env.timeout(1.0)

    def call_gemini(self, prompt):
        if not API_KEY: return None
        model = "gemini-2.0-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
        headers = {'Content-Type': 'application/json'}
        data = { "contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1} }
        
        try:
            r = requests.post(url, headers=headers, json=data, verify=False)
            if r.status_code == 200: return r.json()['candidates'][0]['content']['parts'][0]['text']
        except: return None
        return None

    def decide(self, task):
        # PROMPT ATUALIZADO COM REGRA DE SAFE MODE
        prompt = "Scheduler Rules:\n"
        prompt += f"1. REGION: Sat Region == Task Region ({task.region}).\n"
        prompt += f"2. SAFETY: Battery > {SAFE_MODE_THRESHOLD}%. Do NOT kill the satellite.\n"
        prompt += f"3. RAM: Free RAM > {task.data} MB.\n"
        
        prompt += f"\nTASK: ID {task.id} | Region {task.region} | RAM {task.data}\n"
        prompt += "SATELLITES:\n"
        
        valid_exists = False
        for sid, data in self.telemetry.items():
            if not data['alive']: continue
            valid_exists = True
            safe_str = "SAFE_MODE" if data['safe_mode'] else "ACTIVE"
            prompt += f"SAT {sid}: {data['region']} | Bat {data['battery_pct']:.1f}% ({safe_str}) | RAM {data['ram_free']}\n"
            
        if not valid_exists: return None
        
        prompt += "\nOutput JSON: {\"satellite_id\": 101} or null"
        
        resp = self.call_gemini(prompt)
        # log_event("LLM_THOUGHT", self.env.now, resp) # Descomente para ver o raciocínio no log
        
        if resp:
            try:
                match = re.search(r'"satellite_id"\s*:\s*(\d+)', resp)
                if match:
                    tid = int(match.group(1))
                    for s in self.satellites:
                        if s.node_id == tid: return s
            except: pass
            
        return None

# =============================================================================
# [SCENARIO] Traffic Generator
# =============================================================================
def traffic_gen(env, brain, stations):
    i = 0
    while True:
        yield env.timeout(random.expovariate(LAMBDA_REQUESTS_PER_MIN / 60.0))
        i += 1
        
        # Cenário Misto
        origin = random.choice(stations)
        r = random.random()
        if r < 0.4:   type, mips, data = "IOT_DATA", 150, 5
        elif r < 0.7: type, mips, data = "GOV_DATA", 500, 200
        else:         type, mips, data = "IMG_PROC", 2000, 1000 # Pesado
        
        # Ajuste: Aumentamos a duração das tarefas pesadas para drenar bateria
        if type == "IMG_PROC": mips = 5000 
        
        task = Task(i, origin['id'], type, mips, data, 60, env.now)
        log_event("NEW_REQUEST", env.now, task.to_dict())
        
        sat = brain.decide(task)
        if sat:
            log_event("ASSIGNED", env.now, f"Task {i} -> {sat.name}")
            env.process(sat.process_task(task))
        else:
            log_event("SCHEDULER_REJECT", env.now, f"Task {i} not scheduled")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print(f"--- SIMULATION START (Logs in {LOG_FILE}) ---")
    env = simpy.Environment()
    
    # Sat 1: Começa com 25% (perto do limite de 20%) para testar o Safe Mode rápido
    s1 = SatelliteMEC(env, 101, "SAT_BRAZIL", "BRAZIL", 25.0, 512, 2000)
    # Sat 2: Tanque cheio
    s2 = SatelliteMEC(env, 102, "SAT_USA", "USA", 100.0, 4096, 1000)
    
    gs_list = [{'id': i} for i in range(3, 15)]
    brain = CentralBrainGS(env, [s1, s2])
    
    env.process(traffic_gen(env, brain, gs_list))
    env.run(until=SIM_DURATION)
    print("--- SIMULATION END ---")