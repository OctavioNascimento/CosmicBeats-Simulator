# src/ai_logic.py
import random
import re
from src.ai_adapter import GeminiBrain

class MECOrchestrator:
    def __init__(self, brain):
        self.brain = brain
        self.task_id = 0
        self.mec_satellites = []
        # Inicia com um tempo negativo para disparar logo
        self.next_task_time = -1.0 
        self.lambda_rate = 4.0 / 60.0

    def setup_nodes(self, all_nodes):
        print(f">>> [MEC] Setup Nodes called with {len(all_nodes)} nodes.")
        for node in all_nodes:
            # Duck typing: checa se tem iName
            name = getattr(node, 'iName', str(node))
            if "Satellite" in name or "SAT" in name:
                nid = getattr(node, 'nodeID', 0)
                is_strong = (nid % 2 == 0)
                
                # Injeta atributos no objeto
                node.mec_ram_total = 4096.0 if is_strong else 512.0
                node.mec_ram_free = node.mec_ram_total
                node.mec_battery = 100.0 if is_strong else 25.0
                node.mec_region = "USA" if is_strong else "BRAZIL"
                
                self.mec_satellites.append(node)
                print(f">>> [MEC] Upgraded {name} (ID {nid}) -> {node.mec_region}")

    def step(self, current_time_sec):
        # 1. Atualiza Bateria
        for s in self.mec_satellites:
            if getattr(s, 'mec_battery', 0) > 0:
                s.mec_battery = max(0, s.mec_battery - 0.05)

        # 2. Verifica Tarefa (Poisson)
        # Se for a primeira vez ou passou do tempo
        if self.next_task_time < 0:
            self.next_task_time = current_time_sec + random.expovariate(self.lambda_rate)

        if current_time_sec >= self.next_task_time:
            self.task_id += 1
            region = "BRAZIL" if random.random() < 0.5 else "USA"
            task = {"id": self.task_id, "region": region, "ram": 500}
            
            # Coleta Estado
            fleet = []
            for s in self.mec_satellites:
                fleet.append({
                    "id": getattr(s, 'nodeID', 0),
                    "region": getattr(s, 'mec_region', 'UNK'),
                    "battery": getattr(s, 'mec_battery', 0),
                    "ram_free": getattr(s, 'mec_ram_free', 0)
                })
            
            print(f"\n[MEC T+{current_time_sec:.1f}] New Task {self.task_id} ({region})")
            
            decision = self.brain.decide(task, fleet)
            if decision:
                print(f"   ---> [AI] Assign to SAT {decision}")
            else:
                print(f"   ---> [AI] NO ROUTE")
            
            # Agenda próxima
            self.next_task_time = current_time_sec + random.expovariate(self.lambda_rate)