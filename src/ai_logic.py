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
        self.lambda_rate = 4.0 / 60.0 # 4 tarefas por minuto

    def setup_nodes(self, all_nodes):
        print(f">>> [MEC] Setup Nodes called with {len(all_nodes)} nodes.")
        for node in all_nodes:
            # Duck typing: checa se tem iName ou usa str()
            name = getattr(node, 'iName', str(node))
            
            # Verifica se é satélite (nome ou tipo)
            if "Satellite" in name or "SAT" in name:
                nid = getattr(node, 'nodeID', 0)
                is_strong = (nid % 2 == 0)
                
                # Injeta atributos no objeto (Monkey Patching)
                node.mec_ram_total = 4096.0 if is_strong else 512.0
                node.mec_ram_free = node.mec_ram_total
                node.mec_battery = 100.0 if is_strong else 25.0
                node.mec_region = "USA" if is_strong else "BRAZIL"
                
                # Inicializa contadores
                node.mec_tasks_completed = 0
                node.mec_tasks_dropped = 0
                
                self.mec_satellites.append(node)
                print(f">>> [MEC] Upgraded {name} (ID {nid}) -> {node.mec_region}, Bat: {node.mec_battery}%")

    def step(self, current_time_sec):
        # 1. Atualiza Bateria (Consumo Basal)
        for s in self.mec_satellites:
            if getattr(s, 'mec_battery', 0) > 0:
                s.mec_battery = max(0, s.mec_battery - 0.05)

        # 2. Verifica se chegou o momento da próxima tarefa (Poisson)
        if self.next_task_time < 0:
            self.next_task_time = current_time_sec + random.expovariate(self.lambda_rate)

        if current_time_sec >= self.next_task_time:
            self.task_id += 1
            
            # Sorteio de tipo de tarefa e região
            rand_val = random.random()
            if rand_val < 0.4:
                region = "BRAZIL"
            elif rand_val < 0.8:
                region = "USA"
            else:
                region = "GLOBAL" 
            
            task = {"id": self.task_id, "region": region, "ram": 500}
            
            # Coleta Estado da Frota
            fleet = []
            for s in self.mec_satellites:
                fleet.append({
                    "id": getattr(s, 'nodeID', 0),
                    "region": getattr(s, 'mec_region', 'UNK'),
                    "battery": getattr(s, 'mec_battery', 0),
                    "ram_free": getattr(s, 'mec_ram_free', 0)
                })
            
            print(f"\n[MEC T+{current_time_sec:.1f}] New Task {self.task_id} ({region})")
            
            # CHAMA A IA
            decision_id = self.brain.decide(task, fleet)
            
            if decision_id:
                print(f"   ---> [AI] Assign to SAT {decision_id}")
                
                # Atualiza estatísticas e recursos do satélite escolhido
                found = False
                for s in self.mec_satellites:
                    if getattr(s, 'nodeID') == decision_id:
                        s.mec_tasks_completed += 1
                        # Consome um pouco de bateria extra pelo processamento
                        s.mec_battery = max(0, s.mec_battery - 1.0)
                        found = True
                        break
                if not found:
                    print(f"   ---> [ERROR] AI selected ID {decision_id} but satellite not found!")

            else:
                print(f"   ---> [AI] NO ROUTE")
            
            # Agenda a próxima tarefa
            self.next_task_time = current_time_sec + random.expovariate(self.lambda_rate)

    def print_stats(self):
        print("\n" + "="*40)
        print("       MEC ORCHESTRATOR REPORT")
        print("="*40)
        total_completed = 0
        
        for s in self.mec_satellites:
            completed = getattr(s, 'mec_tasks_completed', 0)
            battery = getattr(s, 'mec_battery', 0)
            region = getattr(s, 'mec_region', 'UNK')
            status = "DEAD" if battery <= 0 else "ALIVE"
            
            total_completed += completed
            
            print(f"SAT {getattr(s, 'nodeID')} ({region}) [{status}]")
            print(f"  - Battery: {battery:.1f}%")
            print(f"  - Tasks Completed: {completed}")
            print("-" * 20)
            
        print(f"TOTAL TASKS PROCESSED: {self.task_id}")
        print(f"TOTAL COMPLETED: {total_completed}")
        print(f"TOTAL REJECTED (NO ROUTE): {self.task_id - total_completed}")
        print("="*40 + "\n")