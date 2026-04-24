# src/ai_logic.py
import csv
import json
import math
import os
import random
import time

from src.schedulers.llm_scheduler import LLMScheduler
from src.schedulers.baseline_scheduler import BaselineScheduler
from src.schedulers.slm_scheduler import SLMScheduler, NPU_LATENCY_MS as SLM_NPU_LATENCY_MS
from src.schedulers.drl_scheduler import DRLScheduler

# Custo energético por decisão — parâmetros calibrados por tipo de hardware
JOULES_PER_DECISION = {
    "BASELINE": 0.001,   # CPU if/else, ~1μs @ 1W
    "DRL":      0.005,   # Q-table lookup + update, ~2ms @ 2W
    "SLM":      0.100,   # NPU inference, 50ms @ 2W
    "LLM":      5.000,   # LEO→GS TX, ~1s @ 5W
}

# Atraso de propagação LEO↔GS round-trip (550km orbit: 2×550000/3e8 ≈ 3.67ms)
LLM_PROPAGATION_MS = 3.67

# Duração de processamento de uma tarefa — após este período a RAM é liberada
TASK_DURATION_S = 60.0

# Modelo de qualidade de link por passe orbital (senoide, período = 10 min)
LINK_QUALITY_PERIOD_S = 600.0   # LEO simplificado: um passe a cada 10 min na janela de simulação
LINK_QUALITY_MIN      = 20.0    # abaixo deste valor o satélite não recebe tarefas


class MECOrchestrator:
    def __init__(self, brain=None, anomaly_rate=0.10, *args, **kwargs):
        self._engine = os.environ.get("MEC_ENGINE", "BASELINE")
        print(f">>> [MECOrchestrator] Inicializando com o motor: {self._engine}")

        if self._engine == "LLM":
            self.brain = LLMScheduler()
        elif self._engine == "SLM":
            self.brain = SLMScheduler()
        elif self._engine == "DRL":
            self.brain = DRLScheduler()
        elif self._engine == "BASELINE":
            self.brain = BaselineScheduler()
        else:
            print(f">>> [MECOrchestrator] Motor '{self._engine}' desconhecido, usando BASELINE.")
            self.brain = BaselineScheduler()
            self._engine = "BASELINE"

        self.anomaly_rate  = anomaly_rate
        self.task_id       = 0
        self.mec_satellites = []
        self.mec_tasks_dropped = 0
        self.next_task_time = -1.0
        self.lambda_rate   = 4.0 / 60.0   # 4 tarefas/minuto

        self.task_log    = []   # uma entrada por tarefa finalizada
        self.active_tasks = []  # (completion_time, sat_id, ram_used) — para liberar RAM
        self._last_step_time = 0.0

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def setup_nodes(self, all_nodes):
        print(f">>> [MEC] Setup Nodes called with {len(all_nodes)} nodes.")
        region_map = {0: "USA", 1: "BRAZIL", 2: "EUROPE"}

        for node in all_nodes:
            name = getattr(node, 'iName', str(node))
            if "Satellite" in name or "SAT" in name:
                nid = getattr(node, 'nodeID', 0)
                region = region_map[nid % 3]

                node.mec_ram_total  = 4096.0
                node.mec_ram_free   = node.mec_ram_total
                node.mec_region     = region
                node.mec_tasks_completed = 0
                node.mec_tasks_dropped   = 0

                # Fase do passe orbital — satélites igualmente espaçados em fase
                sat_idx = len(self.mec_satellites)
                node.mec_link_phase = sat_idx * (LINK_QUALITY_PERIOD_S / 3.0)

                self.mec_satellites.append(node)
                lq0 = self._link_quality(node, 0.0)
                print(f">>> [MEC] Upgraded {name} (ID {nid}) -> {region} | "
                      f"link_quality(t=0)={lq0:.1f}% | phase={node.mec_link_phase:.0f}s")

    # ------------------------------------------------------------------ #
    # Modelo de qualidade de link                                          #
    # ------------------------------------------------------------------ #

    def _link_quality(self, node, t):
        """
        Qualidade do link LEO↔GS simulada como senoide (representa ângulo de elevação orbital).
        Resultado: [0%, 100%]. Período = LINK_QUALITY_PERIOD_S. Fase distinta por satélite.
        """
        phase = getattr(node, 'mec_link_phase', 0.0)
        return 50.0 + 50.0 * math.sin(2 * math.pi * (t + phase) / LINK_QUALITY_PERIOD_S)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_fleet(self, current_time_sec):
        fleet = []
        for s in self.mec_satellites:
            lq = self._link_quality(s, current_time_sec)
            fleet.append({
                "id":           getattr(s, 'nodeID', 0),
                "region":       getattr(s, 'mec_region', 'UNK'),
                "link_quality": round(lq, 1),
                "ram_free":     getattr(s, 'mec_ram_free', 0),
            })
        return fleet

    def _release_completed_tasks(self, current_time_sec):
        """Libera RAM das tarefas que concluíram processamento (após TASK_DURATION_S)."""
        still_active = []
        for comp_t, sid, ram in self.active_tasks:
            if comp_t <= current_time_sec:
                for s in self.mec_satellites:
                    if getattr(s, 'nodeID') == sid:
                        s.mec_ram_free = min(s.mec_ram_total, s.mec_ram_free + ram)
                        break
            else:
                still_active.append((comp_t, sid, ram))
        self.active_tasks = still_active

    @staticmethod
    def _semantic_compliant(task, decision_id, fleet):
        """Verifica se a decisão de roteamento respeita a restrição semântica da tarefa."""
        anomaly = task.get('semantic_anomaly')
        if not anomaly:
            return True

        if anomaly == "restricao_gdpr_europa":
            if decision_id is None:
                return False
            sat = next((s for s in fleet if s['id'] == decision_id), {})
            return sat.get('region') == 'EUROPE'

        if anomaly == "restricao_soberania_brasil":
            if decision_id is None:
                return False
            sat = next((s for s in fleet if s['id'] == decision_id), {})
            return sat.get('region') == 'BRAZIL'

        if anomaly == "falha_hardware_camera_esq":
            return decision_id is None   # correto = drop

        return True   # anomalia desconhecida — sem critério de verificação

    def _apply_decision(self, task, decision_id, current_time_sec,
                        latency_ms, joules_cost, fleet):
        anomalia    = task.get('semantic_anomaly', '')
        anomalia_str = f" [anomalia={anomalia}]" if anomalia else ""
        compliant   = self._semantic_compliant(task, decision_id, fleet)

        if decision_id is not None:
            print(f"   ---> [{self._engine}] Assign Task {task['id']} to SAT {decision_id}"
                  f"{anomalia_str} | {latency_ms:.0f}ms | {joules_cost:.3f}J"
                  f"{' ✗semantic' if not compliant else ''}")
            for s in self.mec_satellites:
                if getattr(s, 'nodeID') == decision_id:
                    s.mec_tasks_completed += 1
                    s.mec_ram_free = max(0, s.mec_ram_free - task.get('ram', 0))
                    break
            self.active_tasks.append(
                (current_time_sec + TASK_DURATION_S, decision_id, task.get('ram', 0))
            )
        else:
            print(f"   ---> [{self._engine}] NO ROUTE Task {task['id']}"
                  f"{anomalia_str} | {latency_ms:.0f}ms | {joules_cost:.3f}J")
            self.mec_tasks_dropped += 1

        self.task_log.append({
            "task_id":            task['id'],
            "region":             task.get('region', ''),
            "anomaly":            anomalia,
            "arrival_time_s":     task.get('arrival_time', current_time_sec),
            "decision_time_s":    current_time_sec,
            "latency_ms":         round(latency_ms, 3),
            "joules_cost":        round(joules_cost, 4),
            "decision_sat_id":    decision_id if decision_id is not None else "",
            "success":            1 if decision_id is not None else 0,
            "semantic_compliant": 1 if compliant else 0,
            "engine":             self._engine,
        })

    # ------------------------------------------------------------------ #
    # Main loop step                                                       #
    # ------------------------------------------------------------------ #

    def step(self, current_time_sec):
        self._last_step_time = current_time_sec

        # 1. Libera RAM de tarefas concluídas
        self._release_completed_tasks(current_time_sec)

        # 2. Processa inferências SLM concluídas neste tick
        if hasattr(self.brain, 'check_completed_inferences'):
            for task, decision_id in self.brain.check_completed_inferences(current_time_sec):
                latency_ms  = SLM_NPU_LATENCY_MS   # latência NPU simulada (não wall-clock da API)
                joules_cost = JOULES_PER_DECISION.get("SLM", 0.1)
                fleet       = task.get('slm_fleet_snapshot', [])
                self._apply_decision(task, decision_id, current_time_sec,
                                     latency_ms, joules_cost, fleet)

        # 3. Poisson: inicializa ou verifica chegada de nova tarefa
        if self.next_task_time < 0:
            self.next_task_time = current_time_sec + random.expovariate(self.lambda_rate)

        if current_time_sec < self.next_task_time:
            return

        # 4. Gera nova tarefa
        self.task_id += 1
        rand_val = random.random()
        if rand_val < 0.33:
            region = "BRAZIL"
        elif rand_val < 0.66:
            region = "USA"
        else:
            region = "EUROPE"

        task = {"id": self.task_id, "region": region, "ram": 500,
                "arrival_time": current_time_sec}

        if random.random() < self.anomaly_rate:
            task["semantic_anomaly"] = random.choice([
                "restricao_gdpr_europa",
                "restricao_soberania_brasil",
                "falha_hardware_camera_esq",
            ])

        fleet = self._build_fleet(current_time_sec)
        anomalia_str = f" [anomalia={task['semantic_anomaly']}]" if task.get('semantic_anomaly') else ""
        # Mostra link_quality da região da tarefa para diagnóstico
        region_lq = next((s['link_quality'] for s in fleet if s['region'] == region), '?')
        print(f"\n[MEC T+{current_time_sec:.1f}] New Task {self.task_id} ({region} LQ={region_lq}%){anomalia_str}")

        if hasattr(self.brain, 'receive_task'):
            # SLM: async
            self.brain.receive_task(task, current_time_sec, fleet)
        else:
            # LLM / BASELINE / DRL: sync
            t_wall      = time.perf_counter()
            decision_id = self.brain.decide(task, fleet, min_link_quality=LINK_QUALITY_MIN)
            wall_ms     = (time.perf_counter() - t_wall) * 1000

            latency_ms  = wall_ms + LLM_PROPAGATION_MS if self._engine == "LLM" else wall_ms
            joules_cost = JOULES_PER_DECISION.get(self._engine, 0.001)
            self._apply_decision(task, decision_id, current_time_sec,
                                 latency_ms, joules_cost, fleet)

        self.next_task_time = current_time_sec + random.expovariate(self.lambda_rate)

    # ------------------------------------------------------------------ #
    # Relatórios                                                           #
    # ------------------------------------------------------------------ #

    def print_stats(self):
        print("\n" + "=" * 44)
        print("         MEC ORCHESTRATOR REPORT")
        print("=" * 44)
        total_completed = 0

        for s in self.mec_satellites:
            completed = getattr(s, 'mec_tasks_completed', 0)
            region    = getattr(s, 'mec_region', 'UNK')
            lq        = self._link_quality(s, self._last_step_time)
            total_completed += completed

            print(f"SAT {getattr(s, 'nodeID')} ({region})")
            print(f"  Link Quality:    {lq:.1f}%")
            print(f"  RAM Free:        {getattr(s, 'mec_ram_free', 0):.0f} MB")
            print(f"  Tasks Completed: {completed}")
            print("-" * 22)

        n = len(self.task_log)
        print(f"\nTOTAL TASKS GENERATED:    {self.task_id}")
        print(f"TOTAL COMPLETED:          {total_completed}")
        print(f"TOTAL REJECTED:           {self.mec_tasks_dropped}")
        if n > 0:
            success_rate  = total_completed / self.task_id * 100
            avg_lat       = sum(r['latency_ms']  for r in self.task_log) / n
            total_joules  = sum(r['joules_cost'] for r in self.task_log)
            avg_joules    = total_joules / n
            compliant_n   = sum(r['semantic_compliant'] for r in self.task_log)
            anomaly_tasks = [r for r in self.task_log if r['anomaly']]

            print(f"SUCCESS RATE:             {success_rate:.1f}%")
            print(f"AVG LATENCY:              {avg_lat:.1f} ms")
            print(f"TOTAL ENERGY:             {total_joules:.3f} J")
            print(f"JOULES/DECISION:          {avg_joules:.4f} J")
            print(f"SEMANTIC COMPLIANCE:      {compliant_n}/{n} ({compliant_n/n*100:.1f}%)")
            if anomaly_tasks:
                anom_ok = sum(r['semantic_compliant'] for r in anomaly_tasks)
                print(f"ANOMALY COMPLIANCE:       {anom_ok}/{len(anomaly_tasks)}")

        if hasattr(self.brain, 'print_qtable'):
            self.brain.print_qtable()

        print("=" * 44 + "\n")

    def save_metrics(self):
        os.makedirs("logs", exist_ok=True)
        engine = self._engine

        csv_path   = f"logs/mec_metrics_{engine}.csv"
        fieldnames = [
            "task_id", "region", "anomaly",
            "arrival_time_s", "decision_time_s", "latency_ms",
            "joules_cost", "decision_sat_id",
            "success", "semantic_compliant", "engine",
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.task_log)

        n       = len(self.task_log)
        summary = {"engine": engine, "total_tasks": n}
        if n > 0:
            completed      = sum(r['success']            for r in self.task_log)
            anomaly_tasks  = [r for r in self.task_log   if r['anomaly']]
            compliant      = sum(r['semantic_compliant'] for r in self.task_log)
            anom_compliant = sum(r['semantic_compliant'] for r in anomaly_tasks)

            summary.update({
                "success_count":            completed,
                "success_rate":             round(completed / n, 4),
                "drop_count":               self.mec_tasks_dropped,
                "drop_rate":                round(self.mec_tasks_dropped / n, 4),
                "avg_latency_ms":           round(sum(r['latency_ms']  for r in self.task_log) / n, 3),
                "total_joules":             round(sum(r['joules_cost'] for r in self.task_log), 4),
                "joules_per_decision":      round(JOULES_PER_DECISION.get(engine, 0.001), 4),
                "anomaly_task_count":       len(anomaly_tasks),
                "semantic_compliance_rate": round(compliant / n, 4),
                "anomaly_compliance_rate":  round(anom_compliant / len(anomaly_tasks), 4) if anomaly_tasks else None,
            })

        json_path = f"logs/mec_summary_{engine}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"[MEC] Metricas salvas: {csv_path} | {json_path}")
