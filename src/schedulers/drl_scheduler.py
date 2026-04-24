import collections
import random


class DRLScheduler:
    """
    Q-learning tabular online: agente de RL que aprende durante a simulação.

    Estado: (region_idx, link_quality_bin, ram_ok, has_anomaly) — 36 estados possíveis.
    Ações: PREFER_LINK | PREFER_RAM | PREFER_BALANCED | DROP (0–3).

    Invariante científico: DRL conhece que há anomalia (has_anomaly=1) mas
    não conhece o tipo (GDPR, soberania, hardware). Isso é intencional —
    justifica menor semantic_compliance vs SLM/LLM nos resultados do TCC.
    """

    ACTIONS   = [0, 1, 2, 3]   # PREFER_LINK, PREFER_RAM, PREFER_BALANCED, DROP
    ALPHA     = 0.10
    EPS_0     = 0.30
    EPS_MIN   = 0.05
    EPS_STEPS = 200

    def __init__(self):
        print(">>> [ENGINE] DRL Scheduler Initialized (Q-Learning tabular | ε=0.30→0.05 | α=0.10 | 36 states)")
        self.q_table = collections.defaultdict(lambda: {a: 0.1 for a in self.ACTIONS})
        self._step   = 0

    # ------------------------------------------------------------------ #
    # Propriedades                                                         #
    # ------------------------------------------------------------------ #

    @property
    def epsilon(self):
        decay = (self.EPS_0 - self.EPS_MIN) / self.EPS_STEPS
        return max(self.EPS_MIN, self.EPS_0 - self._step * decay)

    # ------------------------------------------------------------------ #
    # Codificação de estado                                                #
    # ------------------------------------------------------------------ #

    def _encode_state(self, task_dict, fleet):
        region_map = {"USA": 0, "BRAZIL": 1, "EUROPE": 2}
        region_idx = region_map.get(task_dict.get("region", "USA"), 0)

        candidates = [
            s for s in fleet
            if s.get("region") == task_dict.get("region")
            and s.get("ram_free", 0) >= task_dict.get("ram", 0)
        ]
        ram_ok   = 1 if candidates else 0
        best_lq  = max((s.get("link_quality", 0) for s in candidates), default=0)
        lq_bin   = 0 if best_lq < 40 else (1 if best_lq < 70 else 2)

        has_anomaly = 1 if task_dict.get("semantic_anomaly") else 0
        return (region_idx, lq_bin, ram_ok, has_anomaly)

    # ------------------------------------------------------------------ #
    # Mapeamento ação → satélite                                          #
    # ------------------------------------------------------------------ #

    def _resolve_action(self, action, candidates):
        if action == 3 or not candidates:
            return None
        if action == 0:   # PREFER_LINK
            return max(candidates, key=lambda s: s.get("link_quality", 0)).get("id")
        if action == 1:   # PREFER_RAM
            return max(candidates, key=lambda s: s.get("ram_free", 0)).get("id")
        # PREFER_BALANCED (action == 2)
        max_ram = max(s.get("ram_free", 1) for s in candidates)
        return max(
            candidates,
            key=lambda s: (0.5 * s.get("link_quality", 0) / 100.0
                           + 0.5 * s.get("ram_free", 0) / max(max_ram, 1)),
        ).get("id")

    # ------------------------------------------------------------------ #
    # Interface principal                                                  #
    # ------------------------------------------------------------------ #

    def decide(self, task_dict, fleet, min_link_quality=20.0):
        candidates = [
            s for s in fleet
            if s.get("region") == task_dict.get("region")
            and s.get("link_quality", 0) >= min_link_quality
            and s.get("ram_free", 0) >= task_dict.get("ram", 0)
        ]

        state  = self._encode_state(task_dict, fleet)
        q_vals = self.q_table[state]

        # ε-greedy
        if random.random() < self.epsilon:
            action = random.choice(self.ACTIONS)
        else:
            action = max(q_vals, key=q_vals.get)

        decision_id = self._resolve_action(action, candidates)
        forced_drop = (decision_id is None and action != 3)

        # Recompensa imediata — sem componente semântico (DRL é caixa-preta para anomalias)
        if decision_id is not None:
            chosen_lq = next(
                (s.get("link_quality", 0) for s in candidates if s.get("id") == decision_id), 0
            )
            reward = 1.0 + 0.1 * (chosen_lq / 100.0)   # bônus por link de qualidade
        elif forced_drop:
            reward = -0.5   # drop inevitável (sem candidatos válidos)
        else:
            reward = -1.0   # DROP voluntário quando havia candidatos

        # Atualização TD(0)
        q_vals[action] += self.ALPHA * (reward - q_vals[action])
        self._step += 1

        action_names = ["PREFER_LINK", "PREFER_RAM", "PREFER_BALANCED", "DROP"]
        print(f"   [DRL ε={self.epsilon:.2f} step={self._step}] "
              f"action={action_names[action]} → SAT {decision_id} | r={reward:.2f} "
              f"| Q={q_vals[action]:.3f}")

        return decision_id

    # ------------------------------------------------------------------ #
    # Diagnóstico                                                          #
    # ------------------------------------------------------------------ #

    def print_qtable(self):
        region_names = ["USA", "BRAZIL", "EUROPE"]
        lq_names     = ["low(<40%)", "mid(40-70%)", "high(>70%)"]
        action_names = ["PREFER_LINK", "PREFER_RAM", "PREFER_BALANCED", "DROP"]
        print("\n--- DRL Q-Table (estados visitados) ---")
        for (reg, lq_bin, ram, anom), qv in sorted(self.q_table.items()):
            row = f"  [{region_names[reg]} | lq={lq_names[lq_bin]} | ram_ok={ram} | anomaly={anom}] "
            row += " | ".join(f"{action_names[a]}={qv[a]:.3f}" for a in self.ACTIONS)
            print(row)
        print(f"  Estados visitados: {len(self.q_table)} / 36")
        print(f"  Decisões: {self._step} | ε final: {self.epsilon:.3f}")
        print("--------------------------------------\n")
