class DRLScheduler:
    """
    DRL stub: simula comportamento de um agente de Deep Reinforcement Learning.

    Características intencionais para a comparação científica:
    - Opera como "caixa-preta": ignora completamente semantic_anomaly (não treinado para isso).
    - Usa Q-value simulado (score ponderado bateria + RAM) em vez de heurística gulosa pura.
    - Serve de baseline comparativo até o treinamento real com Stable Baselines3.
    """

    def __init__(self):
        print(">>> [ENGINE] DRL Scheduler Initialized (Stub — Q-value simulation)")
        # Pesos do Q-value simulado (seriam aprendidos via PPO/DQN no treinamento real)
        self.w_battery = 0.6
        self.w_ram = 0.4

    def decide(self, task_dict, fleet, safe_mode_threshold=20.0):
        candidatos = []
        for sat in fleet:
            if not sat.get('alive', True):
                continue
            if sat.get('region') != task_dict.get('region'):
                continue
            if sat.get('battery', 0) <= safe_mode_threshold:
                continue
            if sat.get('ram_free', 0) < task_dict.get('ram', 0):
                continue
            candidatos.append(sat)

        if not candidatos:
            return None

        max_ram = max(s.get('ram_free', 1) for s in candidatos)
        best = max(
            candidatos,
            key=lambda s: (
                self.w_battery * s.get('battery', 0) / 100.0
                + self.w_ram * s.get('ram_free', 0) / max(max_ram, 1)
            ),
        )
        return best.get('id')
