class BaselineScheduler:
    def __init__(self):
        print(">>> [ENGINE] Baseline Scheduler Initialized (Greedy — link_quality + RAM)")

    def decide(self, task_dict, fleet, min_link_quality=20.0):
        candidatos = []

        for sat in fleet:
            if sat.get('region') != task_dict.get('region'):
                continue
            if sat.get('link_quality', 0) < min_link_quality:
                continue
            if sat.get('ram_free', 0) < task_dict.get('ram', 0):
                continue
            candidatos.append(sat)

        if not candidatos:
            return None

        # Prioriza melhor link, desempata por maior RAM livre
        best = max(candidatos, key=lambda s: (s.get('link_quality', 0), s.get('ram_free', 0)))
        return best.get('id')
