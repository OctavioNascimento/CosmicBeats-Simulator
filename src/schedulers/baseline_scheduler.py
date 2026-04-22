class BaselineScheduler:
    def __init__(self):
        print(">>> [ENGINE] Baseline Scheduler Initialized (Filtered Shortest Queue/Battery)")

    def decide(self, task_dict, fleet, safe_mode_threshold=20.0):
        candidatos_validos = []

        # 1. FILTRO RÍGIDO (As regras do teu TCC)
        for sat in fleet:
            if not sat.get('alive', True): continue
            if sat.get('region') != task_dict.get('region'): continue
            if sat.get('battery', 0) <= safe_mode_threshold: continue
            if sat.get('ram_free', 0) < task_dict.get('ram', 0): continue
            
            candidatos_validos.append(sat)

        if not candidatos_validos:
            return None # Ninguém cumpriu as regras (Vai para "No Route")

        # 2. HEURÍSTICA DE ESCOLHA (Energy-Aware com desempate em RAM)
        # Retorna uma tupla (bateria, ram_free). O Python avalia o primeiro e desempata com o segundo!
        best_sat = max(candidatos_validos, key=lambda s: (s.get('battery', 0), s.get('ram_free', 0)))
        
        return best_sat.get('id')