import os
import requests
import json
import urllib3
import time
from dotenv import load_dotenv

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
API_KEY = os.environ.get("GEMINI_API_KEY")

class LLMScheduler:
    def __init__(self):
        if not API_KEY:
            print("CRITICAL: GEMINI_API_KEY not found in .env!")
        else:
            print(">>> [ENGINE] LLM Scheduler Initialized (gemini-3.1-flash-lite-preview)")

    def _call_api(self, prompt):
        if not API_KEY: return None
        
        model = "gemini-3.1-flash-lite-preview"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
        headers = {'Content-Type': 'application/json'}
        data = { 
            "contents": [{"parts": [{"text": prompt}]}], 
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"} 
        }

        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                response = requests.post(url, headers=headers, json=data, verify=False, timeout=15)
                latencia_ms = (time.perf_counter() - t0) * 1000

                if response.status_code == 200:
                    print(f"[LLM] Resposta em {latencia_ms:.0f}ms")
                    return response.json()['candidates'][0]['content']['parts'][0]['text']

                elif response.status_code == 429:
                    erro_detalhado = response.json().get('error', {}).get('message', 'Sem detalhes')
                    print(f"[LLM Rate Limit] Tentativa {attempt+1}/3: {erro_detalhado}")
                    continue

                else:
                    print(f"[LLM Error] HTTP {response.status_code} - {response.text}")
                    break

            except Exception as e:
                print(f"[LLM Connection Error] Tentativa {attempt+1}/3: {e}")
                continue

        return None

    def decide(self, task_dict, fleet, safe_mode_threshold=20.0):
        prompt = "You are a satellite network orchestrator. Output valid JSON only.\n"
        
        t_id = task_dict.get('id')
        t_reg = task_dict.get('region')
        t_ram = task_dict.get('ram')
        
        prompt += f"TASK: ID {t_id} | Region {t_reg} | RAM Required: {t_ram} MB\n"
        prompt += "AVAILABLE SATELLITES:\n"
        
        valid_exists = False
        
        # OTIMIZAÇÃO DO PROMPT: Pré-filtro para não gastar tokens com satélites inúteis
        for sat in fleet:
            # Ignora satélites mortos ou de outras regiões (poupando Tokens/TPM)
            if not sat.get('alive', True): continue
            if sat.get('region') != t_reg: continue 
            
            valid_exists = True
            safe_str = "SAFE_MODE" if sat.get('safe_mode', False) else "ACTIVE"
            prompt += f"SAT {sat.get('id')}: Region {sat.get('region')} | Bat {sat.get('battery', 0):.1f}% ({safe_str}) | RAM {sat.get('ram_free')}MB\n"
            
        if not valid_exists: 
            # Se o Python já viu que não há satélites válidos, nem chama a API!
            return None
        
        prompt += f"\nRULES:\n"
        prompt += f"1. Match Region strictly.\n"
        prompt += f"2. Battery > {safe_mode_threshold}%.\n"
        prompt += f"3. RAM >= Task RAM.\n"
        prompt += "Output: {\"satellite_id\": <id>} or {\"satellite_id\": null}."
        
        resp = self._call_api(prompt)
        
        if resp:
            try:
                return json.loads(resp).get("satellite_id")
            except:
                pass
        return None