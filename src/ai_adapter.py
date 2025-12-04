import os
import requests
import json
import re
import urllib3
# Desabilita avisos para o Netskope/Proxy
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = os.environ.get("GEMINI_API_KEY")

class GeminiBrain:
    def __init__(self):
        if not API_KEY:
            print("CRITICAL: GEMINI_API_KEY not found.")
        print(">>> [MEC] Gemini Brain Initialized.")

    def _call_api(self, prompt):
        if not API_KEY: return None
        # Tenta modelos estáveis
        models = ["gemini-2.0-flash", "gemini-pro"]
        headers = {'Content-Type': 'application/json'}
        data = { 
            "contents": [{"parts": [{"text": prompt}]}], 
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500} 
        }

        for m in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={API_KEY}"
            try:
                # verify=False para passar pelo Netskope/Proxy da rede
                response = requests.post(url, headers=headers, json=data, verify=False)
                if response.status_code == 200:
                    return response.json()['candidates'][0]['content']['parts'][0]['text']
            except: continue
        return None

    def decide(self, task, satellites_state):
        """
        Recebe o estado REAL do CosmicBeats e decide.
        satellites_state: Lista de dicionários com dados dos satélites.
        """
        prompt = "You are the Satellite Network Scheduler.\n"
        prompt += "Rules: 1. Region Match Strict. 2. Battery > 20%. 3. RAM Available.\n\n"
        
        prompt += f"TASK: ID {task['id']} | Region {task['region']} | RAM {task['ram']}MB\n"
        prompt += "SATELLITES:\n"
        
        candidates = False
        for s in satellites_state:
            prompt += f"SAT {s['id']}: {s['region']} | Bat {s['battery']:.1f}% | RAM {s['ram_free']}\n"
            candidates = True
            
        if not candidates: return None
        
        prompt += "\nOutput JSON: {\"satellite_id\": 101}"
        
        resp = self._call_api(prompt)
        
        if resp:
            try:
                match = re.search(r'"satellite_id"\s*:\s*(\d+)', resp)
                if match: return int(match.group(1))
            except: pass
            
        return None