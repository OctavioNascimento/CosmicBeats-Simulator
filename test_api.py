import os
import requests
import urllib3

# Desabilita avisos de segurança (para o Netskope não atrapalhar o log)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = os.environ.get("GEMINI_API_KEY")

def test_connectivity():
    print("--- 1. Verificando Chave API ---")
    if not API_KEY:
        print("ERRO: A variável GEMINI_API_KEY não está definida.")
        return
    print(f"Chave encontrada: {API_KEY[:5]}...{API_KEY[-5:]}")

    print("\n--- 2. Listando Modelos Disponíveis (GET) ---")
    # Este endpoint lista tudo o que sua chave tem permissão para ver
    url_list = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    
    try:
        response = requests.get(url_list, verify=False) # verify=False para o Netskope
        
        if response.status_code == 200:
            data = response.json()
            print("SUCESSO! Conexão estabelecida.")
            print("Modelos disponíveis para você:")
            available_models = []
            if 'models' in data:
                for m in data['models']:
                    # Filtramos apenas os que servem para gerar texto
                    if 'generateContent' in m['supportedGenerationMethods']:
                        print(f" - {m['name']}")
                        available_models.append(m['name'])
            else:
                print("Nenhum modelo encontrado na lista.")
            
            return available_models
        else:
            print(f"FALHA AO LISTAR MODELOS. Código: {response.status_code}")
            print(f"Resposta: {response.text}")
            return []

    except Exception as e:
        print(f"ERRO CRÍTICO DE CONEXÃO: {e}")
        return []

def test_generation(model_full_name):
    print(f"\n--- 3. Testando Geração com {model_full_name} ---")
    # A URL precisa ser exata. O nome do modelo já vem como "models/gemini-pro" da lista
    url_gen = f"https://generativelanguage.googleapis.com/v1beta/{model_full_name}:generateContent?key={API_KEY}"
    
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": "Hello, are you working?"}]}]
    }
    
    try:
        response = requests.post(url_gen, headers=headers, json=data, verify=False)
        if response.status_code == 200:
            print(f"SUCESSO! O modelo {model_full_name} respondeu:")
            print(response.json()['candidates'][0]['content']['parts'][0]['text'])
            return True
        else:
            print(f"ERRO NA GERAÇÃO. Código: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Erro na requisição: {e}")
        return False

if __name__ == "__main__":
    models = test_connectivity()
    
    if models:
        # Tenta testar o primeiro modelo da lista que pareça ser o Gemini
        print("\n--- Tentando validar o primeiro modelo da lista ---")
        # Preferência por gemini-1.5-flash ou gemini-pro
        chosen = None
        for m in models:
            if "gemini-1.5-flash" in m:
                chosen = m
                break
        if not chosen and len(models) > 0:
            chosen = models[0]
            
        if chosen:
            test_generation(chosen)