# plot_results.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# 1. Encontra o log mais recente
list_of_files = glob.glob('logs/*.jsonl')
latest_file = max(list_of_files, key=os.path.getctime)
print(f"Plotting results from: {latest_file}")

# 2. Carrega dados
data = []
with open(latest_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# 3. Processamento para Gráficos

# --- Gráfico A: Bateria ao Longo do Tempo ---
telemetry = df[df['type'] == 'TELEMETRY'].copy()
telemetry['sat_id'] = telemetry['details'].apply(lambda x: x['sat_id'])
telemetry['battery'] = telemetry['details'].apply(lambda x: x['battery'])

plt.figure(figsize=(10, 5))
for sat in telemetry['sat_id'].unique():
    sat_data = telemetry[telemetry['sat_id'] == sat]
    plt.plot(sat_data['timestamp'], sat_data['battery'], label=f"Sat {sat}")

plt.axhline(y=20, color='r', linestyle='--', label='Safe Mode (20%)')
plt.title("Battery Level over Time")
plt.xlabel("Simulation Time (s)")
plt.ylabel("Battery %")
plt.legend()
plt.grid(True)
plt.savefig("logs/plot_battery.png")
print("Saved logs/plot_battery.png")

# --- Gráfico B: Status das Tarefas ---
# Conta eventos de sucesso vs falha
completed = len(df[df['type'] == 'TASK_COMPLETED'])
dropped = len(df[df['type'] == 'TASK_DROPPED']) # OOM
rejected = len(df[df['type'] == 'TASK_REJECTED']) # Safe Mode
sched_fail = len(df[df['type'] == 'SCHEDULER_REJECT']) # LLM não achou ninguém

labels = ['Completed', 'OOM Dropped', 'Safe Mode Rejected', 'No Route']
values = [completed, dropped, rejected, sched_fail]

plt.figure(figsize=(8, 8))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Task Outcome Distribution")
plt.savefig("logs/plot_outcomes.png")
print("Saved logs/plot_outcomes.png")

plt.show()