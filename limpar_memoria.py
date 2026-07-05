import json
import os

ARQUIVO = "memoria_consultas.json"

if os.path.exists(ARQUIVO):
    with open(ARQUIVO, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    nova_memoria = {}
    for k, v in data.items():
        chave_limpa = k.strip().lower()
        # Se houver duplicatas, mantém a mais recente (ou a que já estiver lá)
        nova_memoria[chave_limpa] = v
    
    with open(ARQUIVO, "w", encoding="utf-8") as f:
        json.dump(nova_memoria, f, ensure_ascii=False, indent=4)
    
    print(f"Memória limpa! {len(data)} entradas processadas para {len(nova_memoria)} chaves únicas.")
else:
    print("Arquivo não encontrado.")
