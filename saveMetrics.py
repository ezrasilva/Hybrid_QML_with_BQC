import json
import numpy as np
import os
from datetime import datetime

def make_json_serializable(obj):
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    return obj

def saveMetrics(
    media,
    num_eprs,
    QMLMetrics,
    MatrixConfusion,
    folder: str,
    filename: str
) -> None:
    # Cria a pasta se não existir
    os.makedirs(f"metrics/{folder}", exist_ok=True)

    # Caminho completo do arquivo
    filepath = os.path.join(f"metrics/{folder}", filename)

    # Monta dicionário de métricas
    metrics = {
        "media": media,
        "num_eprs": num_eprs,
        "QMLMetrics": make_json_serializable(QMLMetrics),
        "MatrixConfusion": make_json_serializable(MatrixConfusion),
        "timestamp": datetime.now().isoformat()
    }

    # Salva em JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"✅ Métricas salvas em {filepath}")
