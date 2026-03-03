#!/usr/bin/env python3
"""Mission: quantum_code — génère un circuit quantique + enregistre dans Doctorat."""
import os, sys, json, subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = "E:/QuantumForge"

FRAMEWORK_MAP = {
    "qiskit": "qiskit", "cirq": "cirq", "pennylane": "pennylane",
    "braket": "braket", "numpy": "numpy",
}

def detect_framework(text: str) -> str:
    text = text.lower()
    for key, fw in FRAMEWORK_MAP.items():
        if key in text:
            return fw
    return "qiskit"

def record_doctorat(model: str, framework: str, score: int, attempts: int, success: bool):
    """Enregistre le résultat dans ai_rankings.json via ai_rankings.py."""
    try:
        payload = json.dumps({
            "model": model, "framework": framework,
            "score": score, "attempts": attempts, "success": success,
        })
        subprocess.run(
            [sys.executable, f"{BASE_DIR}/ai_rankings.py", "record", payload],
            capture_output=True, text=True, timeout=10, cwd=BASE_DIR
        )
        print(f"[DOCTORAT] {model} | score={score} | {'✅' if success else '❌'}")
    except Exception as e:
        print(f"[DOCTORAT] Erreur enregistrement: {e}")

def run(mission: dict) -> dict:
    prompt    = mission.get("prompt", mission.get("title", ""))
    framework = detect_framework(prompt + mission.get("source_url", ""))
    model     = mission.get("model", "deepseek-v3.2:cloud") or "deepseek-v3.2:cloud"

    print(f"[QUANTUM_CODE] Prompt: {prompt[:60]}")
    print(f"[QUANTUM_CODE] Framework: {framework} | Model: {model}")

    payload = json.dumps({"prompt": prompt, "framework": framework, "model": model})

    try:
        result = subprocess.run(
            [sys.executable, f"{BASE_DIR}/generate.py", payload],
            capture_output=True, text=True, timeout=300, cwd=BASE_DIR
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0 or not stdout:
            print(f"[QUANTUM_CODE] STDERR: {stderr[-300:]}")
            record_doctorat(model, framework, 0, 3, False)
            return {"error": stderr[-300:], "success": False}

        json_lines = [l for l in stdout.splitlines() if l.strip().startswith("{")]
        if json_lines:
            output = json.loads(json_lines[-1])
            score    = output.get("score", 0)
            attempts = output.get("attempts", 1)
            success  = output.get("success", False)
            print(f"[QUANTUM_CODE] Score: {score}/100 | {output.get('title_fr', '')[:50]}")
            # ← Enregistrement Doctorat (c'était manquant)
            record_doctorat(model, framework, score, attempts, success)
            return output

        record_doctorat(model, framework, 0, 3, False)
        return {"error": "generate.py n'a pas produit de JSON", "stdout": stdout[-300:], "success": False}

    except subprocess.TimeoutExpired:
        record_doctorat(model, framework, 0, 3, False)
        return {"error": "Timeout 300s", "success": False}
    except Exception as e:
        record_doctorat(model, framework, 0, 3, False)
        return {"error": str(e), "success": False}
