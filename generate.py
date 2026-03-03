import sys, os, json, subprocess, tempfile, re, requests
import env_loader
from ollama import Client
from datetime import datetime, timezone

PRO_URL        = os.getenv("OLLAMA_PRO_URL",   "https://ollama.com")
PRO_KEY        = os.getenv("OLLAMA_API_KEY",   "")
LOCAL_OLLAMA_URL = os.getenv("LOCAL_OLLAMA_URL", "http://localhost:11434")

# Dossier de tracking pour le programme Doctorat
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TRACKING_FILE = os.path.join(BASE_DIR, "data", "model_performance.jsonl")

FRAMEWORKS = {
    "qiskit":    "Use Qiskit 2.x and qiskit-aer. Use AerSimulator. End with: print(json.dumps(counts))",
    "cirq":      "Use Cirq 1.x. Simulate with cirq.Simulator(). End with: print(json.dumps({'result': str(result)}))",
    "pennylane": "Use PennyLane with default.qubit. End with: print(json.dumps({'result': str(result.tolist() if hasattr(result,'tolist') else result)}))",
    "braket":    "Use Amazon Braket LocalSimulator. End with: print(json.dumps(dict(result.measurement_counts)))",
    "numpy":     "Use NumPy only. End with: print(json.dumps({'result': result.tolist() if hasattr(result,'tolist') else str(result)}))",
}

SYSTEM_PROMPT = (
    "You are an expert quantum computing programmer.\n"
    "Generate ONLY valid Python code, no explanations, no markdown blocks.\n"
    "Always import json at the top.\n"
    "The last line MUST print a JSON dict of results to stdout.\n"
)


def is_cloud_model(name: str) -> bool:
    return name.endswith("-cloud")


def extract_code(text: str) -> str:
    fence   = chr(96) * 3
    pattern = fence + r"(?:python)?\n(.*?)" + fence
    match   = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def call_ollama_pro(messages: list, model: str) -> str:
    """Route les appels vers Ollama Cloud (REST) ou Ollama local."""
    if is_cloud_model(model):
        url     = f"{PRO_URL}/api/chat"
        headers = {
            "Authorization": f"Bearer {PRO_KEY}",
            "Content-Type":  "application/json",
        }
        payload = {"model": model, "messages": messages, "options": {"temperature": 0.2}}
        resp    = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    client = Client(host=LOCAL_OLLAMA_URL)
    resp   = client.chat(model=model, messages=messages, options={"temperature": 0.2})
    return resp.message.content


def run_code(code: str, execution_timeout: int = 60):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            ["python", tmp], capture_output=True, text=True, timeout=execution_timeout
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", f"Timeout: execution exceeded {execution_timeout}s"
    finally:
        os.unlink(tmp)


def track_performance(model: str, success: bool, attempts: int, score: int = -1, prompt: str = ""):
    """
    Enregistre les performances du modele pour le programme Doctorat.
    Chaque ligne JSONL = une generation. ai_rankings.py agrege ces donnees.
    """
    os.makedirs(os.path.dirname(TRACKING_FILE), exist_ok=True)
    record = {
        "ts":       datetime.now(timezone.utc).isoformat(),
        "model":    model,
        "success":  success,
        "attempts": attempts,
        "score":    score,          # -1 = pas encore evalue (scoring LLM separé)
        "prompt":   prompt[:80],
    }
    with open(TRACKING_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_and_run(
    prompt: str,
    framework: str,
    model: str,
    max_retries: int = 3,
    execution_timeout: int = 60,
) -> dict:
    # Framework "auto" → qiskit par defaut
    if not framework or framework == "auto":
        framework = "qiskit"

    fw_hint  = FRAMEWORKS.get(framework, FRAMEWORKS["qiskit"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Framework: {framework}. {fw_hint}\n\nTask: {prompt}"},
    ]

    code = ""; stderr = ""
    for attempt in range(1, max_retries + 1):
        print(f" [Tentative {attempt}/{max_retries}] Generation...", flush=True)
        raw  = call_ollama_pro(messages, model)
        code = extract_code(raw)

        print(f" [Tentative {attempt}/{max_retries}] Execution...", flush=True)
        ok, stdout, stderr = run_code(code, execution_timeout)

        if ok and stdout:
            try:
                result = json.loads(stdout)
                track_performance(model, True, attempt, prompt=prompt)
                return {
                    "success":   True,
                    "code":      code,
                    "result":    result,
                    "framework": framework,
                    "model":     model,
                    "attempts":  attempt,
                    "prompt":    prompt,
                }
            except json.JSONDecodeError:
                stderr = f"Output is not valid JSON: {stdout}"

        print(f" [ERREUR] {stderr[:300]}", flush=True)
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role":    "user",
            "content": f"Your code produced this error:\n{stderr}\nFix it. Return ONLY corrected Python code, no markdown.",
        })

    track_performance(model, False, max_retries, prompt=prompt)
    return {
        "success":   False,
        "code":      code,
        "error":     stderr,
        "framework": framework,
        "model":     model,
        "attempts":  max_retries,
        "prompt":    prompt,
    }


if __name__ == "__main__":
    data   = json.loads(sys.argv[1])
    result = generate_and_run(
        prompt            = data["prompt"],
        framework         = data.get("framework", "qiskit"),
        model             = data["model"],
        max_retries       = data.get("max_retries", 3),
        execution_timeout = data.get("execution_timeout", 60),
    )
    print(json.dumps(result, ensure_ascii=False))
