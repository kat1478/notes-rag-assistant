import urllib.request
import urllib.error
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def log_test(name, success, message=""):
    status = "PASS" if success else "FAIL"
    print(f"[{status}] {name}" + (f" - {message}" if message else ""))

def test_health():
    try:
        url = f"{BASE_URL}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as res:
            if res.status != 200:
                log_test("GET /health", False, f"Status: {res.status}")
                return False
            data = json.loads(res.read().decode())
            if data.get("status") == "healthy" and ("default_model_id" in data or "ollama_model" in data):
                log_test("GET /health", True, f"Default model: {data.get('default_model_id') or data.get('ollama_model')}")
                return True
            else:
                log_test("GET /health", False, f"Invalid JSON response: {data}")
                return False
    except Exception as e:
        log_test("GET /health", False, f"Error: {e}")
        return False

def test_root():
    try:
        url = f"{BASE_URL}/"
        with urllib.request.urlopen(url, timeout=5) as res:
            final_url = res.geturl()
            if "/frontend/" in final_url or res.status == 200:
                log_test("GET /", True, f"Resolved URL: {final_url}")
                return True
            else:
                log_test("GET /", False, f"Status: {res.status}")
                return False
    except Exception as e:
        log_test("GET /", False, f"Error: {e}")
        return False

def test_frontend():
    try:
        url = f"{BASE_URL}/frontend/"
        with urllib.request.urlopen(url, timeout=5) as res:
            if res.status != 200:
                log_test("GET /frontend/", False, f"Status: {res.status}")
                return False
            content = res.read().decode()
            if "<!DOCTYPE html>" in content or "<html" in content:
                log_test("GET /frontend/", True, "HTML loaded successfully")
                return True
            else:
                log_test("GET /frontend/", False, "Not an HTML response")
                return False
    except Exception as e:
        log_test("GET /frontend/", False, f"Error: {e}")
        return False

def test_ask():
    try:
        url = f"{BASE_URL}/ask"
        payload = {"query": "What is ReLU?", "top_k": 3}
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        
        print("Sending POST /ask ('What is ReLU?') with the default model...")
        with urllib.request.urlopen(req, timeout=90) as res:
            if res.status != 200:
                log_test("POST /ask", False, f"Status: {res.status}")
                return False
            response_data = json.loads(res.read().decode())
            
            has_answer = "answer" in response_data
            has_sources = "sources" in response_data
            has_metadata = "metadata" in response_data
            
            if has_answer and has_sources and has_metadata:
                metadata = response_data["metadata"]
                time_ms = metadata.get("response_time_ms", "N/A")
                log_test("POST /ask", True, f"Answer length: {len(response_data['answer'])} chars, time: {time_ms} ms")
                return True
            else:
                missing = []
                if not has_answer: missing.append("answer")
                if not has_sources: missing.append("sources")
                if not has_metadata: missing.append("metadata")
                log_test("POST /ask", False, f"Missing fields: {', '.join(missing)}")
                return False
    except Exception as e:
        log_test("POST /ask", False, f"Error: {e}")
        return False

def main():
    print(f"Starting Stage 4 Smoke Test against {BASE_URL}...\n")
    results = [
        test_health(),
        test_root(),
        test_frontend(),
        test_ask()
    ]
    
    print("\n--- Summary ---")
    if all(results):
        print("All smoke tests PASSED!")
        sys.exit(0)
    else:
        print("Some smoke tests FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
