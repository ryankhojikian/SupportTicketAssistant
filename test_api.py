import requests
import time

# Test the API
try:
    # Test health check
    print("Testing health check...")
    response = requests.get("http://localhost:8002/", timeout=5)
    print(f"Health check: {response.status_code} - {response.json()}")

    # Test prediction
    print("\nTesting prediction...")
    data = {"query": "My computer won't start"}
    response = requests.post("http://localhost:8002/predict", json=data, timeout=30)
    print(f"Prediction: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("Query:", result.get("query"))
        print("Results:")
        for res in result.get("results", []):
            print(f"  {res['System']}: {res['Response']} ({res['Latency']})")
    else:
        print("Error:", response.text)

except Exception as e:
    print(f"Error: {e}")