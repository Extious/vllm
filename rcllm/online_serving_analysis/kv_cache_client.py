\
import requests
import time
import json

# --- Configuration ---
SERVER_URL = "http://localhost:8000"  # Assuming the server runs on localhost:8000
ITEM_IDS_TO_QUERY = [
    "B001A3ML3K",  # From history.json
    "B001Q878WI",  # From history.json
    "B0051OKX98",  # From history.json
    "B005C48WB8",  # From candidate.json
    "B006GQEWEC",  # From candidate.json
    "B0026BA9HA",  # From candidate.json
    "B007RZDDO6",  # From candidate.json
    "NON_EXISTENT_ID_12345" # Example of an ID that might not be in the cache
]
NUM_REQUESTS_PER_ITEM = 3 # Number of times to query each item to get a more stable average

def get_kv_from_server(item_id):
    """Fetches KV cache for a given item_id from the server."""
    try:
        response = requests.get(f"{SERVER_URL}/get_kv/{item_id}")
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()  # Assuming the server returns JSON
    except requests.exceptions.RequestException as e:
        print(f"Error fetching KV for item_id {item_id}: {e}")
        return None

def test_query_performance():
    print(f"Starting KV cache query performance test against {SERVER_URL}...")
    total_time_taken = 0
    successful_queries = 0
    
    # First, check server health
    try:
        health_response = requests.get(f"{SERVER_URL}/health")
        health_response.raise_for_status()
        health_data = health_response.json()
        print(f"Server health: {health_data}")
        if health_data.get("cache_size", 0) == 0:
            print("Warning: Server cache is empty. Query tests might not be meaningful.")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server or server is unhealthy: {e}")
        print("Please ensure the kv_cache_server.py is running and has collected KVs.")
        return

    print(f"\\nQuerying {len(ITEM_IDS_TO_QUERY)} item(s), {NUM_REQUESTS_PER_ITEM} time(s) each...")

    for item_id in ITEM_IDS_TO_QUERY:
        item_total_time = 0
        item_successful_queries = 0
        print(f"  Testing item_id: {item_id}")
        for i in range(NUM_REQUESTS_PER_ITEM):
            start_time = time.perf_counter()
            kv_data = get_kv_from_server(item_id)
            end_time = time.perf_counter()

            if kv_data is not None:
                # You could add a basic check here to see if kv_data looks valid if needed
                # For example, check if it's a list and not empty
                # print(f"    Attempt {i+1}: Received {len(kv_data)} layers.")
                duration = (end_time - start_time)
                item_total_time += duration
                item_successful_queries += 1
            else:
                print(f"    Attempt {i+1}: Failed to retrieve KV for {item_id}")
        
        if item_successful_queries > 0:
            avg_item_time_ms = (item_total_time / item_successful_queries) * 1000
            print(f"    Average query time for {item_id} ({item_successful_queries} successful): {avg_item_time_ms:.4f} ms")
            total_time_taken += item_total_time
            successful_queries += item_successful_queries
        else:
            print(f"    No successful queries for item_id: {item_id}")

    print("\\n--- Test Summary ---")
    if successful_queries > 0:
        average_total_time_ms = (total_time_taken / successful_queries) * 1000
        print(f"Total successful queries: {successful_queries}")
        print(f"Overall average query time for successful requests: {average_total_time_ms:.4f} ms")
        print(f"Total time spent on successful queries: {total_time_taken:.4f} seconds")
    else:
        print("No items were successfully queried from the server.")
    print("--------------------")

if __name__ == "__main__":
    test_query_performance()
