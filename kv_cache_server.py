import os
import json
import torch
from http.server import BaseHTTPRequestHandler, HTTPServer
import traceback

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Or your desired GPU
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- Global Variables ---
ITEM_KV_CACHE_STORE = {}
LLM_MODEL = None
TOKENIZER = None
CACHE_METADATA = None
NUM_LAYERS = None

# --- Initialization Functions ---
def initialize_model():
    global LLM_MODEL, TOKENIZER, CACHE_METADATA, NUM_LAYERS
    print("Initializing model and tokenizer...")
    LLM_MODEL = LLM(model="mistralai/Mistral-7B-Instruct-v0.3", gpu_memory_utilization=0.95, max_model_len=10000) # Adjust max_model_len if needed
    TOKENIZER = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    LLM_MODEL.set_tokenizer(TOKENIZER)
    
    try:
        CACHE_METADATA = LLM_MODEL.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata
        NUM_LAYERS = 32
    except AttributeError as e:
        print(f"Error accessing model internals for cache_metadata or num_layers: {e}")
        print("KV cache collection might not work as expected.")
        # Fallback or exit if essential parts are missing
        NUM_LAYERS = 32 # Default for Mistral-7B if dynamic access fails
        CACHE_METADATA = {'collect': False, 'check': False, 'imp_indices': None} # Dummy
        # Consider exiting if CACHE_METADATA is critical and cannot be obtained.
    
    print(f"Model and tokenizer initialized. Number of layers: {NUM_LAYERS}")

def collect_all_item_kvs():
    global ITEM_KV_CACHE_STORE, LLM_MODEL, TOKENIZER, CACHE_METADATA, NUM_LAYERS

    if not all([LLM_MODEL, TOKENIZER, CACHE_METADATA, NUM_LAYERS]):
        print("Model not initialized. Cannot collect KVs.")
        return

    print("Starting KV cache collection for all items from all_items.json...")
    all_items_path = os.path.join(os.path.dirname(__file__), "..", "all_items.json") # Path to all_items.json

    if not os.path.exists(all_items_path):
        print(f"Error: all_items.json not found at {all_items_path}")
        return

    try:
        with open(all_items_path, 'r') as f:
            all_items_data = json.load(f)
    except Exception as e:
        print(f"Error loading all_items.json: {e}")
        return

    if not all_items_data:
        print("No items found in all_items.json")
        return

    total_items_processed = 0
    processed_item_ids_globally = set() # To ensure itemID from field is unique

    for item_data in all_items_data: # Iterate through items from all_items.json
        # Extract itemID from the item_data itself
        actual_item_id = item_data.get("itemID") # Assuming the field is named "itemID"
        source_file_id = f"all_items.json_item_{actual_item_id}" # For logging

        if not actual_item_id:
            print(f"  Warning: 'itemID' field missing or empty in item from {source_file_id}. Skipping.")
            continue
        
        if actual_item_id in processed_item_ids_globally:
            print(f"  Warning: itemID '{actual_item_id}' (from item field) has already been processed. Skipping item from {source_file_id} to avoid overwriting. Ensure itemIDs are unique.")
            continue
        
        # Use actual_item_id as the key for the cache store
        if actual_item_id in ITEM_KV_CACHE_STORE: 
            print(f"  Skipping already processed item_id (from field): {actual_item_id} (source: {source_file_id})")
            continue

        item_prompt_str = f"\n- {json.dumps(item_data)}" 

        try:
            CACHE_METADATA['collect'] = True
            CACHE_METADATA['check'] = False
            CACHE_METADATA['imp_indices'] = None

            # Generate with max_tokens=1 as we only need the KV cache from the prefill pass
            LLM_MODEL.generate([item_prompt_str], SamplingParams(temperature=0.1, max_tokens=1, prompt_logprobs=None))
            
            item_kv_layers_cpu = []
            model_layers_ref = LLM_MODEL.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
            
            for j in range(NUM_LAYERS):
                # hack_kv contains [key_cache, value_cache] for the last processed sequence (item_prompt_str)
                # These tensors are on GPU.
                # llm.generate likely tokenizes item_prompt_str by adding BOS.
                # We slice [1:] to get KV for the item content, excluding BOS.
                key_cache_gpu = model_layers_ref[j].self_attn.hack_kv[0]
                value_cache_gpu = model_layers_ref[j].self_attn.hack_kv[1]

                if key_cache_gpu.shape[0] > 1 and value_cache_gpu.shape[0] > 1 : # Ensure there's something to slice
                    key_cpu = key_cache_gpu[1:].clone().cpu() # Slice off BOS, clone, move to CPU
                    value_cpu = value_cache_gpu[1:].clone().cpu()
                elif key_cache_gpu.shape[0] == 1 and value_cache_gpu.shape[0] == 1: # Only BOS token? Or single token item.
                    print(f"    Warning: Item {actual_item_id} (source: {source_file_id}) resulted in KV cache of sequence length 1. Storing empty KV for content part.")
                    key_cpu = torch.empty(0, *key_cache_gpu.shape[1:], device='cpu', dtype=key_cache_gpu.dtype)
                    value_cpu = torch.empty(0, *value_cache_gpu.shape[1:], device='cpu', dtype=value_cache_gpu.dtype)
                else: # seq_len is 0, should not happen if BOS is added.
                     key_cpu = key_cache_gpu.clone().cpu()
                     value_cpu = value_cache_gpu.clone().cpu()

                item_kv_layers_cpu.append({
                    "key": key_cpu, 
                    "value": value_cpu,
                    "key_shape": list(key_cpu.shape), # Store shape for easier reconstruction
                    "value_shape": list(value_cpu.shape)
                    })
            
            ITEM_KV_CACHE_STORE[actual_item_id] = item_kv_layers_cpu
            processed_item_ids_globally.add(actual_item_id) # Add to set of processed IDs
            total_items_processed += 1
            if total_items_processed % 50 == 0:
                print(f"  Processed {total_items_processed} items so far. Last actual_item_id: {actual_item_id} (source: {source_file_id})")

        except Exception as e:
            print(f"  Error processing item with actual_item_id {actual_item_id} (source: {source_file_id}): {e}")
            traceback.print_exc()
        finally:
            CACHE_METADATA['collect'] = False # Reset flag

    print(f"KV cache collection finished. Total items in store: {len(ITEM_KV_CACHE_STORE)}")

# --- HTTP Server ---
class KVRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global ITEM_KV_CACHE_STORE
        if self.path.startswith('/get_kv/'):
            item_id = self.path.split('/')[-1]
            if item_id in ITEM_KV_CACHE_STORE:
                kv_data_for_item_layers_tensor = ITEM_KV_CACHE_STORE[item_id]
                
                # Serialize tensors to lists for JSON
                serialized_kv_layers = []
                for layer_kv in kv_data_for_item_layers_tensor:
                    serialized_kv_layers.append({
                        'key': layer_kv['key'].tolist(),
                        'value': layer_kv['value'].tolist(),
                        'key_shape': layer_kv['key_shape'],
                        'value_shape': layer_kv['value_shape']
                    })
                
                response_data = json.dumps(serialized_kv_layers)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Content-Length', str(len(response_data)))
                self.end_headers()
                self.wfile.write(response_data.encode('utf-8'))
            else:
                self.send_error(404, f'Item ID {item_id} not found')
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "cache_size": len(ITEM_KV_CACHE_STORE)}).encode('utf-8'))
        else:
            self.send_error(404, 'Endpoint not found. Use /get_kv/<item_id> or /health')

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, KVRequestHandler)
    print(f"Starting HTTP server on port {port}...")
    print(f"Available endpoints: /get_kv/<item_id>, /health")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
    finally:
        httpd.server_close()
        print("Server stopped.")

# --- Main Execution ---
if __name__ == "__main__":
    initialize_model()
    if LLM_MODEL: # Proceed only if model initialization was successful
        collect_all_item_kvs()
        run_server()
    else:
        print("Failed to initialize LLM. Server cannot start.")

