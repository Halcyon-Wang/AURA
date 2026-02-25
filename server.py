import os
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from flask import Flask, request, jsonify
from flask_cors import CORS
from analyzer import analyze_audio

app = Flask(__name__)
CORS(app)

# --- AURA 16.6: ASYNCHRONOUS CONCURRENCY POOL ---
# Dynamically detect CPU cores (leave 1 core for OS/Browser stability)
CORE_COUNT = max(1, multiprocessing.cpu_count() - 1)
dsp_pool = ProcessPoolExecutor(max_workers=CORE_COUNT)
# -------------------------------------------------

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        temp_fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1])
        try:
            with os.fdopen(temp_fd, 'wb') as f:
                file.save(f)
                
            # OFF-LOAD TO CPU POOL: Bypasses Python GIL completely
            future = dsp_pool.submit(analyze_audio, temp_path)
            trajectory = future.result() 
            
            return jsonify(trajectory)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    print(f"AURA ENGINE: Igniting Concurrency Pool with {CORE_COUNT} Dedicated DSP Cores...")
    # threaded=True allows Flask to accept multiple HTTP requests simultaneously and map them to the Process Pool
    app.run(host='0.0.0.0', port=5001, threaded=True)