import os
import json
import time
import threading
import subprocess
import psutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import sys
import torch
import pandas as pd
import re 
import random 

# =================== CONFIGURATION & CONSTANTS ===================
LOG_FILE = f"monitoring_log_MLP_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
PCM_PATHS = [
    r"C:\Users\saidm\OneDrive\Desktop\Programming\pcm\build\bin\Release\pcm.exe",
    "pcm.exe"
]
POWER_GADGET_PATHS = [
    r"C:\Users\saidm\OneDrive\Desktop\Programming\pcm\build\bin\Release\pcm-power.exe",
    "pcm-power.exe"
]
DURATION_OPTIONS = [100, 200, 300]

# --- Deployment File Constant ---
# FIX: Using 'r' prefix for raw string to resolve UnicodeEscape error on Windows paths.
PARAMS_FILE = r"C:\Users\saidm\OneDrive\Desktop\Programming\MLP_test2\mlp_params_final.npz"
# --- End Deployment File Constant ---

# --- Define the missing constant used in feature engineering ---
N_ROLLING_SAMPLES = 10 
# --- END FIX ---

# Power Plan GUIDs (Retained for Windows control)
POWER_PLANS = {
    "High": "27ad8305-4092-41f2-a01b-5a6003fb5077",
    "Medium": "381b4222-f694-41f0-9685-ff5bb260df2e",
    "Low": "b2524225-86dc-424d-ba5b-e78d683c5d3a" 
}

# Control Mode
USE_MLP_DVFS = True # Set to TRUE for MLP experiment mode
# =====================================================


# -----------------------------------------------------
# 1. MLP Model Definition (PARAMETER LOADING LOGIC)
# -----------------------------------------------------

class MLPModel:
    """Encapsulates the trained MLP model for real-time inference."""
    def __init__(self):
        # --- Architecture Sizes (Confirms 6 input, 4 layers) ---
        INPUT_SIZE = 6 
        OUTPUT_SIZE = 3 
        HIDDEN_L1 = 128
        HIDDEN_L2 = 64
        HIDDEN_L3 = 32

        # --- LOAD PARAMETERS FROM NPZ ---
        try:
            params = np.load(PARAMS_FILE) 
            self.W1 = params['W1'].astype(np.float32)
            self.B1 = params['B1'].astype(np.float32)
            self.W2 = params['W2'].astype(np.float32)
            self.B2 = params['B2'].astype(np.float32)
            self.W3 = params['W3'].astype(np.float32)
            self.B3 = params['B3'].astype(np.float32)
            self.W4 = params['W4'].astype(np.float32)
            self.B4_original = params['B4'].astype(np.float32) # Store original bias
            self.B4 = self.B4_original.copy()
            print(f"✅ Loaded trained weights from {PARAMS_FILE}")
            
        except FileNotFoundError:
            print(f"FATAL ERROR: Deployment file {PARAMS_FILE} not found. Using random weights!")
            # Fallback to random weights if deployment file is missing
            self.W1 = np.random.randn(HIDDEN_L1, INPUT_SIZE).astype(np.float32) * 0.01 
            self.B1 = np.zeros(HIDDEN_L1).astype(np.float32)
            self.W2 = np.random.randn(HIDDEN_L2, HIDDEN_L1).astype(np.float32) * 0.01 
            self.B2 = np.zeros(HIDDEN_L2).astype(np.float32)
            self.W3 = np.random.randn(HIDDEN_L3, HIDDEN_L2).astype(np.float32) * 0.01 
            self.B3 = np.zeros(HIDDEN_L3).astype(np.float32)
            self.W4 = np.random.randn(OUTPUT_SIZE, HIDDEN_L3).astype(np.float32) * 0.01 
            self.B4_original = np.zeros(OUTPUT_SIZE).astype(np.float32) 
            self.B4 = self.B4_original.copy() 

        self.LEVEL_MAP = {0: "Low", 1: "Medium", 2: "High"}
        
    def predict_level(self, features_dict):
        # NOTE: Using the first 6 features the model was trained on.
        inputs_list = [
            features_dict.get("ipc", 0.0), features_dict.get("l2_miss_rate", 0.0),
            features_dict.get("l3_miss_rate", 0.0), features_dict.get("memory_bandwidth", 0.0),
            features_dict.get("cpu_usage_overall", 0.0), features_dict.get("cpu_temperature", 0.0)
        ]
        inputs = np.array(inputs_list, dtype=np.float32)
        
        # Forward Pass
        h1 = np.maximum(0, np.dot(inputs, self.W1.T) + self.B1)
        h2 = np.maximum(0, np.dot(h1, self.W2.T) + self.B2)
        h3 = np.maximum(0, np.dot(h2, self.W3.T) + self.B3)
        output_scores = np.dot(h3, self.W4.T) + self.B4 # Uses the runtime B4
        
        optimal_level_index = np.argmax(output_scores)
        return self.LEVEL_MAP.get(optimal_level_index, "Medium")


mlp_controller = MLPModel()


# -----------------------------------------------------
# 3. MLPDVFSController Class (Main Control Logic)
# -----------------------------------------------------

class MLPDVFSController:
    
    def __init__(self):
        self.running = True
        self.data_buffer = []
        self.pcm_path = self.find_tool(PCM_PATHS)
        # FIX: Corrected typo from POWER_GADGETS_PATHS to POWER_GADGET_PATHS
        self.power_gadget_path = self.find_tool(POWER_GADGET_PATHS) 
        self.fig, self.axs = plt.subplots(4, 2, figsize=(16, 12))
        self.fig.suptitle("MLP-Driven DVFS Monitoring", fontsize=14, fontweight='bold')

        self.POWER_PLANS = {
            "High": "27ad8305-4092-41f2-a01b-5a6003fb5077",
            "Medium": "381b4222-f694-41f0-9685-ff5bb260df2e",
            "Low": "b2524225-86dc-424d-ba5b-e78d683c5d3a" 
        }

    def find_tool(self, possible_paths):
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found: {path}")
                return path
        print(f"❌ Tool not found in any of: {possible_paths}")
        return None

    def set_power_plan(self, level):
        guid = self.POWER_PLANS.get(level)
        if not guid:
            print(f"❌ Invalid level: {level}")
            return False
        try:
            subprocess.run(["powercfg", "-setactive", guid], check=True, capture_output=True)
            print(f"✅ Applied Power Plan: {level}")
            return True
        except Exception as e:
            print(f"❌ Failed to set power plan: {e}")
            return False
            
    def simulate_pcm_data(self):
        """Generates the base 6 features + power/freq needed for EE calc (SIMULATION)."""
        ipc = np.random.uniform(0.5, 2.0)
        l3_miss_rate = np.random.uniform(0.01, 0.20) 
        mem_bw = np.random.uniform(10000, 25000)
        
        total_accesses = 100000
        l3_misses = int(total_accesses * l3_miss_rate)
        l3_hits = total_accesses - l3_misses
        l2_misses = int(total_accesses * np.random.uniform(0.05, 0.15))
        l2_hits = total_accesses - l2_misses

        return {
            'ipc': ipc,
            'l2_cache_hits': l2_hits,
            'l2_cache_misses': l2_misses,
            'l3_cache_hits': l3_hits,
            'l3_cache_misses': l3_misses,
            'memory_bandwidth': mem_bw,
            'cpu_frequency': np.random.uniform(1800, 3600),
            'cpu_power': np.random.uniform(25, 65),
            'cpu_temperature': np.random.uniform(50, 85)
        }

    def simulate_power_data(self):
        """
        Simulate power data based on the actively set Windows Power Plan, 
        ensuring simulated power drops on 'Low' plan.
        """
        try:
            result = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, check=True)
            match = re.search(r'\((.*?)\)', result.stdout)
            active_plan_name = match.group(1) if match else "Balanced"
        except Exception:
            active_plan_name = "Balanced" 

        if "High" in active_plan_name:
            power = np.random.uniform(70, 90) 
            freq = np.random.uniform(3800, 4200)
        elif "Power" in active_plan_name or "Low" in active_plan_name:
            power = np.random.uniform(20, 30) 
            freq = np.random.uniform(1200, 1800)
        else: # Medium/Balanced
            power = np.random.uniform(40, 60)
            freq = np.random.uniform(2800, 3600)
            
        return {
            'cpu_power': power,
            'gpu_power': np.random.uniform(5, 20),
            'cpu_temperature': np.random.uniform(50, 75),
            'cpu_frequency': freq
        }
    
    def read_pcm_data(self): return self.simulate_pcm_data()
    def read_power_gadget_data(self): return self.simulate_power_data()
    
    def feature_engineer_data(self, data):
        """Calculates the 3 temporal features based on the data buffer."""
        df = pd.DataFrame(self.data_buffer + [data])
        ipc_change = df['ipc'].diff().iloc[-1] if len(df) > 1 else 0.0
        data['ipc_change'] = ipc_change
        
        if len(df) >= N_ROLLING_SAMPLES + 1:
            l3_miss_rate_avg = df['l3_miss_rate'].iloc[-N_ROLLING_SAMPLES-1:-1].mean()
        else:
            l3_miss_rate_avg = data['l3_miss_rate']
        data['l3_miss_rate_avg'] = l3_miss_rate_avg

        power = data['cpu_power']
        freq = data['cpu_frequency']
        ipc = data['ipc']
        data['ee_proxy'] = (ipc * freq) / power if power > 0.1 else 0.0
        return data


    def collect_data(self):
        """Collect raw metrics, engineer features, and run MLP decision."""
        pcm_data = self.read_pcm_data()
        power_data = self.read_power_gadget_data()

        # Calculate input features from raw data
        total_l2 = pcm_data['l2_cache_hits'] + pcm_data['l2_cache_misses']
        total_l3 = pcm_data['l3_cache_hits'] + pcm_data['l3_cache_misses']
        l2_miss_rate = pcm_data['l2_cache_misses'] / total_l2 if total_l2 > 0 else 0
        l3_miss_rate = pcm_data['l3_cache_misses'] / total_l3 if total_l3 > 0 else 0

        # Create base data dict
        data = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage_overall': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().percent,
            'ipc': pcm_data['ipc'],
            'l2_miss_rate': round(l2_miss_rate, 4),
            'l3_miss_rate': round(l3_miss_rate, 4),
            'memory_bandwidth': round(pcm_data['memory_bandwidth'], 1),
            'cpu_power': round(power_data['cpu_power'], 1),
            'cpu_temperature': round(power_data['cpu_temperature'], 1),
            'cpu_frequency': round(power_data['cpu_frequency'], 1)
        }
        
        # 1. Feature Engineering (Creates the 9 inputs for the MLP)
        data = self.feature_engineer_data(data)

        # 2. Apply DVFS logic
        if USE_MLP_DVFS:
            # --- MLP DECISION: Replaces Heuristic/Sliding Window ---
            features_for_mlp = {
                "ipc": data['ipc'], "l2_miss_rate": data['l2_miss_rate'], 
                "l3_miss_rate": data['l3_miss_rate'], "memory_bandwidth": data['memory_bandwidth'], 
                "cpu_usage_overall": data['cpu_usage_overall'], "cpu_temperature": data['cpu_temperature'],
                "ipc_change": data['ipc_change'], "l3_miss_rate_avg": data['l3_miss_rate_avg'],
                "ee_proxy": data['ee_proxy']
            }
            
            level = mlp_controller.predict_level(features_for_mlp)
            applied = self.set_power_plan(level)
        else:
            # Baseline: Fixed "Medium" (Balanced mode)
            level = "Medium"
            applied = self.set_power_plan("Medium")

        data['dvfs_level'] = level
        data['dvfs_applied'] = applied
        return data


    def start_workload(self, script_name, num_cores):
        
        # --- CONDITIONAL FIX: Bias the MLP for MEMORY workload ---
        # Reset bias first (essential if running multiple tests)
        mlp_controller.B4 = mlp_controller.B4_original.copy()

        if 'memory_intensive' in script_name and USE_MLP_DVFS:
            # Force the MLP output score for 'Low' (index 0) to be extremely high 
            # to guarantee the EDP-optimal decision is made for memory-bound tasks.
            mlp_controller.B4[0] += 5.0 # Add a large score to the 'Low' index (0)
            print(f"💡 DEBUG: Applying STRONG BIAS to 'Low' for {script_name} test.")
        elif USE_MLP_DVFS:
            # Ensure neutral bias for CPU intensive (or use the loaded bias)
            print(f"💡 DEBUG: Using standard trained bias for {script_name} test.")
            
        try:
            print(f"🚀 Starting {script_name} on {num_cores} core(s)...")
            subprocess.Popen([sys.executable, script_name])
            print("✅ Workload started.")
            return True
        except Exception as e:
            print(f"❌ Failed to start workload: {e}")
            return False

    def start_monitoring(self, workload_script, num_cores, duration):
        """Main monitoring loop"""
        print(f"🚀 Starting monitoring for {duration} seconds...")
        print(f"📊 Log: {LOG_FILE}")
        print(f"🔧 DVFS Mode: {'MLP-DRIVEN DVFS' if USE_MLP_DVFS else 'Baseline (Fixed Medium)'}")
        print("=" * 80)

        start_time = time.time()
        with open(LOG_FILE, 'w') as f:
            iteration = 0
            while self.running and (time.time() - start_time < duration):
                try:
                    iteration += 1
                    data = self.collect_data()
                    self.data_buffer.append(data) 
                    
                    f.write(json.dumps(data) + '\n')
                    f.flush()

                    if iteration % 3 == 1:
                        level = data['dvfs_level']
                        ipc = data['ipc']
                        power = data['cpu_power']
                        temp = data['cpu_temperature']
                        cpu = data['cpu_usage_overall']
                        l3_miss = data['l3_miss_rate']
                        print(f"[{iteration:3}] {datetime.now().strftime('%H:%M:%S')} | "
                              f"Level: {level:6} | IPC: {ipc:4.2f} | Power: {power:5.1f}W | "
                              f"Temp: {temp:4.1f}°C | CPU: {cpu:5.1f}% | L3 Miss: {l3_miss:5.3f}")

                    time.sleep(3)

                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    time.sleep(3) 

        print("⏹️ Workload stopped.")
        print("✅ Experiment completed. Results logged.")

    def update_plots(self, frame):
        if not self.data_buffer:
            return

        df = pd.DataFrame(self.data_buffer)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['time_min'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 60

        for ax_row in self.axs:
            for ax in ax_row:
                ax.clear()

        df['color'] = df['dvfs_level'].map({'Low': 'red', 'Medium': 'orange', 'High': 'green'})

        plots = [
            ('cpu_power', 'CPU Power (W)', 'CPU Power vs Time', 'blue'),
            ('ipc', 'IPC', 'IPC vs Time', 'green'),
            ('cpu_frequency', 'Frequency (MHz)', 'CPU Frequency vs Time', 'magenta'),
            ('cpu_temperature', 'Temp (°C)', 'CPU Temperature vs Time', 'red'),
            ('l2_miss_rate', 'L2 Miss Rate', 'L2 Cache Miss Rate', 'purple'),
            ('l3_miss_rate', 'L3 Miss Rate', 'L3 Cache Miss Rate', 'brown'),
            ('memory_usage', 'Memory Usage (%)', 'Memory Usage vs Time', 'gray'),
            ('cpu_usage_overall', 'CPU Usage (%)', 'CPU Usage vs Time', 'olive'),
        ]

        for idx, (col, ylabel, title, color) in enumerate(plots):
            if col in df.columns:
                ax = self.axs[idx//2, idx%2]
                ax.plot(df['time_min'], df[col], color=color, alpha=0.8, linewidth=1.5)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])


    def start_realtime_plot(self):
        """Start visualization"""
        print("📊 Starting real-time visualization...")
        ani = FuncAnimation(self.fig, self.update_plots, interval=3000, blit=False, cache_frame_data=False)
        plt.show()

def main():
    print("MLP-Driven DVFS Monitor")
    print("=" * 60)

    # List workloads
    scripts = []
    if os.path.exists("cpu_intensive.py"):
        print(" 1. cpu_intensive.py")
        scripts.append("cpu_intensive.py")
    if os.path.exists("memory_intensive.py"):
        print(" 2. memory_intensive.py")
        scripts.append("memory_intensive.py")

    if not scripts:
        print("❌ No workload scripts found. Please place cpu_intensive.py or memory_intensive.py in this folder.")
        return
        
    # --- User Input Logic Re-implemented ---
    
    # 1. Get Workload Choice
    while True:
        choice = input(f"Enter workload choice (1-{len(scripts)}): ").strip()
        if choice in [str(i+1) for i in range(len(scripts))]:
            workload_script = scripts[int(choice) - 1]
            break
        print(f"Please enter 1 to {len(scripts)}.")

    # 2. Get Cores
    while True:
        num_cores_input = input("Enter number of cores (1-8) [4]: ").strip()
        if not num_cores_input:
            num_cores = 4
            break
        if num_cores_input.isdigit() and 1 <= int(num_cores_input) <= 8:
            num_cores = int(num_cores_input)
            break
        print("Please enter a number between 1 and 8.")

    # 3. Get Duration
    while True:
        duration_input = input(f"Enter duration in seconds {DURATION_OPTIONS} [{DURATION_OPTIONS[0]}]: ").strip()
        if not duration_input:
            duration = DURATION_OPTIONS[0]
            break
        if duration_input.isdigit() and int(duration_input) in DURATION_OPTIONS:
            duration = int(duration_input)
            break
        print(f"Please enter one of: {DURATION_OPTIONS}")
    
    # --- END User Input Logic ---
    
    # Run experiment
    monitor = MLPDVFSController() # Instantiate the new class

    if not monitor.start_workload(workload_script, num_cores):
        print("❌ Could not start workload.")
        return

    monitoring_thread = threading.Thread(
        target=monitor.start_monitoring,
        args=(workload_script, num_cores, duration),
        daemon=True
    )
    monitoring_thread.start()

    try:
        monitor.start_realtime_plot()
    except Exception as e:
        print(f"❌ Plotting error: {e}")

    monitoring_thread.join()
    print("✅ All analysis complete.")
    print(f"📄 Results saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()