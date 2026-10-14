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
import pandas as pd
import re
import random
import signal  # for Ctrl+C handling

# ============ GLOBAL STOP FLAG & CTRL+C HANDLER ============
STOP_REQUESTED = False

def handle_ctrl_c(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠️ Ctrl+C detected — requesting shutdown...")

signal.signal(signal.SIGINT, handle_ctrl_c)
# ===========================================================


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

# Deployment files (update paths if needed)
PARAMS_FILE = r"C:\Users\saidm\OneDrive\Desktop\Programming\MLP_test2\mlp_params_final.npz"
SCALER_FILE = r"C:\Users\saidm\OneDrive\Desktop\Programming\MLP_test2\dvfs_scaler_stats.npz"

# Rolling window for temporal features (must match training: 5)
N_ROLLING_SAMPLES = 5

# Power Plan GUIDs
POWER_PLANS = {
    "High":   "27ad8305-4092-41f2-a01b-5a6003fb5077",
    "Medium": "381b4222-f694-41f0-9685-ff5bb260df2e",
    "Low":    "b2524225-86dc-424d-ba5b-e78d683c5d3a"
}

# Control mode: True = MLP-driven DVFS, False = baseline fixed "Medium"
USE_MLP_DVFS = True  # <- Option A: we now run MLP DVFS by default
# =====================================================

# For Baseline mode, we always set Medium
# USE_MLP_DVFS = False

# -----------------------------------------------------
# 1. MLP Model Definition (PARAMETER + SCALER LOADING)
# -----------------------------------------------------
class MLPModel:
    """Encapsulates the trained MLP model for real-time inference."""

    def __init__(self):
        # Architecture sizes (for sanity/info)
        INPUT_SIZE = 12
        OUTPUT_SIZE = 3

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
            self.B4 = params['B4'].astype(np.float32)
            print(f"✅ Loaded trained weights from {PARAMS_FILE}")
        except Exception as e:
            print(f"❌ FATAL ERROR: Could not load deployment file {PARAMS_FILE}: {e}")
            raise e

        # --- LOAD SCALER STATS (mean & scale from training) ---
        try:
            scaler_stats = np.load(SCALER_FILE)
            self.scaler_mean = scaler_stats["mean"].astype(np.float32)
            self.scaler_scale = scaler_stats["scale"].astype(np.float32)
            print(f"✅ Loaded scaler stats from {SCALER_FILE}")
        except Exception as e:
            print(f"⚠️ WARNING: Could not load scaler stats from {SCALER_FILE}: {e}")
            print("   The MLP will run WITHOUT proper normalization. "
                  "Predictions may collapse to one level.")
            self.scaler_mean = None
            self.scaler_scale = None

        self.LEVEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

    def predict_level(self, features_dict):
        """
        Build the 12-D feature vector exactly as used in training:

          [ipc, l2_miss_rate, l3_miss_rate, memory_bandwidth,
           cpu_usage_overall, cpu_temperature, cpu_power, cpu_frequency,
           ipc_change, ipc_avg_5, l3_miss_rate_avg_5, bw_util]

        then apply StandardScaler normalization, then run the MLP.
        """
        x = np.array([
            features_dict.get("ipc", 0.0),
            features_dict.get("l2_miss_rate", 0.0),
            features_dict.get("l3_miss_rate", 0.0),
            features_dict.get("memory_bandwidth", 0.0),
            features_dict.get("cpu_usage_overall", 0.0),
            features_dict.get("cpu_temperature", 0.0),
            features_dict.get("cpu_power", 0.0),
            features_dict.get("cpu_frequency", 0.0),
            features_dict.get("ipc_change", 0.0),
            features_dict.get("ipc_avg_5", 0.0),
            features_dict.get("l3_miss_rate_avg_5", 0.0),
            features_dict.get("bw_util", 0.0),
        ], dtype=np.float32)

        # Apply scaler if available
        if self.scaler_mean is not None and self.scaler_scale is not None:
            x = (x - self.scaler_mean) / self.scaler_scale

        # Forward pass (ReLU MLP)
        h1 = np.maximum(0, np.dot(x, self.W1.T) + self.B1)
        h2 = np.maximum(0, np.dot(h1, self.W2.T) + self.B2)
        h3 = np.maximum(0, np.dot(h2, self.W3.T) + self.B3)
        output_scores = np.dot(h3, self.W4.T) + self.B4

        idx = int(np.argmax(output_scores))
        return self.LEVEL_MAP.get(idx, "Medium")


mlp_controller = MLPModel()


# -----------------------------------------------------
# 2. MLPDVFSController Class (Main Control Logic)
# -----------------------------------------------------
class MLPDVFSController:

    def __init__(self):
        self.running = True
        self.data_buffer = []
        self.max_bw_observed = 25000.0  # MB/s heuristic; updated online

        self.pcm_path = self.find_tool(PCM_PATHS)
        self.power_gadget_path = self.find_tool(POWER_GADGET_PATHS)

        self.fig, self.axs = plt.subplots(4, 2, figsize=(16, 12))
        self.fig.suptitle("MLP-Driven DVFS Monitoring", fontsize=14, fontweight='bold')

        self.POWER_PLANS = POWER_PLANS.copy()

    # ---------- Utility ----------
    def find_tool(self, possible_paths):
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found: {path}")
                return path
        print(f"⚠️ Tool not found in any of: {possible_paths}")
        return None

    def set_power_plan(self, level):
        guid = self.POWER_PLANS.get(level)
        if not guid:
            print(f"❌ Invalid level: {level}")
            return False
        try:
            subprocess.run(["powercfg", "-setactive", guid],
                           check=True, capture_output=True)
            # print(f"✅ Applied Power Plan: {level}")
            return True
        except Exception as e:
            print(f"❌ Failed to set power plan: {e}")
            return False

    # ---------- Simulation fallback ----------
    def simulate_pcm_data(self):
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
        }

    def simulate_power_data(self):
        # Simple heuristic based on active power plan
        try:
            result = subprocess.run(["powercfg", "/getactivescheme"],
                                    capture_output=True, text=True, check=True)
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
        else:  # Balanced / Medium
            power = np.random.uniform(40, 60)
            freq = np.random.uniform(2800, 3600)

        return {
            'cpu_power': power,
            'gpu_power': np.random.uniform(5, 20),
            'cpu_temperature': np.random.uniform(50, 75),
            'cpu_frequency': freq
        }

    # ---------- REAL PCM + POWER (with fallback) ----------
    def read_pcm_data(self):
        """Try to read from Intel PCM; fall back to simulation on failure."""
        if not self.pcm_path:
            return self.simulate_pcm_data()

        try:
            result = subprocess.run(
                [self.pcm_path, "1", "-nc", "-ns"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return self.simulate_pcm_data()

            data_line = lines[-1].strip()
            parts = [x for x in data_line.split(' ') if x]

            # WARNING: This parsing depends on your pcm.exe format.
            # Adjust indices if needed.
            if len(parts) >= 8:
                ipc = float(parts[1])
                mem_bw = float(parts[2])      # MB/s
                l2_hits = int(parts[4])
                l2_misses = int(parts[5])
                l3_hits = int(parts[6])
                l3_misses = int(parts[7])

                return {
                    'ipc': ipc,
                    'l2_cache_hits': l2_hits,
                    'l2_cache_misses': l2_misses,
                    'l3_cache_hits': l3_hits,
                    'l3_cache_misses': l3_misses,
                    'memory_bandwidth': mem_bw
                }
        except Exception as e:
            print(f"PCM Error: {e}")
        return self.simulate_pcm_data()

    def read_power_gadget_data(self):
        """Try to read from pcm-power.exe; fall back to simulation on failure."""
        if not self.power_gadget_path:
            return self.simulate_power_data()

        temp_file = "power_temp.csv"
        cmd = [self.power_gadget_path, "-duration", "1", "-file", temp_file]

        try:
            subprocess.run(cmd, capture_output=True, timeout=5)
            if not os.path.exists(temp_file):
                return self.simulate_power_data()

            with open(temp_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:
                    os.remove(temp_file)
                    return self.simulate_power_data()

                data = lines[-1].strip().split(',')
                power_data = {
                    'cpu_power': float(data[1]) if len(data) > 1 and data[1] else 0.0,
                    'gpu_power': float(data[2]) if len(data) > 2 and data[2] else 0.0,
                    'cpu_temperature': float(data[3]) if len(data) > 3 and data[3] else 0.0,
                    'cpu_frequency': float(data[4]) if len(data) > 4 and data[4] else 0.0
                }
                os.remove(temp_file)
                return power_data
        except Exception as e:
            print(f"Power Gadget Error: {e}")
        return self.simulate_power_data()

    # ---------- Feature Engineering ----------
    def feature_engineer_data(self, data):
        """
        Compute temporal features to match training:
          - ipc_change
          - ipc_avg_5
          - l3_miss_rate_avg_5
          - bw_util
        """
        df = pd.DataFrame(self.data_buffer + [data])

        # ipc_change: difference vs previous sample
        if len(df) > 1:
            ipc_change = df['ipc'].diff().iloc[-1]
        else:
            ipc_change = 0.0
        data['ipc_change'] = float(ipc_change)

        # Rolling average IPC (window N_ROLLING_SAMPLES)
        data['ipc_avg_5'] = float(
            df['ipc'].rolling(window=N_ROLLING_SAMPLES, min_periods=1).mean().iloc[-1]
        )

        # Rolling average L3 miss rate
        data['l3_miss_rate_avg_5'] = float(
            df['l3_miss_rate'].rolling(window=N_ROLLING_SAMPLES, min_periods=1).mean().iloc[-1]
        )

        # Bandwidth utilization (normalized)
        bw = data['memory_bandwidth']
        # Update max observed (avoid zero)
        self.max_bw_observed = max(self.max_bw_observed, bw, 1e-3)
        bw_util = bw / self.max_bw_observed
        # Clip to [0,1.5] to avoid crazy values if spikes
        data['bw_util'] = float(min(max(bw_util, 0.0), 1.5))

        return data

    # ---------- Data Collection + DVFS Logic ----------
    def collect_data(self):
        """Collect raw metrics, engineer features, and run MLP decision."""

        pcm_data = self.read_pcm_data()
        power_data = self.read_power_gadget_data()

        # L2/L3 miss rates
        total_l2 = pcm_data['l2_cache_hits'] + pcm_data['l2_cache_misses']
        total_l3 = pcm_data['l3_cache_hits'] + pcm_data['l3_cache_misses']
        l2_miss_rate = pcm_data['l2_cache_misses'] / total_l2 if total_l2 > 0 else 0.0
        l3_miss_rate = pcm_data['l3_cache_misses'] / total_l3 if total_l3 > 0 else 0.0

        data = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage_overall': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().percent,
            'ipc': pcm_data['ipc'],
            'l2_miss_rate': round(l2_miss_rate, 4),
            'l3_miss_rate': round(l3_miss_rate, 4),
            'memory_bandwidth': float(pcm_data['memory_bandwidth']),
            'cpu_power': round(power_data['cpu_power'], 1),
            'cpu_temperature': round(power_data['cpu_temperature'], 1),
            'cpu_frequency': round(power_data['cpu_frequency'], 1)
        }

        # Feature engineering (adds ipc_change, ipc_avg_5, l3_miss_rate_avg_5, bw_util)
        data = self.feature_engineer_data(data)

        # Apply DVFS logic
        if USE_MLP_DVFS:
            features_for_mlp = {
                "ipc": data['ipc'],
                "l2_miss_rate": data['l2_miss_rate'],
                "l3_miss_rate": data['l3_miss_rate'],
                "memory_bandwidth": data['memory_bandwidth'],
                "cpu_usage_overall": data['cpu_usage_overall'],
                "cpu_temperature": data['cpu_temperature'],
                "cpu_power": data['cpu_power'],
                "cpu_frequency": data['cpu_frequency'],
                "ipc_change": data['ipc_change'],
                "ipc_avg_5": data['ipc_avg_5'],
                "l3_miss_rate_avg_5": data['l3_miss_rate_avg_5'],
                "bw_util": data['bw_util'],
            }

            level = mlp_controller.predict_level(features_for_mlp)
            applied = self.set_power_plan(level)
        else:
            # Baseline: fixed "Medium"
            level = "Medium"
            applied = self.set_power_plan("Medium")

        data['dvfs_level'] = level
        data['dvfs_applied'] = applied
        return data

    # ---------- Workload Control ----------
    def start_workload(self, script_name, num_cores):
        # Pure MLP, no manual bias now
        if USE_MLP_DVFS:
            print(f"💡 MLP-based DVFS active for {script_name}")
        else:
            print(f"💡 Baseline fixed-Medium mode for {script_name}")

        try:
            print(f"🚀 Starting {script_name} on {num_cores} core(s)...")
            subprocess.Popen([sys.executable, script_name])
            print("✅ Workload started.")
            return True
        except Exception as e:
            print(f"❌ Failed to start workload: {e}")
            return False

    # ---------- Monitoring Loop ----------
    def start_monitoring(self, workload_script, num_cores, duration):
        print(f"🚀 Starting monitoring for {duration} seconds...")
        print(f"📊 Log: {LOG_FILE}")
        print(f"🔧 DVFS Mode: {'MLP-DRIVEN DVFS' if USE_MLP_DVFS else 'Baseline (Fixed Medium)'}")
        print("=" * 80)

        start_time = time.time()
        with open(LOG_FILE, 'w') as f:
            iteration = 0
            while self.running and not STOP_REQUESTED and (time.time() - start_time < duration):
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
                              f"Level: {level:6} | IPC: {ipc:4.2f} | "
                              f"Power: {power:5.1f}W | Temp: {temp:4.1f}°C | "
                              f"CPU: {cpu:5.1f}% | L3 Miss: {l3_miss:5.3f}")

                    time.sleep(3)

                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    time.sleep(3)

        print("⏹️ Workload stopped.")
        print("✅ Experiment completed. Results logged.")

    # ---------- Plotting ----------
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
                ax = self.axs[idx // 2, idx % 2]
                ax.plot(df['time_min'], df[col], color=color, alpha=0.8, linewidth=1.5)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

    def start_realtime_plot(self):
        print("📊 Starting real-time visualization...")
        ani = FuncAnimation(self.fig, self.update_plots,
                            interval=3000, blit=False, cache_frame_data=False)
        plt.show()


# -----------------------------------------------------
# 3. Main Entry Point
# -----------------------------------------------------
def main():
    print("MLP-Driven DVFS Monitor")
    print("=" * 60)

    # Discover workloads
    scripts = []
    if os.path.exists("cpu_intensive.py"):
        print(" 1. cpu_intensive.py")
        scripts.append("cpu_intensive.py")
    if os.path.exists("memory_intensive.py"):
        print(" 2. memory_intensive.py")
        scripts.append("memory_intensive.py")
    if os.path.exists("mixed_workload.py"):
        print(" 3. mixed_workload.py")
        scripts.append("mixed_workload.py")    

    if not scripts:
        print("❌ No workload scripts found. Place cpu_intensive.py or memory_intensive.py here.")
        return

    # Workload choice
    while True:
        choice = input(f"Enter workload choice (1-{len(scripts)}): ").strip()
        if choice in [str(i + 1) for i in range(len(scripts))]:
            workload_script = scripts[int(choice) - 1]
            break
        print(f"Please enter 1 to {len(scripts)}.")

    # Core count
    while True:
        num_cores_input = input("Enter number of cores (1-8) [4]: ").strip()
        if not num_cores_input:
            num_cores = 4
            break
        if num_cores_input.isdigit() and 1 <= int(num_cores_input) <= 8:
            num_cores = int(num_cores_input)
            break
        print("Please enter a number between 1 and 8.")

    # Duration
    while True:
        duration_input = input(f"Enter duration in seconds {DURATION_OPTIONS} [{DURATION_OPTIONS[0]}]: ").strip()
        if not duration_input:
            duration = DURATION_OPTIONS[0]
            break
        if duration_input.isdigit() and int(duration_input) in DURATION_OPTIONS:
            duration = int(duration_input)
            break
        print(f"Please enter one of: {DURATION_OPTIONS}")

    monitor = MLPDVFSController()

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
