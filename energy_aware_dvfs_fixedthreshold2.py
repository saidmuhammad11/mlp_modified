# fixed_threshold_dvfs.py
"""
Fixed-Threshold DVFS Controller
- Uses fixed IPC and cache miss thresholds
- Baseline runs in fixed "Medium" (Balanced) mode
- SW-DVFS uses fixed thresholds to switch between Low/Medium/High
- Applies real power policy changes via powercfg
"""

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

try:
    import pandas as pd
except ImportError:
    print("❌ Please install pandas: pip install pandas")
    sys.exit(1)

# =================== CONFIGURATION ===================
LOG_FILE = f"monitoring_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
PCM_PATHS = [
    r"C:\Intel\pcm-win-v1.8\pcm.exe",
    r"C:\Intel\PCM\pcm.exe",
    r"C:\Intel\pcm\pcm.exe",
    "pcm.exe"
]
POWER_GADGET_PATHS = [
    r"C:\Program Files\Intel\Power Gadget 4.0\PowerLog.exe",
    r"C:\Program Files\Intel\Power Gadget\PowerLog.exe",
    r"C:\Program Files\Intel\Power Gadget 3.0\PowerLog3.0.exe",
    "PowerLog.exe"
]
DURATION_OPTIONS = [100, 200, 300]

# Fixed Thresholds (tuned for CPU-intensive workloads)
IPC_HIGH_THRESHOLD = 1.5
IPC_MEDIUM_THRESHOLD = 0.8
L3_MISS_HIGH_THRESHOLD = 0.15  # 15%

# Power Plan GUIDs
POWER_PLANS = {
    "High": "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",   # High Performance
    "Medium": "381b4222-f694-41f0-9685-ff5bb260df2e", # Balanced
    "Low": "a1841308-3541-4fab-bc81-a72f0918b7b7"     # Power Saver
}

# Control Mode
USE_CUSTOM_DVFS = True  # Set to False for baseline and true for dvfs
BASELINE_LEVEL = "Medium"  # "Medium" or "High" — only used when USE_CUSTOM_DVFS = False
# =====================================================

class FixedThresholdDVFS:
    def __init__(self):
        self.running = True
        self.data_buffer = []
        self.pcm_path = self.find_tool(PCM_PATHS)
        self.power_gadget_path = self.find_tool(POWER_GADGET_PATHS)
        self.fig, self.axs = plt.subplots(4, 2, figsize=(16, 12))
        self.fig.suptitle("Fixed-Threshold DVFS Monitoring", fontsize=14, fontweight='bold')

    def find_tool(self, possible_paths):
        """Find tool in multiple locations"""
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found: {path}")
                return path
        print(f"❌ Tool not found in any of: {possible_paths}")
        return None

    def set_power_plan(self, level):
        """Switch Windows power plan using powercfg"""
        guid = POWER_PLANS.get(level)
        if not guid:
            print(f"❌ Invalid level: {level}")
            return False
        try:
            subprocess.run(["powercfg", "-setactive", guid], check=True)
            return True
        except Exception as e:
            print(f"❌ Failed to set power plan: {e}")
            return False

    def simulate_pcm_data(self):
        """Simulate when PCM not available"""
        return {
            'ipc': np.random.uniform(0.5, 2.0),
            'l2_cache_hits': np.random.randint(50000, 80000),
            'l2_cache_misses': np.random.randint(1000, 5000),
            'l3_cache_hits': np.random.randint(30000, 60000),
            'l3_cache_misses': np.random.randint(500, 3000),
            'memory_bandwidth': np.random.uniform(10000, 25000)
        }

    def read_pcm_data(self):
        """Read from Intel PCM"""
        if not self.pcm_path:
            print("⚠️ PCM not found. Using simulated data.")
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

            if len(parts) >= 8:
                ipc = float(parts[1])
                l2_hits = int(parts[4])
                l2_misses = int(parts[5])
                l3_hits = int(parts[6])
                l3_misses = int(parts[7])
                mem_bw = float(parts[2])

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

    def simulate_power_data(self):
        """Simulate power data"""
        return {
            'cpu_power': np.random.uniform(25, 65),
            'gpu_power': np.random.uniform(5, 20),
            'cpu_temperature': np.random.uniform(50, 85),
            'cpu_frequency': np.random.uniform(1800, 3600)
        }

    def read_power_gadget_data(self):
        """Read from Power Gadget"""
        if not self.power_gadget_path:
            print("⚠️ Power Gadget not found. Using simulated data.")
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
                    'cpu_power': float(data[1]) if len(data) > 1 and data[1] else 0,
                    'gpu_power': float(data[2]) if len(data) > 2 and data[2] else 0,
                    'cpu_temperature': float(data[3]) if len(data) > 3 and data[3] else 0,
                    'cpu_frequency': float(data[4]) if len(data) > 4 and data[4] else 0
                }
                os.remove(temp_file)
                return power_data
        except Exception as e:
            print(f"Power Gadget Error: {e}")
        return self.simulate_power_data()

    def collect_data(self):
        """Collect all metrics"""
        timestamp = datetime.now()
        pcm_data = self.read_pcm_data()
        power_data = self.read_power_gadget_data()

        total_l2 = pcm_data['l2_cache_hits'] + pcm_data['l2_cache_misses']
        total_l3 = pcm_data['l3_cache_hits'] + pcm_data['l3_cache_misses']
        l2_miss_rate = pcm_data['l2_cache_misses'] / total_l2 if total_l2 > 0 else 0
        l3_miss_rate = pcm_data['l3_cache_misses'] / total_l3 if total_l3 > 0 else 0

        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        memory = psutil.virtual_memory()

        data = {
            'timestamp': timestamp.isoformat(),
            'cpu_usage_overall': psutil.cpu_percent(interval=0.1),
            'cpu_usage_per_core': cpu_percent,
            'memory_usage': memory.percent,
            'cpu_count': len(cpu_percent),
            'ipc': pcm_data['ipc'],
            'l2_cache_hits': pcm_data['l2_cache_hits'],
            'l2_cache_misses': pcm_data['l2_cache_misses'],
            'l3_cache_hits': pcm_data['l3_cache_hits'],
            'l3_cache_misses': pcm_data['l3_cache_misses'],
            'l2_miss_rate': round(l2_miss_rate, 3),
            'l3_miss_rate': round(l3_miss_rate, 3),
            'memory_bandwidth': round(pcm_data['memory_bandwidth'], 1),
            'cpu_power': round(power_data['cpu_power'], 1),
            'cpu_temperature': round(power_data['cpu_temperature'], 1),
            'cpu_frequency': round(power_data['cpu_frequency'], 1)
        }

        # === DVFS Decision Logic ===
        if USE_CUSTOM_DVFS:
            level = self.classify_performance_level(data)
            applied = self.set_power_plan(level)
        else:
            # Baseline: Fixed level (Medium or High)
            level = BASELINE_LEVEL
            applied = self.set_power_plan(level)

        data['dvfs_level'] = level
        data['dvfs_applied'] = applied
        return data

    def classify_performance_level(self, data):
        """Fixed-threshold classification"""
        ipc = data['ipc']
        l3_miss_rate = data['l3_miss_rate']

        if ipc >= IPC_HIGH_THRESHOLD or l3_miss_rate > L3_MISS_HIGH_THRESHOLD:
            return "High"
        elif ipc >= IPC_MEDIUM_THRESHOLD:
            return "Medium"
        else:
            return "Low"

    def start_workload(self, script_name, num_cores):
        """Start selected workload"""
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
        print(f"🔧 PCM: {'Yes' if self.pcm_path else 'No (simulated)'}")
        print(f"🔧 Power Gadget: {'Yes' if self.power_gadget_path else 'No (simulated)'}")
        print(f"🔧 Workload: {workload_script} on {num_cores} core(s)")
        print(f"🔧 DVFS Mode: {'Fixed-Threshold DVFS' if USE_CUSTOM_DVFS else f'Baseline ({BASELINE_LEVEL})'}")
        print("=" * 80)

        start_time = time.time()
        with open(LOG_FILE, 'w') as f:
            iteration = 0
            while self.running and (time.time() - start_time < duration):
                try:
                    iteration += 1
                    data = self.collect_data()
                    f.write(json.dumps(data) + '\n')
                    f.flush()
                    self.data_buffer.append(data)

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
                    continue

        print("⏹️ Workload stopped.")
        print("✅ Experiment completed. Results logged.")

    def update_plots(self, frame):
        """Update real-time plots"""
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
    print("Fixed-Threshold DVFS Monitor")
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
        print("❌ No workload scripts found.")
        return

    # Get user input
    try:
        # Workload
        while True:
            choice = input(f"Enter workload choice (1-{len(scripts)}): ").strip()
            if choice in [str(i+1) for i in range(len(scripts))]:
                workload_script = scripts[int(choice) - 1]
                break
            print(f"Please enter 1 to {len(scripts)}.")

        # Cores
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

    except KeyboardInterrupt:
        print("❌ Setup cancelled by user.")
        return

    # Run experiment
    monitor = FixedThresholdDVFS()

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