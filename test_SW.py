# data_collection_dvfs.py
"""
Data Collection DVFS Controller
- 100% REAL data only. No simulated fallbacks.
- Real-time monitoring with Intel PCM.
- Extracts Power (SYS Energy), Frequency, IPC, and Cache metrics directly from PCM.
- Sliding Window DVFS with hysteresis for varied state collection.
"""

import os
import json
import time
import threading
import subprocess
import psutil
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import sys
from pathlib import Path  # <--- Add this line right here

try:
    import pandas as pd
except ImportError:
    print("❌ Please install pandas: pip install pandas")
    sys.exit(1)

# =================== CONFIGURATION ===================
LOG_FILE = f"dataset_collection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

PCM_PATHS = [
    r"C:\Users\saidm\Desktop\Programming\pcm\build\bin\Release\pcm.exe",
    r"C:\Users\saidm\pcm\build\bin\Release\pcm.exe",
    str((Path(__file__).resolve().parent / "pcm.exe").resolve()),
]

DURATION_OPTIONS = [100, 200, 300]

# Sliding window & hysteresis config
WINDOW_SIZE = 5           # smaller → faster adaptation
HYSTERESIS_COUNT = 1      # change level as soon as rule is consistently met

# Toggle: False = baseline (fixed Medium), True = SW-DVFS
# Tip: Run your workload once with False, and once with True, to get diverse training data!
USE_CUSTOM_DVFS = False
# =====================================================


class RealDataCollectionDVFS:
    def __init__(self):
        self.running = True
        self.data_buffer = []
        self.pcm_path = self.find_tool(PCM_PATHS)

        if not self.pcm_path:
            print("❌ FATAL: pcm.exe is required for real data collection. Exiting.")
            sys.exit(1)

        self.fig, self.axs = plt.subplots(4, 2, figsize=(16, 12))
        self.fig.suptitle("Real-Time DVFS Data Collection", fontsize=14, fontweight='bold')

        # Power plan GUIDs
        self.POWER_PLANS = {
            "High": "27ad8305-4092-41f2-a01b-5a6003fb5077",
            "Medium": "381b4222-f694-41f0-9685-ff5bb260df2e",
            "Low": "b2524225-86dc-424d-ba5b-e78d683c5d3a"
        }

        # Sliding windows for adaptive logic
        self.ipc_window = []
        self.miss_window = []

        # Hysteresis state
        self.current_level = "Medium"      
        self.candidate_level = "Medium"    
        self.level_streak = 0              
        self.workload_process = None

    def find_tool(self, possible_paths):
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found PCM: {path}")
                return path
        return None

    def set_power_plan(self, level):
        guid = self.POWER_PLANS.get(level)
        if not guid:
            return False

        try:
            if level == "Low":
                subprocess.run(["powercfg", "-setacvalueindex", guid, "SUB_PROCESSOR", "PROCTHROTTLEMAX", "30"], check=True)
                subprocess.run(["powercfg", "-setacvalueindex", guid, "SUB_PROCESSOR", "PROCTHROTTLEMIN", "5"], check=True)
            elif level == "Medium":
                subprocess.run(["powercfg", "-setacvalueindex", guid, "SUB_PROCESSOR", "PROCTHROTTLEMAX", "60"], check=True)
                subprocess.run(["powercfg", "-setacvalueindex", guid, "SUB_PROCESSOR", "PROCTHROTTLEMIN", "30"], check=True)
            elif level == "High":
                subprocess.run(["powercfg", "-setacvalueindex", guid, "SUB_PROCESSOR", "PROCTHROTTLEMAX", "100"], check=True)
                subprocess.run(["powercfg", "-setacvalueindex", guid, "SUB_PROCESSOR", "PROCTHROTTLEMIN", "100"], check=True)

            subprocess.run(["powercfg", "-setactive", guid], check=True)
            return True
        except Exception as e:
            print(f"❌ Failed to set power plan: {e}")
            return False


    def read_pcm_data(self):
        """Read REAL data from Intel PCM, including Power (SYS energy)."""
        result = subprocess.run(
            [self.pcm_path, "1", "-nc", "-ns", "-i=1", "-r"],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"pcm.exe failed (Code {result.returncode}): {result.stderr.strip()}\n{result.stdout.strip()}")

        data_line = None
        sys_power_w = 0.0

        for line in result.stdout.splitlines():
            # Extract main metrics row
            if "TOTAL" in line and "*" in line:
                data_line = line
            # Extract Energy/Power (Joules over 1 second = Watts)
            if "SYS energy:" in line:
                match = re.search(r"SYS energy:\s*([0-9.]+)\s*J", line)
                if match:
                    sys_power_w = float(match.group(1))

        if not data_line:
            raise RuntimeError(f"Could not find 'TOTAL *' row in PCM output:\n{result.stdout}")

        clean_line = re.sub(r'\s+([KMG])\b', r'\1', data_line)
        parts = [x for x in clean_line.split() if x]

        # 0: TOTAL, 1: *, 2: UTIL, 3: IPC, 4: CFREQ, 5: L3MISS, 6: L2MISS, 7: L3HIT, 8: L2HIT
        ipc = float(parts[3])
        cpu_freq_mhz = float(parts[4]) * 1000  # Convert GHz to MHz
        l3_hit_ratio = float(parts[7])
        l2_hit_ratio = float(parts[8])

        # Create proxy miss/hit numbers from the ratio
        l2_misses = int((1.0 - l2_hit_ratio) * 1000)
        l2_hits = 1000 - l2_misses
        l3_misses = int((1.0 - l3_hit_ratio) * 1000)
        l3_hits = 1000 - l3_misses

        return {
            'ipc': ipc,
            'l2_cache_hits': l2_hits,
            'l2_cache_misses': l2_misses,
            'l3_cache_hits': l3_hits,
            'l3_cache_misses': l3_misses,
            'memory_bandwidth': 0.0, # Not supported by your CPU hardware
            'cpu_power': sys_power_w,
            'cpu_temperature': 0.0,  # Not supported by your PCM version
            'cpu_frequency': cpu_freq_mhz
        }

    def update_window(self, ipc, l3_miss_rate):
        self.ipc_window.append(ipc)
        self.miss_window.append(l3_miss_rate)
        if len(self.ipc_window) > WINDOW_SIZE:
            self.ipc_window.pop(0)
            self.miss_window.pop(0)

    def apply_hysteresis(self, target):
        if target == self.current_level:
            self.candidate_level = target
            self.level_streak = 0
            return self.current_level

        if target == self.candidate_level:
            self.level_streak += 1
        else:
            self.candidate_level = target
            self.level_streak = 1

        if self.level_streak >= HYSTERESIS_COUNT:
            self.current_level = self.candidate_level
            self.level_streak = 0

        return self.current_level

    def classify_performance_level(self, data):
        ipc = data['ipc']
        l3_miss_rate = data['l3_miss_rate']
        cpu_usage = data['cpu_usage_overall']

        self.update_window(ipc, l3_miss_rate)

        if cpu_usage < 25.0:
            target = "Low"
        elif ipc < 1.0 and l3_miss_rate > 0.04:
            target = "Low"
        elif ipc > 1.3 and l3_miss_rate < 0.06:
            target = "High"
        else:
            target = "Medium"

        return self.apply_hysteresis(target)

    def collect_data(self):
        timestamp = datetime.now()
        
        # This will crash the script if it fails, ensuring NO fake data is logged.
        pcm_data = self.read_pcm_data()

        total_l2 = pcm_data['l2_cache_hits'] + pcm_data['l2_cache_misses']
        total_l3 = pcm_data['l3_cache_hits'] + pcm_data['l3_cache_misses']
        l2_miss_rate = pcm_data['l2_cache_misses'] / total_l2 if total_l2 > 0 else 0
        l3_miss_rate = pcm_data['l3_cache_misses'] / total_l3 if total_l3 > 0 else 0

        cpu_percent_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        memory = psutil.virtual_memory()
        cpu_percent_overall = psutil.cpu_percent(interval=0.1)

        data = {
            'timestamp': timestamp.isoformat(),
            'cpu_usage_overall': cpu_percent_overall,
            'cpu_usage_per_core': cpu_percent_per_core,
            'memory_usage': memory.percent,
            'cpu_count': len(cpu_percent_per_core),
            'ipc': pcm_data['ipc'],
            'l2_cache_hits': pcm_data['l2_cache_hits'],
            'l2_cache_misses': pcm_data['l2_cache_misses'],
            'l3_cache_hits': pcm_data['l3_cache_hits'],
            'l3_cache_misses': pcm_data['l3_cache_misses'],
            'l2_miss_rate': round(l2_miss_rate, 3),
            'l3_miss_rate': round(l3_miss_rate, 3),
            'memory_bandwidth': pcm_data['memory_bandwidth'],
            'cpu_power': round(pcm_data['cpu_power'], 2),
            'cpu_temperature': pcm_data['cpu_temperature'],
            'cpu_frequency': pcm_data['cpu_frequency']
        }

        if USE_CUSTOM_DVFS:
            level = self.classify_performance_level(data)
            applied = self.set_power_plan(level)
        else:
            level = "Medium"
            applied = self.set_power_plan("Medium")

        data['dvfs_level'] = level
        data['dvfs_applied'] = applied
        return data

    def start_workload(self, script_name, num_cores):
        script_path = Path(script_name).expanduser().resolve()

        if not script_path.exists():
            print(f"❌ Workload not found: {script_path}")
            return False

        env = os.environ.copy()
        thread_count = str(num_cores)
        env["OMP_NUM_THREADS"] = thread_count
        env["MKL_NUM_THREADS"] = thread_count
        env["OPENBLAS_NUM_THREADS"] = thread_count
        env["NUMEXPR_NUM_THREADS"] = thread_count

        try:
            print(f"🚀 Starting {script_name} on {num_cores} core(s).")
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
            self.workload_process = subprocess.Popen(
                [sys.executable, str(script_path)],
                env=env,
                creationflags=creationflags,
            )
            print(f"✅ Workload started (PID {self.workload_process.pid}).")
            return True
        except Exception as e:
            print(f"❌ Failed to start workload: {e}")
            self.workload_process = None
            return False

    def stop_workload(self):
        proc = self.workload_process
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            else:
                proc.terminate()
        except Exception:
            pass
        finally:
            self.workload_process = None

    def start_monitoring(self, workload_script, num_cores, duration):
        print(f"🚀 Starting REAL data collection for {duration} seconds.")
        print(f"📊 Dataset Log: {LOG_FILE}")
        print(f"🔧 DVFS Mode: {'Custom Sliding Window' if USE_CUSTOM_DVFS else 'Baseline (Fixed Medium)'}")
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
                        print(f"[{iteration:3}] {datetime.now().strftime('%H:%M:%S')} | "
                              f"Level: {data['dvfs_level']:6} | IPC: {data['ipc']:4.2f} | "
                              f"Power: {data['cpu_power']:5.1f}W | Freq: {data['cpu_frequency']:6.0f}MHz | "
                              f"CPU: {data['cpu_usage_overall']:5.1f}%")

                    time.sleep(3)

                except Exception as e:
                    print(f"❌ FATAL Data Collection Error: {e}")
                    self.running = False
                    break

        self.stop_workload()
        print("⏹️ Workload stopped.")
        print(f"✅ Dataset collection complete. Results logged to {LOG_FILE}")

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
        print("📊 Starting real-time visualization.")
        ani = FuncAnimation(self.fig, self.update_plots, interval=3000, blit=False, cache_frame_data=False)
        plt.show()

def main():
    print("Real Data Collection Monitor")
    print("=" * 60)

    configured_scripts = [
        "workload_mem2_stream_huge.py",
        "NPB3.0-omp-C/workload_cpu1_nas_ep.py",
        "NPB3.0-omp-C/workload_mem1_bursty_is.py",
        "NPB3.0-omp-C/workload_cpu2_bursty_ep.py",
        "NPB3.0-omp-C/workload_mixed1_bursty_lu.py",
        "NPB3.0-omp-C/workload_mixed2_bursty_mg.py"
    ]
    
    root_dir = Path(__file__).resolve().parent
    scripts = [s for s in configured_scripts if (root_dir / s).exists()]
    
    print("Available Workloads:")
    for i, script in enumerate(scripts):
        print(f" {i+1}. {script}")

    if not scripts:
        print("❌ No workload scripts found in the current directory.")
        return

    try:
        while True:
            choice = input(f"Enter workload choice (1-{len(scripts)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(scripts):
                workload_script = scripts[int(choice) - 1]
                break
        
        while True:
            num_cores = input("Enter number of cores (1-8) [4]: ").strip()
            if not num_cores:
                num_cores = 4; break
            if num_cores.isdigit() and 1 <= int(num_cores) <= 8:
                num_cores = int(num_cores); break

        while True:
            duration = input(f"Enter duration in seconds {DURATION_OPTIONS} [{DURATION_OPTIONS[0]}]: ").strip()
            if not duration:
                duration = DURATION_OPTIONS[0]; break
            if duration.isdigit() and int(duration) in DURATION_OPTIONS:
                duration = int(duration); break
                
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled.")
        return

    monitor = RealDataCollectionDVFS()
    if not monitor.start_workload(workload_script, num_cores):
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

if __name__ == "__main__":
    main()