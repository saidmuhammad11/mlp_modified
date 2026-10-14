# integrated_workload_monitor.py
import subprocess
import threading
import time
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import psutil
from collections import deque
import json

class IntegratedWorkloadMonitor:
    def __init__(self):
        self.workload_process = None
        self.running = False
        self.data_buffer = deque(maxlen=200)
        self.pcm_path = self.find_pcm_path()
        self.power_gadget_path = self.find_power_gadget_path()

    def find_pcm_path(self):
        """Find Intel PCM installation path"""
        possible_paths = [
            r"C:\Program Files\Intel\PCM\bin\pcm.exe",
            r"C:\Program Files (x86)\Intel\PCM\bin\pcm.exe",
            r"C:\Intel\PCM\bin\pcm.exe",
            r"C:\Intel\PCM\pcm.exe",
            "pcm.exe"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found PCM at: {path}")
                return path
        print("⚠️  PCM not found. Using simulated data.")
        return None

    def find_power_gadget_path(self):
        """Find Intel Power Gadget installation path"""
        possible_paths = [
            r"C:\Program Files\Intel\Power Gadget 3.0\PowerLog3.0.exe",
            r"C:\Program Files (x86)\Intel\Power Gadget 3.0\PowerLog3.0.exe",
            r"C:\Program Files\Intel\Power Gadget\PowerLog.exe",
            r"C:\Program Files (x86)\Intel\Power Gadget\PowerLog.exe",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found Power Gadget at: {path}")
                return path
        print("⚠️  Power Gadget not found. Using simulated data.")
        return None

    def start_workload(self, workload_script, num_cores):
        """Start the selected workload script"""
        if not os.path.exists(workload_script):
            print(f"❌ Workload script not found: {workload_script}")
            return False

        try:
            print(f"🚀 Starting {workload_script} on {num_cores} core(s)...")
            self.workload_process = subprocess.Popen(
                [sys.executable, workload_script],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            print("✅ Workload started.")
            return True
        except Exception as e:
            print(f"❌ Failed to start workload: {e}")
            return False

    def stop_workload(self):
        """Stop the running workload"""
        if self.workload_process:
            try:
                self.workload_process.terminate()
                self.workload_process.wait(timeout=5)
                print("⏹️  Workload stopped.")
            except subprocess.TimeoutExpired:
                self.workload_process.kill()
                print("⏹️  Workload forcibly killed.")

    def read_pcm_data(self):
        """Read data from Intel PCM"""
        if not self.pcm_path:
            return self.simulate_pcm_data()

        try:
            cmd = [self.pcm_path, "-i", "1", "-nc", "-csv"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return self.simulate_pcm_data()

            data_line = lines[-1].split(',')
            if len(data_line) < 9:
                return self.simulate_pcm_data()

            return {
                'ipc': float(data_line[2]) if data_line[2].replace('.', '').isdigit() else 0.0,
                'l2_cache_hits': int(data_line[3]) if data_line[3].isdigit() else 0,
                'l2_cache_misses': int(data_line[4]) if data_line[4].isdigit() else 0,
                'l3_cache_hits': int(data_line[5]) if data_line[5].isdigit() else 0,
                'l3_cache_misses': int(data_line[6]) if data_line[6].isdigit() else 0,
                'memory_bandwidth': float(data_line[7]) if data_line[7].replace('.', '').isdigit() else 0.0,
                'power_consumption': float(data_line[8]) if data_line[8].replace('.', '').isdigit() else 0.0
            }
        except Exception as e:
            print(f"PCM Error: {e}")
            return self.simulate_pcm_data()

    def simulate_pcm_data(self):
        """Simulate PCM data if tool not available"""
        return {
            'ipc': np.random.uniform(0.5, 2.0),
            'l2_cache_hits': np.random.randint(50000, 80000),
            'l2_cache_misses': np.random.randint(1000, 4000),
            'l3_cache_hits': np.random.randint(30000, 60000),
            'l3_cache_misses': np.random.randint(500, 3000),
            'memory_bandwidth': round(np.random.uniform(10000, 25000), 1),
            'power_consumption': np.random.uniform(30, 70)
        }

    def read_power_gadget_data(self):
        """Read power/frequency/temp from Intel Power Gadget"""
        if not self.power_gadget_path:
            return self.simulate_power_data()

        try:
            log_file = "temp_power.csv"
            cmd = [self.power_gadget_path, "-duration", "1", "-file", log_file]
            subprocess.run(cmd, capture_output=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)

            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        data = lines[-1].strip().split(',')
                        power_data = {
                            'cpu_power': float(data[1]) if len(data) > 1 and data[1] else 0,
                            'gpu_power': float(data[2]) if len(data) > 2 and data[2] else 0,
                            'cpu_temperature': float(data[3]) if len(data) > 3 and data[3] else 0,
                            'cpu_frequency': float(data[4]) if len(data) > 4 and data[4] else 0
                        }
                        os.remove(log_file)
                        return power_data
        except Exception as e:
            print(f"Power Gadget Error: {e}")

        return self.simulate_power_data()

    def simulate_power_data(self):
        return {
            'cpu_power': np.random.uniform(25, 65),
            'gpu_power': np.random.uniform(5, 20),
            'cpu_temperature': np.random.uniform(50, 85),
            'cpu_frequency': np.random.uniform(1800, 3600)
        }

    def collect_data(self):
        """Collect all metrics"""
        timestamp = datetime.now()
        pcm_data = self.read_pcm_data()
        power_data = self.read_power_gadget_data()

        # Calculate derived metrics
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
            'memory_bandwidth': pcm_data['memory_bandwidth'],
            'cpu_power': power_data['cpu_power'],
            'cpu_temperature': power_data['cpu_temperature'],
            'cpu_frequency': power_data['cpu_frequency']
        }

        return data

    def classify_performance_level(self, data):
        """Classify performance level using IPC and miss rate"""
        ipc = data['ipc']
        l3_miss = data['l3_miss_rate']

        if ipc >= 1.5 or l3_miss > 0.15:
            return "High"
        elif ipc > 0.7:
            return "Medium"
        else:
            return "Low"

    def adjust_power_policy(self, level):
        """Adjust Windows power policy based on performance level"""
        policies = {
            "High": "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",    # High performance
            "Medium": "381b4222-f694-41f0-9685-ff5bb260df2e",  # Balanced
            "Low": "a1841308-3541-4fab-bc81-f71556f20b4a"     # Power saver
        }
        try:
            cmd = ["powercfg", "/setactive", policies[level]]
            subprocess.run(cmd, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            return True
        except Exception as e:
            print(f"Failed to set power policy: {e}")
            return False

    def start_monitoring(self, workload_script, num_cores, duration):
        """Main monitoring loop"""
        self.running = True
        LOG_FILE = f"monitoring_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        print(f"🚀 Starting monitoring for {duration} seconds...")
        print(f"📊 Log: {LOG_FILE}")
        print(f"🔧 PCM: {'Yes' if self.pcm_path else 'No (simulated)'}")
        print(f"🔧 Power Gadget: {'Yes' if self.power_gadget_path else 'No (simulated)'}")
        print(f"🔧 Workload: {workload_script} on {num_cores} core(s)")
        print("=" * 80)

        start_time = time.time()
        with open(LOG_FILE, 'w') as f:
            iteration = 0
            while self.running and (time.time() - start_time < duration):
                try:
                    iteration += 1
                    data = self.collect_data()
                    level = self.classify_performance_level(data)
                    # applied = self.adjust_power_policy(level)

                    # Add DVFS decision
                    data['dvfs_level'] = level
                    # data['dvfs_applied'] = applied

                    # Log & buffer
                    self.data_buffer.append(data)
                    f.write(json.dumps(data) + '\n')
                    f.flush()

                    # Terminal output
                    print(f"[{iteration:3d}] {datetime.now().strftime('%H:%M:%S')} | "
                          f"Level: {level:6s} | "
                          f"IPC: {data['ipc']:5.2f} | "
                          f"Power: {data['cpu_power']:5.1f}W | "
                          f"Temp: {data['cpu_temperature']:4.1f}°C | "
                          f"CPU: {data['cpu_usage_overall']:5.1f}% | "
                          f"L3 Miss: {data['l3_miss_rate']:6.3f}")

                    time.sleep(3)

                except Exception as e:
                    print(f"⚠️  Error: {e}")
                    time.sleep(1)

        print(f"✅ Monitoring complete. Data saved to {LOG_FILE}")
        return LOG_FILE


class RealTimeVisualizer:
    def __init__(self, monitor):
        self.monitor = monitor
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('Real-time Intel Power Gadget & PCM DVFS Monitoring', fontsize=16)
        self.setup_plots()

    def setup_plots(self):
        """Setup all subplot configurations"""
        self.axes[0, 0].set_title('CPU Power & Temperature')
        self.axes[0, 0].set_ylabel('Power (W) / Temp (°C)')
        self.axes[0, 1].set_title('IPC & Performance Level')
        self.axes[0, 1].set_ylabel('IPC')
        self.axes[0, 2].set_title('Cache Miss Rates')
        self.axes[0, 2].set_ylabel('Miss Rate')
        self.axes[1, 0].set_title('CPU Usage per Core')
        self.axes[1, 0].set_ylabel('CPU Usage (%)')
        self.axes[1, 1].set_title('CPU Frequency')
        self.axes[1, 1].set_ylabel('Frequency (MHz)')
        self.axes[1, 2].set_title('Power vs Performance')
        self.axes[1, 2].set_xlabel('IPC')
        self.axes[1, 2].set_ylabel('Power (W)')
        for ax in self.axes.flat:
            ax.grid(True, alpha=0.3)

    def update_plots(self, frame):
        """Update all plots with new data"""
        if len(self.monitor.data_buffer) < 2:
            return

        for ax in self.axes.flat:
            ax.clear()
        self.setup_plots()

        data_list = list(self.monitor.data_buffer)
        times = list(range(len(data_list)))

        # 1. CPU Power & Temperature
        power_data = [d['cpu_power'] for d in data_list]
        temp_data = [d['cpu_temperature'] for d in data_list]
        self.axes[0, 0].plot(times, power_data, 'r-', label='CPU Power (W)', linewidth=2)
        self.axes[0, 0].plot(times, temp_data, 'b-', label='CPU Temp (°C)', linewidth=2)
        self.axes[0, 0].legend()

        # 2. IPC & Performance Level
        ipc_data = [d['ipc'] for d in data_list]
        levels = [d['dvfs_level'] for d in data_list]
        level_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
        self.axes[0, 1].plot(times, ipc_data, 'g-', label='IPC', linewidth=2)
        for i, level in enumerate(levels):
            self.axes[0, 1].scatter(i, ipc_data[i], c=level_colors.get(level, 'gray'), s=30)
        self.axes[0, 1].legend()

        # 3. Cache Miss Rates
        l2_miss = [d['l2_miss_rate'] for d in data_list]
        l3_miss = [d['l3_miss_rate'] for d in data_list]
        self.axes[0, 2].plot(times, l2_miss, 'purple', label='L2 Miss Rate', linewidth=2)
        self.axes[0, 2].plot(times, l3_miss, 'brown', label='L3 Miss Rate', linewidth=2)
        self.axes[0, 2].legend()

        # 4. CPU Usage per Core
        if 'cpu_usage_per_core' in data_list[0]:
            core_count = len(data_list[0]['cpu_usage_per_core'])
            for core in range(min(core_count, 8)):
                usage = [d['cpu_usage_per_core'][core] for d in data_list]
                self.axes[1, 0].plot(times, usage, label=f'Core {core}', linewidth=1)
        self.axes[1, 0].legend()

        # 5. CPU Frequency
        freq_data = [d['cpu_frequency'] for d in data_list]
        self.axes[1, 1].plot(times, freq_data, 'navy', label='CPU Frequency (MHz)', linewidth=2)
        self.axes[1, 1].legend()

        # 6. Power vs Performance
        self.axes[1, 2].scatter(ipc_data, power_data, c=[level_colors.get(d['dvfs_level'], 'gray') for d in data_list],
                               s=50, alpha=0.6)
        self.axes[1, 2].set_xlabel('IPC')
        self.axes[1, 2].set_ylabel('Power (W)')

        plt.tight_layout()

    def start_visualization(self):
        """Start real-time visualization"""
        ani = animation.FuncAnimation(self.fig, self.update_plots, interval=3000, blit=False)
        plt.show()


def main():
    print("Integrated Workload & Real-Time Monitor")
    print("=" * 60)

    # Get user input interactively
    try:
        print("Available workloads in current folder:")
        scripts = []
        if os.path.exists("cpu_intensive.py"):
            print("  1. cpu_intensive.py")
            scripts.append("cpu_intensive.py")
        if os.path.exists("memory_intensive.py"):
            print("  2. memory_intensive.py")
            scripts.append("memory_intensive.py")

        if not scripts:
            print("❌ No workload scripts found. Please place cpu_intensive.py or memory_intensive.py in this folder.")
            return

        # Select workload
        while True:
            choice = input(f"\nEnter workload choice (1-{len(scripts)}): ").strip()
            if choice in [str(i+1) for i in range(len(scripts))]:
                workload_script = scripts[int(choice) - 1]
                break
            print(f"Please enter 1 to {len(scripts)}.")

        # Number of cores
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
            duration_input = input("Enter duration in seconds (100, 200, 300) [100]: ").strip()
            if not duration_input:
                duration = 100
                break
            if duration_input in ["100", "200", "300"]:
                duration = int(duration_input)
                break
            print("Please enter 100, 200, or 300.")

    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
        return

    # Run experiment
    monitor = IntegratedWorkloadMonitor()

    if not monitor.start_workload(workload_script, num_cores):
        print("❌ Could not start workload. Exiting.")
        return

    monitoring_thread = threading.Thread(
        target=monitor.start_monitoring,
        args=(workload_script, num_cores, duration),
        daemon=True
    )
    monitoring_thread.start()

    time.sleep(2)

    try:
        print("\n📊 Starting real-time visualization...")
        visualizer = RealTimeVisualizer(monitor)
        visualizer.start_visualization()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.running = False
        monitor.stop_workload()

    print(f"\n✅ Experiment completed. Results logged.")


if __name__ == "__main__":
    main()