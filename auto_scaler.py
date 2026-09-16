import psutil
import time
import subprocess

workers = ["node1.spark.com", "node2.spark.com"]
active_workers = {worker: True for worker in workers}

def stop_worker(worker):
    print(f"Stopping worker on {worker}...")
    cmd = ["ssh", worker, "/usr/local/spark/sbin/stop-worker.sh"]
    subprocess.call(cmd)
    active_workers[worker] = False

def start_worker(worker):
    print(f"Starting worker on {worker}...")
    cmd = ["ssh", worker, "/usr/local/spark/sbin/start-worker.sh", "spark://master.spark.com:7077"]
    subprocess.call(cmd)
    active_workers[worker] = True

def stop_all_workers():
    print("[AutoScaler] Stopping all workers first...")
    for worker, is_active in active_workers.items():
        if is_active:
            stop_worker(worker)
    print("[AutoScaler] All workers have been stopped.")
    for worker in active_workers:
        active_workers[worker] = False

def scale_out():
    for worker, is_active in active_workers.items():
        if not is_active:
            start_worker(worker)
            return
    print("All workers are already active. No scale-out possible.")

def scale_in():
    active = [worker for worker, is_active in active_workers.items() if is_active]
    if len(active) > 1:
        worker_to_stop = active[0]
        stop_worker(worker_to_stop)
    else:
        print("Only one (or zero) worker active. No scale-in possible.")

def monitor_cpu():
    while True:
        cpu_usage = psutil.cpu_percent(interval=5)
        print(f"Current CPU usage: {cpu_usage}%")
        if cpu_usage > 60:
            scale_out()
        elif cpu_usage < 30:
            scale_in()
        time.sleep(5)

if __name__ == "__main__":
    stop_all_workers()
    monitor_cpu()
