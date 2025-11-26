import os, faulthandler
import psutil, time

# Create a Process object for the current process to monitor its resource usage
proc = psutil.Process(os.getpid())

# Enable faulthandler to dump Python stack traces on fatal errors
faulthandler.enable()

# Set an environment variable to enable Python fault handler for kernel restarts
os.environ["PYTHONFAULTHANDLER"] = "1"

# Ensure the GRB_WLSACCESSID environment variable remains stable (no-op if not set)
os.environ["GRB_WLSACCESSID"] = os.environ.get("GRB_WLSACCESSID", "")

# Print a message indicating that crash diagnostics are enabled
print("Crash diagnostics ON")

def log_status(msg):
    """
    Logs the current status of the system, including memory usage, thread count, and system-wide memory usage.

    Parameters:
        msg (str): A custom message to include in the log.
    """
    # Log the provided message with a timestamp
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    # Log the memory usage of the current process (RSS and VMS)
    print(f"  RSS: {proc.memory_info().rss/1e9:.2f} GB, VMS: {proc.memory_info().vms/1e9:.2f} GB")

    # Log the number of threads used by the current process
    print(f"  Threads: {proc.num_threads()}, System memory used: {psutil.virtual_memory().used/1e9:.2f} GB")

    # Retrieve and log system-wide memory usage details
    vm = psutil.virtual_memory()
    print(f"  System memory used: {vm.used/1e9:.2f} GB / {vm.total/1e9:.2f} GB")