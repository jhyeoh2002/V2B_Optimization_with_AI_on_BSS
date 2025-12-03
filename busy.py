import time
import argparse
import multiprocessing
import os
import signal
import sys

def cpu_worker(target_load, core_num):
    """
    Runs a loop that sleeps and works to match the target load percentage.
    """
    # Define a time slice for the duty cycle (e.g., 0.1 seconds)
    cycle_time = 0.1
    
    # Calculate busy time and sleep time
    busy_time = cycle_time * (target_load / 100.0)
    sleep_time = cycle_time * (1.0 - (target_load / 100.0))

    try:
        while True:
            start_time = time.time()
            
            # Busy loop: do math until busy_time is reached
            # We use time.time() checks which adds slight overhead, 
            # but allows for "percentage" control.
            while (time.time() - start_time) < busy_time:
                # Perform simple arithmetic to burn cycles without RAM
                _ = 2134 * 2134
            
            # Sleep for the remainder of the cycle
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        # Allow worker to exit cleanly on interrupt
        pass

def main():
    # Detect available CPU cores
    max_cores = os.cpu_count() or 1

    parser = argparse.ArgumentParser(description="Lightweight CPU Heater to prevent idle crashes.")
    
    parser.add_argument(
        '-c', '--cores', 
        type=int, 
        default=max_cores,
        help=f"Number of CPU cores to utilize (Default: {max_cores})"
    )
    
    parser.add_argument(
        '-l', '--load', 
        type=int, 
        default=20,
        choices=range(1, 101),
        metavar="[1-100]",
        help="Target CPU load percentage per core (Default: 20)"
    )

    parser.add_argument(
        '-m', '--memory',
        type=int,
        default=0,
        help="Amount of RAM to occupy in MB (Default: 0)"
    )

    args = parser.parse_args()

    # Input validation
    if args.cores > max_cores:
        print(f"Warning: You requested {args.cores} cores, but system only reports {max_cores}.")
    
    # Memory allocation (held by the main process)
    memory_hog = None
    if args.memory > 0:
        print(f"Allocating {args.memory} MB of RAM...")
        try:
            # Create a bytearray. Python allocates this on the heap.
            # We multiply by 1024*1024 to get MB.
            memory_hog = bytearray(args.memory * 1024 * 1024)
            # Touch the memory to ensure it's committed by the OS
            if len(memory_hog) > 0:
                memory_hog[-1] = 1
        except MemoryError:
            print(f"Error: Failed to allocate {args.memory} MB. System might be out of memory.")
            sys.exit(1)
        except Exception as e:
            print(f"Error allocating memory: {e}")
            sys.exit(1)

    print(f"--- CPU Heater Started ---")
    print(f"Workers: {args.cores}")
    print(f"Target Load: {args.load}%")
    if args.memory > 0:
        print(f"Memory Held: {args.memory} MB")
    print(f"Press CTRL+C to stop.")

    processes = []
    
    # handle exit signals gracefully
    def signal_handler(sig, frame):
        print("\nStopping workers...")
        for p in processes:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Spawn processes
    for i in range(args.cores):
        p = multiprocessing.Process(target=cpu_worker, args=(args.load, i))
        p.daemon = True # Ensures processes die if main script dies
        p.start()
        processes.append(p)

    # Keep main thread alive to listen for CTRL+C
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()