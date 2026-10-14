#!/bin/bash
# monitor_pmc.sh

DURATION=130  # Slightly longer than workload (2 min 10 sec)
WINDOW=50     # ms → 0.05 sec
COUNT=$(echo "$DURATION * 1000 / $WINDOW" | bc -l | xargs printf "%.0f")

OUTPUT_FILE="pmc_data.csv"

# Header
echo "timestamp,ipc,cycles,instructions,l2d_cache_miss" > $OUTPUT_FILE

echo "Starting PMC monitoring for $DURATION seconds..."

for i in $(seq 1 $COUNT); do
    # Capture for exactly $WINDOW ms
    PERF_OUT=$(perf stat -e cycles,instructions,mem_inst_retired.all_loads -I $WINDOW -o /dev/stdout sleep 0.001 2>&1 | grep -E "cycles|instructions|all_loads")

    # Extract values (may be empty if no output)
    CYCLES=$(echo "$PERF_OUT" | grep "cycles" | awk '{print $1}' | tr -d ',')
    INST=$(echo "$PERF_OUT" | grep "instructions" | awk '{print $1}' | tr -d ',')
    L2_MISS=$(echo "$PERF_OUT" | grep "all_loads" | awk '{print $1}' | tr -d ',')

    # Skip if any metric missing
    if [[ -z "$CYCLES" || -z "$INST" || -z "$L2_MISS" ]]; then
        continue
    fi

    # Compute IPC
    if [[ "$CYCLES" =~ ^[0-9]+$ && "$INST" =~ ^[0-9]+ ]]; then
        IPC=$(echo "scale=3; $INST / $CYCLES" | bc -l)
    else
        IPC=0
    fi

    # Timestamp
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S.%3N")

    # Log to CSV
    echo "$TIMESTAMP,$IPC,$CYCLES,$INST,$L2_MISS" >> $OUTPUT_FILE

    # Optional: print live
    echo "[$TIMESTAMP] IPC=$IPC, L2_miss=$L2_MISS"
done

echo "Monitoring done. Data saved to $OUTPUT_FILE"