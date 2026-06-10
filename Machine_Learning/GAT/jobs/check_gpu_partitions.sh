#!/bin/bash

echo "GPU partition summary"
echo "========================================================================================================================"
printf "%-12s %-6s %-10s %-24s %-6s %-6s %-6s %-6s %-6s %-6s %-8s %-8s\n" \
    "PARTITION" "AVAIL" "LIMIT" "GPU_TYPE" "GPU/N" "NODES" "IDLE" "MIX" "ALLOC" "BAD" "RUNNING" "PENDING"
echo "------------------------------------------------------------------------------------------------------------------------"

for p in $(sinfo -h -o "%P" | sed 's/*//' | grep -E 'gpu|nv|na|h20|dgx' | sort -u); do

    avail=$(sinfo -h -p "$p" -o "%a" | head -n 1)
    timelimit=$(sinfo -h -p "$p" -o "%l" | head -n 1)

    gpu_info=$(sinfo -N -h -p "$p" -o "%G" | head -n 1)
    gpu_type=$(echo "$gpu_info" | sed -E 's/gpu:([^:]+):.*/\1/')
    gpu_per_node=$(echo "$gpu_info" | awk -F: '{print $3}')

    nodes=$(sinfo -h -p "$p" -o "%D" | awk '{sum+=$1} END{print sum+0}')

    idle=$(sinfo -h -p "$p" -t idle -o "%D" 2>/dev/null | awk '{sum+=$1} END{print sum+0}')
    mix=$(sinfo -h -p "$p" -t mix -o "%D" 2>/dev/null | awk '{sum+=$1} END{print sum+0}')
    alloc=$(sinfo -h -p "$p" -t alloc -o "%D" 2>/dev/null | awk '{sum+=$1} END{print sum+0}')

    down=$(sinfo -h -p "$p" -t down -o "%D" 2>/dev/null | awk '{sum+=$1} END{print sum+0}')
    drain=$(sinfo -h -p "$p" -t drain -o "%D" 2>/dev/null | awk '{sum+=$1} END{print sum+0}')
    maint=$(sinfo -h -p "$p" -t maint -o "%D" 2>/dev/null | awk '{sum+=$1} END{print sum+0}')
    bad=$((down + drain + maint))

    running=$(squeue -h -p "$p" -t R | wc -l)
    pending=$(squeue -h -p "$p" -t PD | wc -l)

    printf "%-12s %-6s %-10s %-24s %-6s %-6s %-6s %-6s %-6s %-6s %-8s %-8s\n" \
        "$p" "$avail" "$timelimit" "$gpu_type" "$gpu_per_node" "$nodes" "$idle" "$mix" "$alloc" "$bad" "$running" "$pending"
done

echo "========================================================================================================================"

echo
echo "Detailed GPU node view"
echo "========================================================================================================================"
printf "%-12s %-8s %-10s %-24s %-16s %-10s\n" \
    "PARTITION" "NODE" "STATE" "GPU" "CPU(A/I/O/T)" "MEM(MB)"
echo "------------------------------------------------------------------------------------------------------------------------"

sinfo -N -h -o "%P %N %t %G %C %m" \
    | sed 's/*//' \
    | grep -E "gpu|nv|na|h20|dgx" \
    | sort \
    | awk '{printf "%-12s %-8s %-10s %-24s %-16s %-10s\n", $1, $2, $3, $4, $5, $6}'

echo "========================================================================================================================"
