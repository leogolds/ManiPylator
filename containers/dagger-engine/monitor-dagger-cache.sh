#!/bin/bash

# Monitor Dagger cache usage and GC behavior

echo "Dagger Cache & Disk Usage Monitor"
echo "=================================="
echo

# Show current disk usage
echo "Current disk usage:"
df -h /
echo

# Show Dagger engine configuration
echo "Dagger GC Configuration:"
if [ -f ~/.config/dagger/engine.json ]; then
    cat ~/.config/dagger/engine.json | jq '.gc // "No GC config found"'
else
    echo "No engine.json found at ~/.config/dagger/engine.json"
fi
echo

# Check if Dagger engine is running
echo "Dagger Engine Status:"
if docker ps --filter "name=dagger-engine-*" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q dagger-engine; then
    docker ps --filter "name=dagger-engine-*" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
    echo "No Dagger engines currently running"
fi
echo

# Show Docker system usage (includes Dagger cache)
echo "Docker System Usage (includes Dagger cache):"
docker system df
echo

# Monitor function
monitor_loop() {
    echo "Starting continuous monitoring (Ctrl+C to stop)..."
    echo "Checking every 30 seconds..."
    echo
    
    while true; do
        echo "$(date): Disk: $(df -h / | tail -1 | awk '{print $4}') free of $(df -h / | tail -1 | awk '{print $2}') total"
        sleep 5
    done
}

monitor_loop