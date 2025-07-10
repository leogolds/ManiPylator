#!/bin/bash

# Example: Running Dagger engine as custom runner with configuration

# Stop any existing engines
docker rm -f $(docker ps -q --filter "name=dagger-engine-*") 2>/dev/null || true
# Ensure the config directory exists
mkdir -p ~/.config/dagger

# Remove existing engine.json if it exists
rm -f ~/.config/dagger/engine.json

# Copy the local engine.json to the config directory
cp engine.json ~/.config/dagger/engine.json

echo "Configuration file copied to ~/.config/dagger/engine.json"



echo "Starting Dagger engine as custom runner with GC configuration..."

# Run custom Dagger engine with mounted config
docker run --rm -d \
    -v /var/lib/dagger:/var/lib/dagger \
    -v ~/.config/dagger/engine.json:/etc/dagger/engine.json \
    --name dagger-engine-custom \
    --privileged \
    -p 1234:1234 \
    registry.dagger.io/engine:v0.18.12

echo "Custom engine started with configuration:"
cat ~/.config/dagger/engine.json
echo

# Wait a moment for engine to start
sleep 3

# Connect to the custom engine
export _EXPERIMENTAL_DAGGER_RUNNER_HOST=docker-container://dagger-engine-custom

echo "Testing connection to custom engine..."
dagger query <<< '{ container { from(address:"alpine:latest") { withExec(args:["echo", "Custom engine working!"]) { stdout } } } }'

echo
echo "Custom engine is running with your GC configuration!"
docker ps --filter "name=dagger-engine-custom" 

echo
echo "Verifying configuration in running container:"
docker exec dagger-engine-custom cat /etc/dagger/engine.json
echo
