# Dagger Engine Management Scripts

This repository contains scripts and configuration files for managing Dagger engines with custom settings, particularly focused on garbage collection (GC) configuration and cache management.

## Files Overview

### Configuration Files
- `engine.json` - Your custom Dagger engine configuration with GC settings

### Scripts
- `reload-dagger-engine.sh` - Reload Dagger engine container with current configuration
- `monitor-dagger-cache.sh` - Monitor cache usage and disk space

## Configuration Details

### Current Engine Configuration (`engine.json`)
```json
{
    "gc": {
        "enabled": true,
        "maxUsedSpace": "35GB",    // Maximum cache size
        "reservedSpace": "5GB",    // Always keep this much disk space free
        "minFreeSpace": "5GB",     // Trigger GC when free space drops below this
        "sweepSize": "35%"         // Percentage of cache to clean during GC
    },
    "logLevel": "error"
}
```

### Configuration Recommendations

| Setting | Current Value | Recommendation | Use Case |
|---------|---------------|----------------|----------|
| **maxUsedSpace** | 35GB | 20-50GB | Adjust based on available disk space and build complexity |
| **reservedSpace** | 5GB | 5-10GB | Keep 5-10% of total disk space as buffer |
| **minFreeSpace** | 5GB | 10-20GB | Should be larger than typical build requirements |
| **sweepSize** | 35% | 25-50% | Higher values clean more aggressively but may impact performance |
| **logLevel** | error | info/debug | Use `info` for troubleshooting, `error` for production |

**Recommendations by Environment:**
- **CI/CD**: Use `logLevel: "error"` to reduce log noise
- **Large Builds**: Increase `maxUsedSpace` to 50GB+ and `minFreeSpace` to 20GB+
- **Limited Disk**: Reduce `maxUsedSpace` to 20GB and increase `sweepSize` to 50%



## Quick Start

### Method 1: Automatic Engine Start (Recommended)
The simplest way to start using your configuration:

```bash
# Dagger will automatically start an engine with your ~/.config/dagger/engine.json
dagger query <<< '{ container { from(address:"alpine:latest") { withExec(args:["echo", "Hello!"]) { stdout } } } }'
```

### Method 2: Manual Docker Container Setup
Start a custom Dagger engine container with your configuration:

```bash
# Ensure config directory exists and copy your configuration
mkdir -p ~/.config/dagger
cp engine.json ~/.config/dagger/engine.json

# Stop any existing engines
docker rm -f $(docker ps -q --filter "name=dagger-engine-*") 2>/dev/null || true

# Start custom engine with v0.18.12
docker run --rm -d \
    -v /var/lib/dagger \
    -v ~/.config/dagger/engine.json:/etc/dagger/engine.json \
    --name dagger-engine-custom \
    --privileged \
    -p 1234:1234 \
    registry.dagger.io/engine:v0.18.12

# Connect to the custom engine
export _EXPERIMENTAL_DAGGER_RUNNER_HOST=docker-container://dagger-engine-custom

# Test the connection
dagger query <<< '{ container { from(address:"alpine:latest") { withExec(args:["echo", "Custom engine working!"]) { stdout } } } }'
```

## Using the Scripts

### Reload Dagger Engine
```bash
./reload-dagger-engine.sh
```
This script will:
- Stop any existing Dagger engines
- Copy your local `engine.json` to the config directory
- Start a new Dagger engine container with v0.18.12
- Test the connection and verify the configuration

### Monitor Cache Usage
```bash
./monitor-dagger-cache.sh
```
This script provides:
- Current disk usage
- Dagger GC configuration display
- Running engine status
- Docker system usage (including cache)
- Optional continuous monitoring

## Engine Version

All examples use Dagger engine version `v0.18.12`. You can update this in the Docker commands if needed.

## Troubleshooting

### Check Running Engines
```bash
docker ps --filter "name=dagger-engine-*"
```

### View Engine Configuration in Container
```bash
docker exec dagger-engine-custom cat /etc/dagger/engine.json
```

### Check Cache Usage
```bash
docker system df
```

### View Engine Logs
```bash
docker logs dagger-engine-custom
```

## Configuration Tips

- **maxUsedSpace**: Set this based on your available disk space and build requirements
- **minFreeSpace**: Should be larger than your typical build requirements
- **reservedSpace**: System-level disk space buffer
- **sweepSize**: Higher percentages clean more aggressively but may impact performance

## Notes

- The engine configuration is automatically loaded from `~/.config/dagger/engine.json`
- Custom runners allow for more control over the engine lifecycle
- Monitor your disk usage to ensure GC settings are working effectively
- The `_EXPERIMENTAL_DAGGER_RUNNER_HOST` environment variable connects to custom engines