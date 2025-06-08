#!/bin/bash
set -e

# Mark-1 AI Orchestrator Docker Entrypoint
# Production-ready entrypoint with proper initialization and error handling

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] MARK-1:${NC} $1"
}

log_info() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Signal handler for graceful shutdown
shutdown_handler() {
    log_info "Received shutdown signal, gracefully stopping Mark-1 AI Orchestrator..."
    
    # Send SIGTERM to all child processes
    if [ ! -z "$MARK1_PID" ]; then
        kill -TERM "$MARK1_PID" 2>/dev/null || true
        wait "$MARK1_PID" 2>/dev/null || true
    fi
    
    log_info "Mark-1 AI Orchestrator stopped gracefully"
    exit 0
}

# Set up signal handlers
trap 'shutdown_handler' SIGTERM SIGINT SIGQUIT

# Environment validation
validate_environment() {
    log_info "Validating environment configuration..."
    
    # Set default values
    export MARK1_ENV=${MARK1_ENV:-production}
    export MARK1_LOG_LEVEL=${MARK1_LOG_LEVEL:-INFO}
    export MARK1_HOST=${MARK1_HOST:-0.0.0.0}
    export MARK1_PORT=${MARK1_PORT:-8000}
    export MARK1_WORKERS=${MARK1_WORKERS:-4}
    export MARK1_MAX_MEMORY=${MARK1_MAX_MEMORY:-2048M}
    
    # Validate required directories
    for dir in "/app/logs" "/app/data" "/app/models" "/app/configs"; do
        if [ ! -d "$dir" ]; then
            log_warn "Creating missing directory: $dir"
            mkdir -p "$dir"
        fi
    done
    
    # Validate permissions
    if [ ! -w "/app/logs" ]; then
        log_error "Logs directory is not writable"
        exit 1
    fi
    
    log_info "Environment validation completed"
}

# System health check
system_health_check() {
    log_info "Performing system health check..."
    
    # Check available memory
    available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_memory" -lt 512 ]; then
        log_warn "Low available memory: ${available_memory}MB"
    fi
    
    # Check disk space
    available_disk=$(df /app | awk 'NR==2{printf "%.0f", $4/1024}')
    if [ "$available_disk" -lt 1024 ]; then
        log_warn "Low available disk space: ${available_disk}MB"
    fi
    
    # Check Python environment
    if ! python3 -c "import mark1" 2>/dev/null; then
        log_error "Mark-1 package not properly installed"
        exit 1
    fi
    
    log_info "System health check completed"
}

# Initialize logging
initialize_logging() {
    log_info "Initializing logging configuration..."
    
    # Create log file with timestamp
    export MARK1_LOG_FILE="/app/logs/mark1-$(date +'%Y%m%d').log"
    
    # Set up log rotation (keep last 7 days)
    find /app/logs -name "mark1-*.log" -mtime +7 -delete 2>/dev/null || true
    
    log_info "Logging initialized: $MARK1_LOG_FILE"
}

# Database initialization
initialize_database() {
    log_info "Initializing database connections..."
    
    # Wait for database if connection string is provided
    if [ ! -z "$DATABASE_URL" ]; then
        log_info "Waiting for database connection..."
        max_attempts=30
        attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            if python3 -c "
import sys
import asyncio
from mark1.core.database import check_database_connection

async def main():
    try:
        await check_database_connection()
        print('Database connection successful')
        sys.exit(0)
    except Exception as e:
        print(f'Database connection failed: {e}')
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
" 2>/dev/null; then
                log_info "Database connection established"
                break
            fi
            
            log_warn "Database connection attempt $attempt/$max_attempts failed"
            sleep 2
            attempt=$((attempt + 1))
        done
        
        if [ $attempt -gt $max_attempts ]; then
            log_error "Failed to establish database connection after $max_attempts attempts"
            exit 1
        fi
    fi
    
    log_info "Database initialization completed"
}

# Model initialization
initialize_models() {
    log_info "Initializing AI models..."
    
    # Check if models directory exists and has content
    if [ -d "/app/models" ] && [ "$(ls -A /app/models)" ]; then
        log_info "Pre-loading models from /app/models"
        
        # Pre-load models asynchronously
        python3 -c "
import asyncio
from mark1.orchestration.model_manager import AIModelManager

async def preload_models():
    try:
        manager = AIModelManager()
        models = await manager.discover_models(['/app/models'])
        print(f'Discovered {len(models)} model(s)')
        return True
    except Exception as e:
        print(f'Model preloading failed: {e}')
        return False

if __name__ == '__main__':
    success = asyncio.run(preload_models())
    exit(0 if success else 1)
" || log_warn "Model preloading failed, continuing with runtime discovery"
    else
        log_info "No pre-loaded models found, using runtime discovery"
    fi
    
    log_info "Model initialization completed"
}

# Security initialization
initialize_security() {
    log_info "Initializing security configuration..."
    
    # Set secure file permissions
    chmod 600 /app/configs/*.json 2>/dev/null || true
    
    # Generate or validate security keys
    if [ ! -f "/app/configs/security.key" ]; then
        log_info "Generating security key..."
        python3 -c "
import secrets
import base64
key = base64.b64encode(secrets.token_bytes(32)).decode()
with open('/app/configs/security.key', 'w') as f:
    f.write(key)
"
        chmod 600 /app/configs/security.key
    fi
    
    log_info "Security initialization completed"
}

# Performance tuning
performance_tuning() {
    log_info "Applying performance optimizations..."
    
    # Set optimal Python settings for production
    export PYTHONHASHSEED=random
    export PYTHONOPTIMIZE=1
    
    # Set memory limits if specified
    if [ ! -z "$MARK1_MAX_MEMORY" ]; then
        ulimit -v $(echo $MARK1_MAX_MEMORY | sed 's/M/000/g' | sed 's/G/000000/g')
    fi
    
    # Set optimal worker count based on CPU cores
    if [ -z "$MARK1_WORKERS" ]; then
        cpu_cores=$(nproc)
        export MARK1_WORKERS=$((cpu_cores * 2 + 1))
        log_info "Auto-configured workers: $MARK1_WORKERS (based on $cpu_cores CPU cores)"
    fi
    
    log_info "Performance tuning completed"
}

# Main initialization
main_init() {
    log "========================================"
    log "Mark-1 AI Orchestrator - Starting Up"
    log "========================================"
    log_info "Environment: $MARK1_ENV"
    log_info "Log Level: $MARK1_LOG_LEVEL"
    log_info "Host: $MARK1_HOST"
    log_info "Port: $MARK1_PORT"
    log_info "Workers: $MARK1_WORKERS"
    
    # Run initialization steps
    validate_environment
    initialize_logging
    system_health_check
    initialize_security
    initialize_database
    initialize_models
    performance_tuning
    
    log_info "Initialization completed successfully"
}

# Start the main application
start_application() {
    log_info "Starting Mark-1 AI Orchestrator..."
    
    # Start the application with proper logging
    exec python3 -m mark1.main "$@" 2>&1 | tee -a "$MARK1_LOG_FILE" &
    
    # Store the PID for graceful shutdown
    MARK1_PID=$!
    
    log_info "Mark-1 AI Orchestrator started with PID: $MARK1_PID"
    
    # Wait for the process to complete
    wait $MARK1_PID
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_info "Mark-1 AI Orchestrator exited successfully"
    else
        log_error "Mark-1 AI Orchestrator exited with code: $exit_code"
    fi
    
    exit $exit_code
}

# Health check endpoint
health_check() {
    log_info "Performing health check..."
    
    # Check if the application is responding
    if curl -f -s "http://localhost:${MARK1_PORT}/health" > /dev/null 2>&1; then
        log_info "Health check passed"
        exit 0
    else
        log_error "Health check failed"
        exit 1
    fi
}

# Main execution logic
case "$1" in
    "health")
        health_check
        ;;
    "mark1"|"")
        main_init
        start_application "$@"
        ;;
    *)
        # Pass through any other commands
        log_info "Executing command: $*"
        exec "$@"
        ;;
esac 