# Mark-1 Universal Plugin System - Complete Documentation

## 🚀 System Overview

Mark-1 is a production-ready AI-powered universal plugin orchestration system that enables intelligent task automation through natural language processing and plugin chaining.

### ✅ Core Features
- **AI-Powered Orchestration**: Natural language to multi-step workflows
- **Universal Plugin System**: Support for any type of plugin or tool
- **Intelligent Context Management**: Advanced caching and data flow
- **Real-time Monitoring**: Comprehensive metrics and observability
- **Database Integration**: Full persistence with SQLite/PostgreSQL support
- **CLI Interface**: Complete command-line management tools

## 📊 Database and Context Management

### Database Commands

```bash
# Database status and health
python -m mark1.cli.simple_cli db status
python -m mark1.cli.simple_cli db init

# Query database tables directly
python -m mark1.cli.simple_cli db query plugins --limit 20
python -m mark1.cli.simple_cli db query plugin_executions --where "status=success"
python -m mark1.cli.simple_cli db query plugin_capabilities

# Execute raw SQL (read-only)
python -m mark1.cli.simple_cli db exec-sql "SELECT * FROM plugins WHERE status='active'"
python -m mark1.cli.simple_cli db exec-sql "SELECT COUNT(*) FROM plugin_executions"

# Context management
python -m mark1.cli.simple_cli db contexts --limit 50
python -m mark1.cli.simple_cli db create-context "my_key" '{"data": "value"}' --type task
python -m mark1.cli.simple_cli db get-context --key "my_key"

# Context search and export
python -m mark1.cli.simple_cli db context-search "workflow" --limit 10
python -m mark1.cli.simple_cli db context-export ./contexts_backup.json

# Comprehensive monitoring
python -m mark1.cli.simple_cli db monitoring
```

### Database Schema

#### Plugins Table
```sql
CREATE TABLE plugins (
    plugin_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    plugin_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    version VARCHAR,
    description TEXT,
    installation_path VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    last_used_at TIMESTAMP
);
```

#### Plugin Executions Table
```sql
CREATE TABLE plugin_executions (
    execution_id VARCHAR PRIMARY KEY,
    plugin_id VARCHAR NOT NULL,
    capability_name VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    execution_time FLOAT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    inputs JSON,
    outputs JSON,
    error_message TEXT,
    FOREIGN KEY (plugin_id) REFERENCES plugins (plugin_id)
);
```

#### Plugin Capabilities Table
```sql
CREATE TABLE plugin_capabilities (
    id INTEGER PRIMARY KEY,
    plugin_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    description TEXT,
    input_types JSON,
    output_types JSON,
    parameters JSON,
    FOREIGN KEY (plugin_id) REFERENCES plugins (plugin_id)
);
```

## 🔧 Plugin Management

### Plugin Commands

```bash
# List and manage plugins
python -m mark1.cli.simple_cli plugin list
python -m mark1.cli.simple_cli plugin info <plugin_id>
python -m mark1.cli.simple_cli plugin test <plugin_id>

# Install plugins
python -m mark1.cli.simple_cli plugin install-local ./my_plugin/
python -m mark1.cli.simple_cli plugin install-github https://github.com/user/repo

# Plugin development
python -m mark1.cli.simple_cli plugin generate my_new_plugin
python -m mark1.cli.simple_cli plugin validate ./my_plugin/
```

### Plugin Development Guide

#### 1. Plugin Structure
```
my_plugin/
├── plugin.yaml           # Plugin metadata
├── main.py               # Main plugin code
├── requirements.txt      # Dependencies
├── README.md            # Documentation
└── tests/               # Test files
    └── test_plugin.py
```

#### 2. Plugin Metadata (plugin.yaml)
```yaml
name: "my_plugin"
version: "1.0.0"
description: "My custom plugin"
plugin_type: "python_library"
execution_mode: "python_function"

capabilities:
  - name: "process_data"
    description: "Process input data"
    input_types: ["text", "json"]
    output_types: ["json"]
    parameters:
      format:
        type: "string"
        default: "json"
        description: "Output format"

entry_points:
  - capability: "process_data"
    function: "main:process_data"

dependencies:
  - "requests>=2.25.0"
  - "pandas>=1.3.0"
```

#### 3. Plugin Implementation (main.py)
```python
def process_data(input_data, format="json", **kwargs):
    """Process input data and return results"""
    try:
        # Your plugin logic here
        result = {"processed": input_data, "format": format}
        return {
            "success": True,
            "data": result,
            "message": "Data processed successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Processing failed"
        }
```

## 🧠 AI Orchestration

### Orchestration Commands

```bash
# Run AI orchestration
python -m mark1.cli.simple_cli orchestrate run "Analyze this text and convert to CSV"
python -m mark1.cli.simple_cli orchestrate plan "Process customer feedback data"

# Workflow management
python -m mark1.cli.simple_cli orchestrate workflows
python -m mark1.cli.simple_cli orchestrate status <orchestration_id>
python -m mark1.cli.simple_cli orchestrate history --limit 20
```

### Example Orchestration Workflows

#### 1. Data Analysis Pipeline
```bash
python -m mark1.cli.simple_cli orchestrate run \
  "I have customer feedback: 'Great product but expensive'. 
   Please analyze sentiment, calculate readability, and create a CSV report."
```

#### 2. Content Processing
```bash
python -m mark1.cli.simple_cli orchestrate run \
  "Convert this JSON to CSV, analyze the text content, and generate insights: 
   {'review': 'Amazing software!', 'rating': 5}"
```

#### 3. Multi-step Research
```bash
python -m mark1.cli.simple_cli orchestrate run \
  "Process multiple customer reviews, extract key themes, 
   and create a comprehensive analysis report."
```

## 📈 Monitoring and Observability

### Prometheus Metrics Setup

#### 1. Install Prometheus
```bash
# macOS
brew install prometheus

# Ubuntu/Debian
sudo apt-get install prometheus

# Docker
docker run -p 9090:9090 prom/prometheus
```

#### 2. Prometheus Configuration (prometheus.yml)
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "mark1_rules.yml"

scrape_configs:
  - job_name: 'mark1-system'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 5s
    metrics_path: '/metrics'

  - job_name: 'mark1-plugins'
    static_configs:
      - targets: ['localhost:8001']
    scrape_interval: 10s
    metrics_path: '/plugin-metrics'

  - job_name: 'mark1-database'
    static_configs:
      - targets: ['localhost:8002']
    scrape_interval: 30s
    metrics_path: '/db-metrics'

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### 3. Mark-1 Metrics Exporter
Create `src/mark1/monitoring/prometheus_exporter.py`:
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import asyncio

# Define metrics
plugin_executions_total = Counter(
    'mark1_plugin_executions_total',
    'Total plugin executions',
    ['plugin_id', 'capability', 'status']
)

plugin_execution_duration = Histogram(
    'mark1_plugin_execution_duration_seconds',
    'Plugin execution duration',
    ['plugin_id', 'capability']
)

active_orchestrations = Gauge(
    'mark1_active_orchestrations',
    'Number of active orchestrations'
)

context_cache_size = Gauge(
    'mark1_context_cache_size_bytes',
    'Context cache size in bytes'
)

database_connections = Gauge(
    'mark1_database_connections',
    'Number of active database connections'
)

def start_metrics_server(port=8000):
    """Start Prometheus metrics server"""
    start_http_server(port)
    print(f"Metrics server started on port {port}")

async def update_metrics():
    """Update metrics periodically"""
    while True:
        try:
            # Update context cache metrics
            from mark1.core.context_manager import ContextManager
            context_manager = ContextManager()
            if hasattr(context_manager, 'cache'):
                cache_stats = context_manager.cache.stats
                context_cache_size.set(cache_stats.total_size_bytes)
            
            # Update orchestration metrics
            # Add your metric collection logic here
            
        except Exception as e:
            print(f"Metrics update error: {e}")
        
        await asyncio.sleep(10)
```

### Grafana Dashboard Setup

#### 1. Install Grafana
```bash
# macOS
brew install grafana

# Ubuntu/Debian
sudo apt-get install grafana

# Docker
docker run -d -p 3000:3000 grafana/grafana
```

#### 2. Grafana Configuration

**Access Grafana:**
- URL: http://localhost:3000
- Default login: admin/admin

**Add Prometheus Data Source:**
1. Go to Configuration → Data Sources
2. Add Prometheus data source
3. URL: http://localhost:9090
4. Save & Test

#### 3. Mark-1 Dashboard JSON
Create `monitoring/grafana-dashboard.json`:
```json
{
  "dashboard": {
    "id": null,
    "title": "Mark-1 Universal Plugin System",
    "tags": ["mark1", "plugins", "ai"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Plugin Executions Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(mark1_plugin_executions_total[5m])",
            "legendFormat": "{{plugin_id}} - {{capability}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Plugin Execution Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(mark1_plugin_execution_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(mark1_plugin_execution_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Active Orchestrations",
        "type": "singlestat",
        "targets": [
          {
            "expr": "mark1_active_orchestrations",
            "legendFormat": "Active"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Context Cache Size",
        "type": "singlestat",
        "targets": [
          {
            "expr": "mark1_context_cache_size_bytes / 1024 / 1024",
            "legendFormat": "MB"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8}
      },
      {
        "id": 5,
        "title": "Plugin Success Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(mark1_plugin_executions_total{status=\"success\"}[5m]) / rate(mark1_plugin_executions_total[5m]) * 100",
            "legendFormat": "{{plugin_id}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

#### 4. Import Dashboard
1. Go to Dashboards → Import
2. Upload the JSON file or paste the JSON
3. Configure data source as Prometheus
4. Save dashboard

### Alert Rules

Create `monitoring/mark1_rules.yml`:
```yaml
groups:
  - name: mark1_alerts
    rules:
      - alert: HighPluginFailureRate
        expr: rate(mark1_plugin_executions_total{status="error"}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High plugin failure rate detected"
          description: "Plugin {{$labels.plugin_id}} has failure rate > 10%"

      - alert: SlowPluginExecution
        expr: histogram_quantile(0.95, rate(mark1_plugin_execution_duration_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow plugin execution detected"
          description: "95th percentile execution time > 30 seconds"

      - alert: ContextCacheOverflow
        expr: mark1_context_cache_size_bytes > 1073741824  # 1GB
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Context cache size exceeded 1GB"
          description: "Context cache is consuming too much memory"
```

## 🔧 System Configuration

### Environment Variables
```bash
# Database
export MARK1_DATABASE_URL="sqlite:///./data/mark1.db"
export MARK1_DATABASE_POOL_SIZE=10

# OLLAMA
export MARK1_OLLAMA_URL="https://f6da-103-167-213-208.ngrok-free.app"
export MARK1_OLLAMA_MODEL="llama3.1:8b"

# Monitoring
export MARK1_METRICS_PORT=8000
export MARK1_METRICS_ENABLED=true

# Plugins
export MARK1_PLUGINS_DIR="~/.mark1/plugins"
export MARK1_PLUGIN_TIMEOUT=300

# Context
export MARK1_CONTEXT_CACHE_SIZE=1000
export MARK1_CONTEXT_COMPRESSION=true
```

### Configuration File (config.yaml)
```yaml
database:
  url: "sqlite:///./data/mark1.db"
  pool_size: 10
  echo: false

ollama:
  base_url: "https://f6da-103-167-213-208.ngrok-free.app"
  default_model: "llama3.1:8b"
  timeout: 300

plugins:
  directory: "~/.mark1/plugins"
  auto_install: true
  validation_strict: false
  timeout: 300

context:
  cache_size: 1000
  compression: true
  ttl: 3600

monitoring:
  metrics_enabled: true
  metrics_port: 8000
  log_level: "INFO"
  
orchestration:
  max_concurrent: 10
  default_timeout: 600
  retry_attempts: 3
```

## 🚀 Production Deployment

### Docker Deployment

#### 1. Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config.yaml .

EXPOSE 8000 8080

CMD ["python", "-m", "mark1.api.server"]
```

#### 2. Docker Compose
```yaml
version: '3.8'

services:
  mark1:
    build: .
    ports:
      - "8080:8080"
      - "8000:8000"
    environment:
      - MARK1_DATABASE_URL=postgresql://user:pass@postgres:5432/mark1
    volumes:
      - ./data:/app/data
      - ./plugins:/app/plugins
    depends_on:
      - postgres
      - prometheus

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: mark1
      POSTGRES_USER: mark1_user
      POSTGRES_PASSWORD: mark1_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/mark1_rules.yml:/etc/prometheus/mark1_rules.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana-dashboard.json:/var/lib/grafana/dashboards/

volumes:
  postgres_data:
  grafana_data:
```

### Kubernetes Deployment

#### 1. Deployment YAML
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mark1-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mark1
  template:
    metadata:
      labels:
        app: mark1
    spec:
      containers:
      - name: mark1
        image: mark1:latest
        ports:
        - containerPort: 8080
        - containerPort: 8000
        env:
        - name: MARK1_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mark1-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## 🧪 Testing and Validation

### Test Commands
```bash
# Run all tests
python -m pytest tests/ -v

# Test specific components
python test_db_commands.py
python test_complex_workflow.py
python test_real_github_plugins.py

# Performance testing
python -m mark1.testing.load_test --concurrent 10 --duration 300

# Integration testing
python -m mark1.testing.integration_test --full-suite
```

### Health Checks
```bash
# System health
curl http://localhost:8080/health

# Metrics endpoint
curl http://localhost:8000/metrics

# Database health
python -m mark1.cli.simple_cli db status

# Plugin health
python -m mark1.cli.simple_cli plugin health-check
```

## 📚 API Reference

### REST API Endpoints

```bash
# Orchestration
POST /api/v1/orchestrate
GET /api/v1/orchestrations/{id}
GET /api/v1/orchestrations

# Plugins
GET /api/v1/plugins
POST /api/v1/plugins/{id}/execute
GET /api/v1/plugins/{id}/status

# Context
GET /api/v1/contexts
POST /api/v1/contexts
GET /api/v1/contexts/{id}

# Monitoring
GET /api/v1/metrics
GET /api/v1/health
GET /api/v1/status
```

## 🔒 Security and Best Practices

### Security Configuration
- Enable authentication for API endpoints
- Use HTTPS in production
- Implement rate limiting
- Validate all plugin inputs
- Sandbox plugin execution
- Regular security audits

### Performance Optimization
- Enable context compression
- Configure appropriate cache sizes
- Use connection pooling
- Monitor resource usage
- Implement circuit breakers
- Use async operations

## 📞 Support and Troubleshooting

### Common Issues
1. **Database Connection**: Check greenlet installation
2. **Plugin Failures**: Validate plugin metadata
3. **OLLAMA Timeout**: Increase timeout settings
4. **Memory Usage**: Adjust cache sizes
5. **Performance**: Enable monitoring and profiling

### Logs and Debugging
```bash
# Enable debug logging
export MARK1_LOG_LEVEL=DEBUG

# View logs
tail -f logs/mark1.log

# Database debugging
python -m mark1.cli.simple_cli db exec-sql "SELECT * FROM plugin_executions ORDER BY started_at DESC LIMIT 10"
```

---

**Mark-1 Universal Plugin System** - Production-ready AI orchestration platform
Version 1.0.0 | Documentation updated: 2024-01-01
