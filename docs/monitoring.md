# Mark-1 Universal Plugin System - Monitoring Guide

## Overview

The Mark-1 system provides comprehensive monitoring capabilities through built-in commands and can be extended with external monitoring tools like Grafana and Prometheus.

## Built-in Monitoring Commands

### Database Status
```bash
python -m mark1.cli.db_commands status
```
- Shows database connection status
- Displays database file information
- Reports table statistics

### System Monitoring
```bash
python -m mark1.cli.db_commands monitoring
```
- Comprehensive system health report
- Plugin execution statistics
- Context management metrics
- OLLAMA integration status

### Context Management
```bash
python -m mark1.cli.db_commands context-list
```
- Lists all active contexts
- Shows context hierarchy
- Memory usage statistics

## Key Metrics

### Plugin Performance
- **Execution Time**: Average time per plugin execution
- **Success Rate**: Percentage of successful plugin executions
- **Error Rate**: Plugin failure statistics
- **Usage Frequency**: Most/least used plugins

### Context Management
- **Cache Hit Rate**: Context retrieval efficiency
- **Memory Usage**: Context storage consumption
- **Context Lifecycle**: Creation, update, and cleanup metrics

### AI Orchestration
- **Intent Analysis Time**: Time to analyze user prompts
- **Planning Time**: Workflow creation duration
- **Workflow Success Rate**: End-to-end orchestration success

### System Health
- **Database Status**: Connection and performance
- **OLLAMA Integration**: Model availability and response times
- **Memory Usage**: System resource consumption

## External Monitoring Integration

### Prometheus Metrics (Future Enhancement)

The system can be extended to export metrics to Prometheus:

```python
# Example metrics that could be exported
plugin_execution_total = Counter('mark1_plugin_executions_total', 'Total plugin executions', ['plugin_id', 'status'])
plugin_execution_duration = Histogram('mark1_plugin_execution_duration_seconds', 'Plugin execution duration')
context_cache_hits = Counter('mark1_context_cache_hits_total', 'Context cache hits')
orchestration_success_rate = Gauge('mark1_orchestration_success_rate', 'Orchestration success rate')
```

### Grafana Dashboard (Future Enhancement)

A Grafana dashboard could include:

1. **System Overview Panel**
   - Total plugins installed
   - Active orchestrations
   - System uptime

2. **Performance Metrics Panel**
   - Plugin execution times (histogram)
   - Success/failure rates (pie chart)
   - Throughput (requests per minute)

3. **Resource Usage Panel**
   - Memory consumption
   - Database size
   - Context cache utilization

4. **Error Tracking Panel**
   - Error rates by component
   - Recent error logs
   - Alert thresholds

## Alerting

### Built-in Alerts
The system logs warnings and errors for:
- Plugin execution failures
- Database connection issues
- Context management problems
- OLLAMA integration failures

### Custom Alerts (Future Enhancement)
- Plugin execution time exceeding thresholds
- High error rates
- Resource exhaustion
- Database performance degradation

## Log Analysis

### Log Levels
- **DEBUG**: Detailed execution information
- **INFO**: Normal operation events
- **WARNING**: Non-critical issues
- **ERROR**: Serious problems requiring attention

### Key Log Patterns
```
# Successful plugin execution
Plugin capability executed successfully capability=analyze_text

# Workflow completion
Intelligent orchestration completed success=True

# Database issues (non-critical)
Database recording failed error='AsyncSession' object has no attribute 'query'

# Context management
Advanced context created successfully
```

## Performance Optimization

### Monitoring Performance Bottlenecks
1. **Plugin Execution Time**: Monitor slow plugins
2. **Database Queries**: Track query performance
3. **Context Serialization**: Monitor JSON conversion overhead
4. **OLLAMA Response Time**: Track AI model performance

### Optimization Strategies
1. **Plugin Caching**: Cache frequently used plugin results
2. **Context Compression**: Compress large context data
3. **Database Indexing**: Optimize query performance
4. **Async Processing**: Parallelize independent operations

## Troubleshooting

### Common Issues
1. **Database Connection Errors**: Usually non-critical, system falls back to memory-only mode
2. **Context Serialization Errors**: Fixed with PluginResult.to_dict() method
3. **Plugin Not Found**: Check plugin installation and registration
4. **OLLAMA Timeout**: Verify OLLAMA service availability

### Health Checks
```bash
# Quick system health check
python test_db_commands.py

# Complex workflow test
python test_complex_workflow.py
```

## Future Enhancements

1. **Real-time Metrics**: WebSocket-based live monitoring
2. **Performance Profiling**: Detailed execution analysis
3. **Predictive Analytics**: Forecast system load and capacity
4. **Automated Scaling**: Dynamic resource allocation
5. **Integration APIs**: REST endpoints for external monitoring tools
