"""
Advanced Agent Selector & Optimization System for Mark-1 Agent Orchestrator

Session 18: Advanced Agent Selector & Optimization
Provides intelligent agent selection, performance optimization, and load balancing:
- Multi-criteria agent selection algorithms
- Performance-based agent ranking
- Load balancing and resource optimization
- Machine learning-based selection
- Agent fitness scoring and adaptation
- Dynamic selection strategy optimization
"""

import asyncio
import time
import json
import math
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import structlog
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta


class SelectionStrategy(Enum):
    """Agent selection strategies"""
    ROUND_ROBIN = "round_robin"              # Simple round-robin selection
    RANDOM = "random"                        # Random selection
    LEAST_LOADED = "least_loaded"            # Select least loaded agent
    PERFORMANCE_BASED = "performance_based"  # Select based on historical performance
    CAPABILITY_MATCH = "capability_match"    # Select based on capability matching
    COST_OPTIMIZED = "cost_optimized"        # Select based on cost optimization
    LATENCY_OPTIMIZED = "latency_optimized"  # Select for lowest latency
    HYBRID = "hybrid"                        # Combine multiple strategies
    ML_OPTIMIZED = "ml_optimized"            # Machine learning-based selection


class AgentState(Enum):
    """Agent availability states"""
    AVAILABLE = "available"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    FAILED = "failed"


class OptimizationGoal(Enum):
    """Optimization objectives"""
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_QUALITY = "maximize_quality"
    BALANCE_LOAD = "balance_load"
    MINIMIZE_ERRORS = "minimize_errors"


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an agent"""
    agent_id: str
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_response_time: float = 0.0
    success_rate: float = 1.0
    current_load: int = 0
    max_concurrent_tasks: int = 10
    last_task_timestamp: Optional[datetime] = None
    error_rate: float = 0.0
    quality_score: float = 1.0
    cost_per_task: float = 1.0
    resource_utilization: float = 0.0
    availability_score: float = 1.0
    
    # Time-series data
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    task_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    load_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update_response_time(self, response_time: float):
        """Update response time metrics"""
        self.response_times.append(response_time)
        if self.response_times:
            self.average_response_time = statistics.mean(self.response_times)
    
    def update_task_completion(self, success: bool, response_time: float, quality: float = 1.0):
        """Update task completion metrics"""
        self.total_tasks += 1
        self.last_task_timestamp = datetime.utcnow()
        
        task_record = {
            "timestamp": self.last_task_timestamp,
            "success": success,
            "response_time": response_time,
            "quality": quality
        }
        self.task_history.append(task_record)
        
        if success:
            self.completed_tasks += 1
        else:
            self.failed_tasks += 1
        
        # Update rates
        self.success_rate = self.completed_tasks / self.total_tasks if self.total_tasks > 0 else 1.0
        self.error_rate = self.failed_tasks / self.total_tasks if self.total_tasks > 0 else 0.0
        
        # Update response time
        self.update_response_time(response_time)
        
        # Update quality score (exponential moving average)
        alpha = 0.1
        self.quality_score = alpha * quality + (1 - alpha) * self.quality_score
    
    def update_load(self, current_load: int):
        """Update current load metrics"""
        self.current_load = current_load
        self.load_history.append({
            "timestamp": datetime.utcnow(),
            "load": current_load
        })
        
        # Update resource utilization
        self.resource_utilization = current_load / self.max_concurrent_tasks if self.max_concurrent_tasks > 0 else 0.0
    
    def get_state(self) -> AgentState:
        """Get current agent state based on metrics"""
        # Check for complete failure
        if self.total_tasks > 5 and self.error_rate >= 0.8:
            return AgentState.FAILED
        
        # Check load states
        if self.resource_utilization >= 1.0:
            return AgentState.OVERLOADED
        elif self.resource_utilization >= 0.8:
            return AgentState.BUSY
        
        # Check for failed state but with recovery possibility
        if self.total_tasks > 2 and self.error_rate >= 0.6:
            # Allow recovery if recent tasks were successful
            recent_tasks = list(self.task_history)[-3:] if self.task_history else []
            if recent_tasks:
                recent_successes = sum(1 for task in recent_tasks if task.get("success", False))
                if recent_successes >= 2:  # At least 2 of last 3 successful
                    return AgentState.AVAILABLE
            return AgentState.FAILED
        
        return AgentState.AVAILABLE
    
    def get_fitness_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate overall fitness score for agent selection"""
        if weights is None:
            weights = {
                "success_rate": 0.3,
                "response_time": 0.2,
                "quality": 0.2,
                "availability": 0.15,
                "load": 0.15
            }
        
        # Normalize metrics (higher is better)
        success_score = self.success_rate
        response_score = max(0, 1 - (self.average_response_time / 10.0))  # Assume 10s is max acceptable
        quality_score = self.quality_score
        availability_score = self.availability_score
        load_score = max(0, 1 - self.resource_utilization)
        
        fitness = (
            weights.get("success_rate", 0.3) * success_score +
            weights.get("response_time", 0.2) * response_score +
            weights.get("quality", 0.2) * quality_score +
            weights.get("availability", 0.15) * availability_score +
            weights.get("load", 0.15) * load_score
        )
        
        return max(0.0, min(1.0, fitness))


@dataclass
class TaskRequirements:
    """Requirements for a task that needs agent selection"""
    task_id: str
    required_capabilities: List[str]
    priority: int = 5  # 1-10 scale
    estimated_duration: float = 60.0  # seconds
    max_acceptable_latency: float = 5.0  # seconds
    quality_threshold: float = 0.8
    cost_budget: float = 10.0
    preferred_agents: List[str] = field(default_factory=list)
    excluded_agents: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None


@dataclass
class SelectionResult:
    """Result of agent selection process"""
    selected_agent_id: str
    selection_confidence: float
    selection_reasoning: str
    alternative_agents: List[str]
    selection_time: float
    predicted_performance: Dict[str, float]
    selection_strategy: SelectionStrategy
    optimization_score: float


class AgentSelectionAlgorithm(ABC):
    """Abstract base class for agent selection algorithms"""
    
    @abstractmethod
    def select_agent(
        self,
        available_agents: List[str],
        agent_metrics: Dict[str, AgentPerformanceMetrics],
        task_requirements: TaskRequirements,
        agent_capabilities: Dict[str, List[str]]
    ) -> SelectionResult:
        """Select the best agent for the given task"""
        pass


class RoundRobinSelector(AgentSelectionAlgorithm):
    """Simple round-robin agent selection"""
    
    def __init__(self):
        self.current_index = 0
    
    def select_agent(
        self,
        available_agents: List[str],
        agent_metrics: Dict[str, AgentPerformanceMetrics],
        task_requirements: TaskRequirements,
        agent_capabilities: Dict[str, List[str]]
    ) -> SelectionResult:
        """Select agent using round-robin strategy"""
        if not available_agents:
            raise ValueError("No available agents")
        
        # Filter agents by capabilities - be more flexible
        capable_agents = []
        for agent_id in available_agents:
            agent_caps = set(cap.lower() for cap in agent_capabilities.get(agent_id, []))
            required_caps = set(cap.lower() for cap in task_requirements.required_capabilities)
            
            # Check if agent has any required capabilities or if no specific requirements
            if not required_caps or required_caps.intersection(agent_caps):
                capable_agents.append(agent_id)
        
        if not capable_agents:
            # If no exact matches, use all available agents as fallback
            capable_agents = available_agents
        
        # Select using round-robin
        selected_agent = capable_agents[self.current_index % len(capable_agents)]
        self.current_index += 1
        
        return SelectionResult(
            selected_agent_id=selected_agent,
            selection_confidence=0.7,
            selection_reasoning="Round-robin selection with flexible capability matching",
            alternative_agents=capable_agents[1:] if len(capable_agents) > 1 else [],
            selection_time=0.001,
            predicted_performance={"response_time": 2.0, "success_rate": 0.9},
            selection_strategy=SelectionStrategy.ROUND_ROBIN,
            optimization_score=0.6
        )


class PerformanceBasedSelector(AgentSelectionAlgorithm):
    """Performance-based agent selection using historical metrics"""
    
    def __init__(self, fitness_weights: Dict[str, float] = None):
        self.fitness_weights = fitness_weights or {
            "success_rate": 0.3,
            "response_time": 0.25,
            "quality": 0.2,
            "availability": 0.15,
            "load": 0.1
        }
    
    def select_agent(
        self,
        available_agents: List[str],
        agent_metrics: Dict[str, AgentPerformanceMetrics],
        task_requirements: TaskRequirements,
        agent_capabilities: Dict[str, List[str]]
    ) -> SelectionResult:
        """Select agent based on performance metrics"""
        start_time = time.time()
        
        # Filter agents by capabilities and availability - be more lenient
        candidate_agents = []
        for agent_id in available_agents:
            agent_caps = set(cap.lower() for cap in agent_capabilities.get(agent_id, []))
            required_caps = set(cap.lower() for cap in task_requirements.required_capabilities)
            
            # More flexible capability matching
            if not required_caps or required_caps.intersection(agent_caps):
                metrics = agent_metrics.get(agent_id)
                # Allow failed agents but with penalty
                if metrics and metrics.get_state() not in [AgentState.OFFLINE]:
                    candidate_agents.append(agent_id)
        
        if not candidate_agents:
            # Fallback to all available agents
            candidate_agents = [
                agent_id for agent_id in available_agents 
                if agent_metrics.get(agent_id) and agent_metrics[agent_id].get_state() != AgentState.OFFLINE
            ]
        
        if not candidate_agents:
            raise ValueError("No suitable agents available")
        
        # Calculate fitness scores for each candidate
        agent_scores = []
        for agent_id in candidate_agents:
            metrics = agent_metrics[agent_id]
            fitness_score = metrics.get_fitness_score(self.fitness_weights)
            
            # Apply state-based adjustments
            if metrics.get_state() == AgentState.FAILED:
                fitness_score *= 0.5  # Heavy penalty for failed state
            elif metrics.get_state() == AgentState.OVERLOADED:
                fitness_score *= 0.7  # Penalty for overloaded state
            
            # Apply task-specific adjustments
            if metrics.average_response_time > task_requirements.max_acceptable_latency:
                fitness_score *= 0.8  # Reduced penalty
            
            if metrics.quality_score < task_requirements.quality_threshold:
                fitness_score *= 0.9  # Reduced penalty
            
            # Bonus for preferred agents
            if agent_id in task_requirements.preferred_agents:
                fitness_score *= 1.2
            
            # Penalty for excluded agents (shouldn't happen due to filtering)
            if agent_id in task_requirements.excluded_agents:
                fitness_score *= 0.1
            
            agent_scores.append((agent_id, fitness_score, metrics))
        
        # Sort by fitness score (descending)
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_agent = agent_scores[0][0]
        selected_metrics = agent_scores[0][2]
        confidence = agent_scores[0][1]
        
        alternative_agents = [agent_id for agent_id, _, _ in agent_scores[1:6]]  # Top 5 alternatives
        
        return SelectionResult(
            selected_agent_id=selected_agent,
            selection_confidence=confidence,
            selection_reasoning=f"Performance-based selection with fitness score {confidence:.3f}",
            alternative_agents=alternative_agents,
            selection_time=time.time() - start_time,
            predicted_performance={
                "response_time": selected_metrics.average_response_time,
                "success_rate": selected_metrics.success_rate,
                "quality": selected_metrics.quality_score
            },
            selection_strategy=SelectionStrategy.PERFORMANCE_BASED,
            optimization_score=confidence
        )


class LoadBalancedSelector(AgentSelectionAlgorithm):
    """Load-balanced agent selection to distribute work evenly"""
    
    def select_agent(
        self,
        available_agents: List[str],
        agent_metrics: Dict[str, AgentPerformanceMetrics],
        task_requirements: TaskRequirements,
        agent_capabilities: Dict[str, List[str]]
    ) -> SelectionResult:
        """Select agent with lowest current load"""
        start_time = time.time()
        
        # Filter agents by capabilities - be more flexible
        candidate_agents = []
        for agent_id in available_agents:
            agent_caps = set(cap.lower() for cap in agent_capabilities.get(agent_id, []))
            required_caps = set(cap.lower() for cap in task_requirements.required_capabilities)
            
            # More flexible capability matching
            if not required_caps or required_caps.intersection(agent_caps):
                metrics = agent_metrics.get(agent_id)
                if metrics and metrics.get_state() != AgentState.OFFLINE:
                    candidate_agents.append((agent_id, metrics))
        
        if not candidate_agents:
            # Fallback to all available agents
            for agent_id in available_agents:
                metrics = agent_metrics.get(agent_id)
                if metrics and metrics.get_state() != AgentState.OFFLINE:
                    candidate_agents.append((agent_id, metrics))
        
        if not candidate_agents:
            raise ValueError("No suitable agents available")
        
        # Sort by current load (ascending), then by success rate (descending), then by state priority
        def sort_key(agent_tuple):
            agent_id, metrics = agent_tuple
            state = metrics.get_state()
            # State priority: Available=0, Busy=1, Overloaded=2, Failed=3
            state_priority = {
                AgentState.AVAILABLE: 0,
                AgentState.BUSY: 1,
                AgentState.OVERLOADED: 2,
                AgentState.FAILED: 3,
                AgentState.MAINTENANCE: 4
            }.get(state, 5)
            
            return (metrics.resource_utilization, state_priority, -metrics.success_rate)
        
        candidate_agents.sort(key=sort_key)
        
        selected_agent = candidate_agents[0][0]
        selected_metrics = candidate_agents[0][1]
        
        # Calculate confidence based on load distribution and state
        loads = [metrics.resource_utilization for _, metrics in candidate_agents]
        load_variance = statistics.variance(loads) if len(loads) > 1 else 0
        base_confidence = max(0.5, 1.0 - load_variance)
        
        # Adjust confidence based on selected agent state
        state_multiplier = {
            AgentState.AVAILABLE: 1.0,
            AgentState.BUSY: 0.9,
            AgentState.OVERLOADED: 0.7,
            AgentState.FAILED: 0.5,
            AgentState.MAINTENANCE: 0.3
        }.get(selected_metrics.get_state(), 0.8)
        
        confidence = base_confidence * state_multiplier
        
        alternative_agents = [agent_id for agent_id, _ in candidate_agents[1:5]]
        
        return SelectionResult(
            selected_agent_id=selected_agent,
            selection_confidence=confidence,
            selection_reasoning=f"Load-balanced selection, agent load: {selected_metrics.resource_utilization:.2f}, state: {selected_metrics.get_state().value}",
            alternative_agents=alternative_agents,
            selection_time=time.time() - start_time,
            predicted_performance={
                "response_time": selected_metrics.average_response_time,
                "success_rate": selected_metrics.success_rate,
                "load": selected_metrics.resource_utilization
            },
            selection_strategy=SelectionStrategy.LEAST_LOADED,
            optimization_score=confidence
        )


class HybridSelector(AgentSelectionAlgorithm):
    """Hybrid selector combining multiple strategies"""
    
    def __init__(self, strategy_weights: Dict[SelectionStrategy, float] = None):
        self.strategy_weights = strategy_weights or {
            SelectionStrategy.PERFORMANCE_BASED: 0.4,
            SelectionStrategy.LEAST_LOADED: 0.3,
            SelectionStrategy.CAPABILITY_MATCH: 0.3
        }
        
        self.selectors = {
            SelectionStrategy.PERFORMANCE_BASED: PerformanceBasedSelector(),
            SelectionStrategy.LEAST_LOADED: LoadBalancedSelector(),
            SelectionStrategy.ROUND_ROBIN: RoundRobinSelector()
        }
    
    def select_agent(
        self,
        available_agents: List[str],
        agent_metrics: Dict[str, AgentPerformanceMetrics],
        task_requirements: TaskRequirements,
        agent_capabilities: Dict[str, List[str]]
    ) -> SelectionResult:
        """Select agent using hybrid approach"""
        start_time = time.time()
        
        # Get selections from different strategies
        strategy_results = {}
        for strategy, weight in self.strategy_weights.items():
            if strategy in self.selectors and weight > 0:
                try:
                    result = self.selectors[strategy].select_agent(
                        available_agents, agent_metrics, task_requirements, agent_capabilities
                    )
                    strategy_results[strategy] = (result, weight)
                except Exception as e:
                    # Skip failed strategies
                    continue
        
        if not strategy_results:
            raise ValueError("No strategy produced valid results")
        
        # Score each agent mentioned in results
        agent_scores = defaultdict(float)
        agent_details = {}
        
        for strategy, (result, weight) in strategy_results.items():
            agent_id = result.selected_agent_id
            score = result.optimization_score * weight
            agent_scores[agent_id] += score
            
            if agent_id not in agent_details:
                agent_details[agent_id] = {
                    "metrics": agent_metrics.get(agent_id),
                    "strategies": [],
                    "total_weight": 0
                }
            
            agent_details[agent_id]["strategies"].append(strategy.value)
            agent_details[agent_id]["total_weight"] += weight
        
        # Select agent with highest combined score
        if not agent_scores:
            raise ValueError("No agents scored by any strategy")
        
        selected_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
        selected_score = agent_scores[selected_agent]
        selected_metrics = agent_details[selected_agent]["metrics"]
        
        # Generate reasoning
        used_strategies = agent_details[selected_agent]["strategies"]
        reasoning = f"Hybrid selection using {', '.join(used_strategies)} with combined score {selected_score:.3f}"
        
        alternative_agents = sorted(
            [aid for aid in agent_scores.keys() if aid != selected_agent],
            key=lambda x: agent_scores[x],
            reverse=True
        )[:5]
        
        return SelectionResult(
            selected_agent_id=selected_agent,
            selection_confidence=min(1.0, selected_score),
            selection_reasoning=reasoning,
            alternative_agents=alternative_agents,
            selection_time=time.time() - start_time,
            predicted_performance={
                "response_time": selected_metrics.average_response_time,
                "success_rate": selected_metrics.success_rate,
                "quality": selected_metrics.quality_score
            },
            selection_strategy=SelectionStrategy.HYBRID,
            optimization_score=selected_score
        )


class MLOptimizedSelector(AgentSelectionAlgorithm):
    """Machine learning-optimized agent selection"""
    
    def __init__(self):
        self.feature_weights = {
            "success_rate": 0.25,
            "avg_response_time": -0.2,  # Negative because lower is better
            "quality_score": 0.2,
            "current_load": -0.15,      # Negative because lower is better
            "error_rate": -0.1,         # Negative because lower is better
            "availability_score": 0.1
        }
        self.selection_history = deque(maxlen=1000)
        self.learning_rate = 0.01
    
    def extract_features(self, agent_id: str, metrics: AgentPerformanceMetrics, task_req: TaskRequirements) -> np.ndarray:
        """Extract feature vector for ML model"""
        features = [
            metrics.success_rate,
            metrics.average_response_time / 10.0,  # Normalize
            metrics.quality_score,
            metrics.resource_utilization,
            metrics.error_rate,
            metrics.availability_score,
            len(task_req.required_capabilities) / 10.0,  # Normalize
            task_req.priority / 10.0,  # Normalize
            task_req.estimated_duration / 300.0,  # Normalize (5 min max)
        ]
        return np.array(features)
    
    def predict_performance(self, features: np.ndarray) -> float:
        """Simple linear model for performance prediction"""
        # Simplified ML model - in production, this could be a trained neural network
        weights = np.array([
            0.3,   # success_rate
            -0.2,  # response_time (negative)
            0.25,  # quality_score
            -0.15, # load (negative)
            -0.1,  # error_rate (negative)
            0.1,   # availability
            0.05,  # task complexity
            0.05,  # priority
            0.0    # duration
        ])
        
        score = np.dot(features, weights)
        return max(0.0, min(1.0, (score + 1) / 2))  # Normalize to 0-1
    
    def select_agent(
        self,
        available_agents: List[str],
        agent_metrics: Dict[str, AgentPerformanceMetrics],
        task_requirements: TaskRequirements,
        agent_capabilities: Dict[str, List[str]]
    ) -> SelectionResult:
        """Select agent using ML-based prediction"""
        start_time = time.time()
        
        # Filter agents by capabilities - be more flexible
        candidate_agents = []
        for agent_id in available_agents:
            agent_caps = set(cap.lower() for cap in agent_capabilities.get(agent_id, []))
            required_caps = set(cap.lower() for cap in task_requirements.required_capabilities)
            
            # More flexible capability matching
            if not required_caps or required_caps.intersection(agent_caps):
                metrics = agent_metrics.get(agent_id)
                if metrics and metrics.get_state() != AgentState.OFFLINE:
                    candidate_agents.append(agent_id)
        
        if not candidate_agents:
            # Fallback to all available agents
            candidate_agents = [
                agent_id for agent_id in available_agents 
                if agent_metrics.get(agent_id) and agent_metrics[agent_id].get_state() != AgentState.OFFLINE
            ]
        
        if not candidate_agents:
            raise ValueError("No suitable agents available")
        
        # Predict performance for each candidate
        agent_predictions = []
        for agent_id in candidate_agents:
            metrics = agent_metrics[agent_id]
            features = self.extract_features(agent_id, metrics, task_requirements)
            predicted_score = self.predict_performance(features)
            
            # Apply state-based adjustments
            state_multiplier = {
                AgentState.AVAILABLE: 1.0,
                AgentState.BUSY: 0.95,
                AgentState.OVERLOADED: 0.8,
                AgentState.FAILED: 0.6,
                AgentState.MAINTENANCE: 0.4
            }.get(metrics.get_state(), 0.7)
            
            adjusted_score = predicted_score * state_multiplier
            agent_predictions.append((agent_id, adjusted_score, metrics))
        
        # Sort by predicted score
        agent_predictions.sort(key=lambda x: x[1], reverse=True)
        
        selected_agent = agent_predictions[0][0]
        selected_score = agent_predictions[0][1]
        selected_metrics = agent_predictions[0][2]
        
        alternative_agents = [agent_id for agent_id, _, _ in agent_predictions[1:5]]
        
        return SelectionResult(
            selected_agent_id=selected_agent,
            selection_confidence=selected_score,
            selection_reasoning=f"ML-optimized selection with predicted score {selected_score:.3f}",
            alternative_agents=alternative_agents,
            selection_time=time.time() - start_time,
            predicted_performance={
                "response_time": selected_metrics.average_response_time,
                "success_rate": selected_metrics.success_rate,
                "quality": selected_metrics.quality_score,
                "predicted_score": selected_score
            },
            selection_strategy=SelectionStrategy.ML_OPTIMIZED,
            optimization_score=selected_score
        )
    
    def update_model(self, agent_id: str, predicted_score: float, actual_performance: Dict[str, float]):
        """Update ML model based on actual performance"""
        # Simple online learning update
        actual_score = (
            actual_performance.get("success", 0) * 0.4 +
            max(0, 1 - actual_performance.get("response_time", 10) / 10) * 0.3 +
            actual_performance.get("quality", 0.5) * 0.3
        )
        
        error = actual_score - predicted_score
        
        # Store for future analysis
        self.selection_history.append({
            "agent_id": agent_id,
            "predicted": predicted_score,
            "actual": actual_score,
            "error": error,
            "timestamp": datetime.utcnow()
        })


class AdvancedAgentSelector:
    """
    Advanced Agent Selector & Optimization System
    
    Provides intelligent agent selection with multiple strategies,
    performance optimization, and load balancing capabilities.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        
        # Performance tracking
        self.agent_metrics = {}
        self.task_history = deque(maxlen=10000)
        self.selection_statistics = defaultdict(int)
        
        # Selection strategies
        self.selectors = {
            SelectionStrategy.ROUND_ROBIN: RoundRobinSelector(),
            SelectionStrategy.PERFORMANCE_BASED: PerformanceBasedSelector(),
            SelectionStrategy.LEAST_LOADED: LoadBalancedSelector(),
            SelectionStrategy.HYBRID: HybridSelector(),
            SelectionStrategy.ML_OPTIMIZED: MLOptimizedSelector()
        }
        
        # Configuration
        self.default_strategy = SelectionStrategy.HYBRID
        self.optimization_goals = [OptimizationGoal.BALANCE_LOAD, OptimizationGoal.MAXIMIZE_QUALITY]
        self.strategy_adaptation_enabled = True
        
        # Agent registry
        self.registered_agents = {}
        self.agent_capabilities = {}
        self.agent_states = {}
        
        self.logger.info("Advanced Agent Selector initialized")
    
    def register_agent(self, agent_id: str, capabilities: List[str], max_concurrent_tasks: int = 10):
        """Register an agent with the selector"""
        self.registered_agents[agent_id] = {
            "registered_at": datetime.utcnow(),
            "max_concurrent_tasks": max_concurrent_tasks
        }
        
        self.agent_capabilities[agent_id] = capabilities
        self.agent_states[agent_id] = AgentState.AVAILABLE
        
        # Initialize metrics
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentPerformanceMetrics(
                agent_id=agent_id,
                max_concurrent_tasks=max_concurrent_tasks
            )
        
        self.logger.info(f"Agent registered: {agent_id} with capabilities: {capabilities}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        self.registered_agents.pop(agent_id, None)
        self.agent_capabilities.pop(agent_id, None)
        self.agent_states.pop(agent_id, None)
        # Keep metrics for historical analysis
        self.logger.info(f"Agent unregistered: {agent_id}")
    
    def update_agent_state(self, agent_id: str, state: AgentState):
        """Update agent state"""
        if agent_id in self.registered_agents:
            self.agent_states[agent_id] = state
            self.logger.debug(f"Agent {agent_id} state updated to {state.value}")
    
    def update_agent_load(self, agent_id: str, current_load: int):
        """Update agent current load"""
        if agent_id in self.agent_metrics:
            self.agent_metrics[agent_id].update_load(current_load)
    
    def record_task_completion(
        self,
        agent_id: str,
        task_id: str,
        success: bool,
        response_time: float,
        quality: float = 1.0,
        error: Optional[str] = None
    ):
        """Record task completion for performance tracking"""
        if agent_id in self.agent_metrics:
            self.agent_metrics[agent_id].update_task_completion(success, response_time, quality)
            
            # Update task history
            task_record = {
                "task_id": task_id,
                "agent_id": agent_id,
                "success": success,
                "response_time": response_time,
                "quality": quality,
                "error": error,
                "timestamp": datetime.utcnow()
            }
            self.task_history.append(task_record)
            
            self.logger.debug(f"Task completion recorded: {task_id} -> {agent_id} (success: {success})")
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        available = []
        for agent_id, state in self.agent_states.items():
            # Be more inclusive - include agents that can potentially handle tasks
            if state in [AgentState.AVAILABLE, AgentState.BUSY, AgentState.OVERLOADED]:
                metrics = self.agent_metrics.get(agent_id)
                # Allow overloaded agents for selection but with lower priority
                if metrics:
                    available.append(agent_id)
        return available
    
    async def select_agent(
        self,
        task_requirements: TaskRequirements,
        strategy: Optional[SelectionStrategy] = None
    ) -> SelectionResult:
        """Select the best agent for a given task"""
        start_time = time.time()
        
        # Use default strategy if none specified
        if strategy is None:
            strategy = self._get_optimal_strategy(task_requirements)
        
        # Get available agents
        available_agents = self.get_available_agents()
        
        if not available_agents:
            raise ValueError("No agents available")
        
        # Apply exclusions and preferences
        if task_requirements.excluded_agents:
            available_agents = [a for a in available_agents if a not in task_requirements.excluded_agents]
        
        if not available_agents:
            raise ValueError("No agents available after exclusions")
        
        # Select using specified strategy
        selector = self.selectors.get(strategy)
        if not selector:
            raise ValueError(f"Unknown selection strategy: {strategy}")
        
        try:
            result = selector.select_agent(
                available_agents,
                self.agent_metrics,
                task_requirements,
                self.agent_capabilities
            )
            
            # Update selection statistics
            self.selection_statistics[strategy] += 1
            self.selection_statistics["total"] += 1
            
            # Log selection
            self.logger.info(
                f"Agent selected for task {task_requirements.task_id}: "
                f"{result.selected_agent_id} using {strategy.value} "
                f"(confidence: {result.selection_confidence:.3f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Agent selection failed: {e}")
            raise
    
    def _get_optimal_strategy(self, task_requirements: TaskRequirements) -> SelectionStrategy:
        """Determine optimal selection strategy based on task and system state"""
        # Simple heuristics for strategy selection
        
        # High priority tasks - use performance-based
        if task_requirements.priority >= 8:
            return SelectionStrategy.PERFORMANCE_BASED
        
        # Low latency requirements - use ML optimized
        if task_requirements.max_acceptable_latency < 1.0:
            return SelectionStrategy.ML_OPTIMIZED
        
        # High load periods - use load balancing
        avg_load = statistics.mean([
            metrics.resource_utilization
            for metrics in self.agent_metrics.values()
        ]) if self.agent_metrics else 0
        
        if avg_load > 0.7:
            return SelectionStrategy.LEAST_LOADED
        
        # Default to hybrid for balanced approach
        return SelectionStrategy.HYBRID
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents"""
        summary = {
            "total_agents": len(self.registered_agents),
            "available_agents": len(self.get_available_agents()),
            "total_tasks": sum(metrics.total_tasks for metrics in self.agent_metrics.values()),
            "selection_statistics": dict(self.selection_statistics),
            "agent_performance": {}
        }
        
        for agent_id, metrics in self.agent_metrics.items():
            summary["agent_performance"][agent_id] = {
                "state": self.agent_states.get(agent_id, AgentState.OFFLINE).value,
                "total_tasks": metrics.total_tasks,
                "success_rate": metrics.success_rate,
                "average_response_time": metrics.average_response_time,
                "current_load": metrics.current_load,
                "fitness_score": metrics.get_fitness_score()
            }
        
        return summary
    
    def optimize_selection_strategy(self) -> Dict[str, Any]:
        """Analyze and optimize selection strategies"""
        if len(self.task_history) < 100:
            return {"message": "Insufficient data for optimization"}
        
        # Analyze strategy performance
        strategy_performance = defaultdict(list)
        
        for task in list(self.task_history)[-1000:]:  # Last 1000 tasks
            # This would require tracking which strategy was used for each task
            # For now, we'll simulate this analysis
            pass
        
        # For demonstration, return current statistics
        total_selections = self.selection_statistics.get("total", 1)
        strategy_usage = {
            strategy.value: count / total_selections
            for strategy, count in self.selection_statistics.items()
            if strategy != "total"
        }
        
        return {
            "strategy_usage": strategy_usage,
            "recommendation": "Continue using hybrid strategy for balanced performance",
            "optimization_opportunities": [
                "Increase ML strategy usage for low-latency tasks",
                "Use load balancing during peak hours",
                "Consider agent specialization for specific task types"
            ]
        }
    
    async def rebalance_load(self) -> Dict[str, Any]:
        """Rebalance load across agents"""
        # Get current load distribution
        agent_loads = {
            agent_id: metrics.resource_utilization
            for agent_id, metrics in self.agent_metrics.items()
        }
        
        if not agent_loads:
            return {"message": "No agents to rebalance"}
        
        # Calculate load statistics
        loads = list(agent_loads.values())
        avg_load = statistics.mean(loads)
        load_variance = statistics.variance(loads) if len(loads) > 1 else 0
        
        # Identify overloaded and underloaded agents
        overloaded_agents = [
            agent_id for agent_id, load in agent_loads.items()
            if load > avg_load + 0.2
        ]
        
        underloaded_agents = [
            agent_id for agent_id, load in agent_loads.items()
            if load < avg_load - 0.2
        ]
        
        rebalancing_actions = []
        
        # Suggest rebalancing actions
        if overloaded_agents and underloaded_agents:
            for overloaded in overloaded_agents[:3]:  # Top 3 overloaded
                for underloaded in underloaded_agents[:3]:  # Top 3 underloaded
                    rebalancing_actions.append({
                        "action": "redirect_tasks",
                        "from_agent": overloaded,
                        "to_agent": underloaded,
                        "current_load_from": agent_loads[overloaded],
                        "current_load_to": agent_loads[underloaded]
                    })
        
        return {
            "current_load_distribution": agent_loads,
            "average_load": avg_load,
            "load_variance": load_variance,
            "overloaded_agents": overloaded_agents,
            "underloaded_agents": underloaded_agents,
            "rebalancing_actions": rebalancing_actions,
            "load_balance_score": max(0, 1 - load_variance)
        }
