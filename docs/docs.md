# Mark-1 AI Orchestrator: Comprehensive Documentation

## 1. Introduction and Overview

Mark-1 AI Orchestrator is a comprehensive, production-ready AI orchestration platform designed to manage and coordinate complex AI workloads. It addresses the growing need for a robust system that can handle sophisticated multi-agent coordination, intelligent AI model management, advanced workflow orchestration, and enterprise-grade security for modern AI applications.

The platform aims to solve common challenges in deploying and scaling AI solutions, such as:
- Efficiently managing the lifecycle and interactions of multiple AI agents.
- Dynamically selecting and deploying appropriate AI models based on demand and performance.
- Orchestrating intricate workflows that may involve several AI models and data processing steps.
- Ensuring the security and reliability of AI applications in enterprise environments.
- Providing comprehensive monitoring and observability for AI systems in production.

Mark-1 empowers organizations to build, deploy, and manage scalable and resilient AI-driven systems, accelerating the adoption of AI technologies and enabling the development of next-generation intelligent applications.

## 2. Project Architecture

### 2.1. Architectural Layers Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mark-1 AI Orchestrator                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Agent Layer   │  │  Workflow Layer │  │  Security Layer │ │
│  │                 │  │                 │  │                 │ │
│  │ • Coordination  │  │ • Orchestration │  │ • Authentication│ │
│  │ • Communication │  │ • Optimization  │  │ • Authorization │ │
│  │ • Load Balancing│  │ • Error Handling│  │ • Encryption    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Model Layer    │  │  Storage Layer  │  │ Monitoring Layer│ │
│  │                 │  │                 │  │                 │ │
│  │ • Model Manager │  │ • PostgreSQL    │  │ • Prometheus    │ │
│  │ • Inference     │  │ • Redis Cache   │  │ • Grafana       │ │
│  │ • Routing       │  │ • File Storage  │  │ • ELK Stack     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    API Gateway Layer                       │ │
│  │  • REST API  • WebSocket  • CLI Interface  • Web UI       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2. Core Components

The Mark-1 AI Orchestrator is built upon a modular, layered architecture designed for scalability, flexibility, and robustness. Each layer encapsulates specific functionalities:

-   **Agent Layer**: Responsible for managing and coordinating AI agents.
    -   **Coordination**: Oversees agent lifecycles, roles, and collaborations. This includes advanced agent management and integration with various agent frameworks like LangChain (including LangGraph for state-based workflows), AutoGPT, and CrewAI.
    -   **Communication**: Facilitates high-performance message passing between agents, potentially using a communication bus and defined protocols.
    -   **Load Balancing**: Dynamically distributes workloads across agent clusters to ensure optimal resource utilization and performance.
    -   **Advanced Codebase Scanning**: While not exclusively part of the agent layer, the `AdvancedCodebaseScanner` (detailed in `project.md`) plays a crucial role in understanding and integrating external agents by analyzing their code, dependencies, and capabilities. This includes multi-language AST analysis, LLM call detection, and agent pattern recognition.

-   **Workflow Layer**: Manages the definition, execution, and optimization of complex AI workflows.
    -   **Orchestration**: Handles multi-step workflows with conditional execution, parallel processing, and error handling. The `LangChainOrchestrator` (from `project.md`) would fit here, managing LangChain-based agentic workflows.
    -   **Optimization**: Includes real-time adaptation of workflows and automated performance optimization.
    -   **Error Handling**: Implements robust error detection and recovery mechanisms within workflows.

-   **Security Layer**: Ensures the integrity, confidentiality, and availability of the platform and its data.
    -   **Authentication**: Implements multi-factor authentication (MFA) and supports various authentication methods (JWT, API Keys, OAuth2).
    -   **Authorization**: Provides Role-Based Access Control (RBAC) to manage permissions.
    -   **Encryption**: Enforces end-to-end encryption (e.g., TLS 1.3) and secure key management.
    -   **Vulnerability Assessment & Input Validation**: Includes automated security scanning and input sanitization.

-   **Model Layer**: Manages AI models, their lifecycle, and inference processes.
    -   **Model Manager**: Handles dynamic model loading, discovery from multiple sources (including local Ollama models), and versioning.
    -   **Inference**: Provides interfaces for running AI model predictions.
    -   **Routing**: Implements intelligent routing to select the optimal model based on performance, capabilities, and current demand. Supports multi-modal models (LLM, Vision, Audio).
    -   **LLM Call Detection & Replacement**: The `LLMCallDetector` (from `project.md`) is a key component, allowing the system to identify and potentially replace LLM calls within integrated agents or codebases, facilitating flexibility in model usage (e.g., switching to Ollama).

-   **Storage Layer**: Provides persistent and cached storage for the platform.
    -   **PostgreSQL**: Used as the primary relational database for structured data (e.g., agent metadata, task states, workflow definitions).
    -   **Redis Cache**: Implements caching for frequently accessed data to improve performance.
    -   **File Storage / Vector Store**: Manages unstructured data, potentially including local model storage, agent code, or vector embeddings (e.g., using ChromaDB as mentioned in `project.md`'s dependencies).

-   **Monitoring Layer**: Collects metrics, logs, and provides observability into the system's health and performance.
    -   **Prometheus**: Gathers real-time metrics from various components.
    -   **Grafana**: Offers visual dashboards for monitoring and analyzing collected metrics.
    -   **ELK Stack (Elasticsearch, Logstash, Kibana)**: Provides a centralized solution for log management, aggregation, and analysis.

-   **API Gateway Layer**: Exposes the platform's functionalities to external clients and users.
    -   **REST API**: Offers synchronous request-response communication for managing resources like agents, tasks, and workflows.
    -   **WebSocket API**: Enables real-time, bidirectional communication for features like live monitoring and agent status updates.
    -   **CLI Interface**: Provides a command-line tool for interacting with and managing the orchestrator.
    -   **Web UI**: (Future possibility) A graphical user interface for easier management and visualization.

## 3. Project Workflow

This section describes the typical operational flows within the Mark-1 AI Orchestrator, from how AI agents are managed to how complex tasks and codebases are processed.

### 3.1. Agent Lifecycle

The agent lifecycle in Mark-1 is designed to be flexible and robust, supporting various types of AI agents, including those built with LangChain, LangGraph, AutoGPT, CrewAI, or custom frameworks.

1.  **Discovery & Registration**:
    *   **Auto-Discovery**: The `agents/discovery.py` module can automatically find and identify potential agents within specified directories (e.g., `/agents/langchain`). This process might involve initial codebase scanning to identify agent patterns.
    *   **Manual Registration**: Agents can also be explicitly registered via the API or CLI.
    *   During registration, metadata about the agent (capabilities, type, required configuration, supported models) is stored in the `Agent Repository` (`storage/repositories/agent_repository.py`), likely using `agent_model.py`.

2.  **Adaptation & Integration**:
    *   The `Universal Agent Adapter` (`agents/adapter.py`) plays a crucial role in providing a standardized interface for interacting with diverse agent types.
    *   Specific integration modules (e.g., `agents/integrations/langchain_integration.py`) handle the nuances of each framework. For LangChain, this includes parsing `LangGraph` state definitions, managing tool schemas (`LangChainToolAdapter`), and adapting different agent types (ReAct, Plan-and-Execute).
    *   The results of the `Code Analysis Workflow` (see 3.3) heavily inform this stage, providing details about the agent's structure, dependencies, and required LLM calls.

3.  **Management & Pooling**:
    *   Registered and adapted agents are managed by the `Agent Pool` (`agents/pool.py`). This component handles agent instantiation, resource allocation, and availability.
    *   The orchestrator can manage a pool of active agents, scaling them based on demand and configuration.

4.  **Communication**:
    *   Inter-agent communication and communication between the orchestrator and agents are handled by the `Communication Bus` (`agents/communication/bus.py`) using defined `Protocols` (`agents/communication/protocols.py`). This might involve a message queue system for asynchronous communication.
    *   For multi-agent systems like those built with CrewAI or LangGraph, the orchestrator ensures that communication patterns and shared context are correctly managed.

5.  **Execution & Monitoring**:
    *   Agents are assigned tasks by the `Orchestrator` (see 3.2).
    *   Their performance, resource usage, and status are monitored, with data fed into the `Monitoring Layer`.

6.  **Termination & Deregistration**:
    *   Agents can be deactivated or decommissioned. The agent pool manages the graceful shutdown of agents, releasing their resources.
    *   Deregistration removes the agent from the active pool and registry.

### 3.2. Task Orchestration Flow

Task orchestration is the core function of Mark-1, involving planning, assigning, and supervising the execution of tasks by AI agents.

1.  **Task Creation**:
    *   Tasks are initiated via the API (REST or WebSocket) or CLI. A task definition includes goals, input data, required capabilities, and potentially a preferred agent or agent type.
    *   Task metadata is stored via `Task Repository` (`storage/repositories/task_repository.py`).

2.  **Task Planning (Optional)**:
    *   For complex tasks, the `Task Planner` (`core/task_planner.py`) may break down the initial request into a sequence or graph of sub-tasks. This plan forms a workflow.
    *   The `Workflow Engine` (`core/workflow_engine.py`) manages the execution of these (potentially DAG-based) workflows.

3.  **Agent Selection**:
    *   The `Agent Selector` (`core/agent_selector.py`) chooses the most appropriate agent(s) for a task or sub-task.
    *   Selection criteria include:
        *   Required capabilities (matched against agent capabilities extracted during the `Code Analysis Workflow`).
        *   Agent availability and current load (from the `Agent Pool`).
        *   Performance metrics and historical success rates.
        *   Cost or resource constraints.
        *   ML-based selection can be employed for more nuanced decisions.

4.  **Context Management**:
    *   The `Context Manager` (`core/context_manager.py`) prepares and provides the necessary context for the selected agent(s) to perform the task. This includes:
        *   Task-specific data and instructions.
        *   Relevant historical data or outputs from previous tasks in a workflow.
        *   Shared memory or state for multi-agent collaborations (especially for LangGraph or CrewAI agents).
    *   Context is stored and retrieved via `Context Repository` (`storage/repositories/context_repository.py`).

5.  **Task Execution**:
    *   The orchestrator, through the `Universal Agent Adapter`, dispatches the task to the selected agent(s).
    *   The agent executes the task, potentially interacting with:
        *   LLMs (via `Model Layer`, potentially using Ollama through `ollama_client.py`).
        *   External tools (adapted via `LangChainToolAdapter` or similar).
        *   Other agents (via the `Communication Bus`).
    *   Security measures like sandboxing (`security/sandbox.py`) may be applied during execution.

6.  **Monitoring & Error Handling**:
    *   Task progress, logs, and intermediate results are streamed back to the orchestrator and can be viewed via APIs.
    *   The `Workflow Engine` and `Orchestrator` handle errors, retries, and conditional logic based on task outcomes.
    *   Performance metrics are sent to the `Monitoring Layer`.

7.  **Result Processing & Storage**:
    *   Upon completion, the agent returns the results.
    *   The orchestrator processes these results, which might involve storing them in the `Storage Layer` (PostgreSQL, Redis, File Storage), updating task states, and triggering subsequent tasks in a workflow.

### 3.3. Code Analysis Workflow

The `Advanced Codebase Scanner` (`scanning/codebase_scanner.py`) provides crucial intelligence for integrating and managing AI agents and understanding their capabilities. This workflow is often a prerequisite for agent registration and effective task assignment.

1.  **Input**: A path to a codebase repository or specific code files.

2.  **File Discovery & Classification**:
    *   The scanner identifies all files and classifies them by language (Python, JavaScript, etc.) and type (source code, documentation, configuration).

3.  **AST Analysis**:
    *   For supported languages (Python, JS, TS, Rust, Go), the `MultiLanguageASTAnalyzer` (or language-specific parsers like `scanning/parsers/python_parser.py`) parses the code into Abstract Syntax Trees (ASTs).
    *   The `AST Analyzer` (`scanning/ast_analyzer.py`) traverses these trees to understand code structure, identify functions, classes, imports, and other constructs.

4.  **Dependency Analysis**:
    *   The `Dependency Mapper` (`scanning/dependency_mapper.py`) analyzes import statements and project configuration files (e.g., `requirements.txt`, `pyproject.toml`) to map out internal and external dependencies. This helps identify required libraries and potential compatibility issues.

5.  **LLM Call Detection**:
    *   The `LLM Call Detector` (`scanning/llm_call_detector.py`) uses predefined patterns (and potentially AST analysis) to find instances where the code makes calls to LLM APIs (e.g., OpenAI, Anthropic, or LangChain's LLM abstractions).
    *   It extracts context around these calls, which is vital for potential replacement by the `LLM Call Replacer` (`llm/call_replacer.py`) to use local models like Ollama.

6.  **Agent Pattern Recognition**:
    *   The `Agent Pattern Recognizer` (`scanning/pattern_recognizer.py`) uses a database of known patterns (signatures, keywords, structural features) and potentially ML classifiers to identify if the codebase contains known agent frameworks or patterns (e.g., LangChain ReAct, AutoGPT, CrewAI).
    *   This helps in categorizing the agent and applying appropriate integration strategies.

7.  **Capability Extraction**:
    *   The `Capability Extractor` (`scanning/capability_extractor.py`) infers the agent's functionalities. This is a multi-faceted process:
        *   **Code-based**: Analyzing function names, class methods, docstrings, and specific code patterns (e.g., tool definitions in LangChain).
        *   **Documentation-based**: Using NLP to parse README files or other documentation associated with the agent.
        *   **Configuration-based**: Analyzing agent configuration files for declared capabilities or tool integrations.
    *   Extracted capabilities are validated and scored.

8.  **Output**: A comprehensive `RepositoryAnalysis` or `AgentMetadata` object containing:
    *   Classified file list.
    *   Dependency graph.
    *   List of detected LLM calls and their contexts.
    *   Identified agent patterns and types.
    *   A list of extracted capabilities with associated confidence scores.
    *   This output is then used by the `Agent Registry`, `Agent Selector`, and `Orchestrator`.

### 3.4. Data Flow

The data flow within Mark-1 is cyclical and interconnected, ensuring that components have the information they need to function effectively.

1.  **User Interaction & Task Initiation (API Layer)**:
    *   Users or external systems interact with the `API Gateway Layer` (REST, WebSocket, CLI). Requests to create tasks, manage agents, or query system status are received here.
    *   Input data and configurations are validated (`security/input_validation.py`).

2.  **Orchestration & Agent Interaction (Core & Agent Layers)**:
    *   Task requests flow to the `Core Orchestrator`.
    *   The `Orchestrator` interacts with the `Agent Registry` and `Agent Pool` to select and allocate agents.
    *   Contextual data (from `Storage Layer` via `Context Manager`) and task instructions are passed to the agent(s).
    *   Agents, during execution, may request data from or send data to other agents via the `Communication Bus`.

3.  **Model Interaction (LLM Layer)**:
    *   Agents requiring LLM processing send requests (prompts, data) to the `Model Layer`.
    *   The `Model Manager` selects an appropriate LLM (e.g., an Ollama model via `ollama_client.py` or other configured providers).
    *   The LLM processes the input and returns a response (e.g., text generation, embedding) to the agent.
    *   LLM API calls can be replaced by the `LLM Call Replacer` based on scanner outputs and system configuration.

4.  **Data Persistence & Caching (Storage Layer)**:
    *   **Input/Output**: Task inputs, intermediate results, final outputs, and agent-generated data are stored in PostgreSQL (via `database.py` and repositories) or file storage.
    *   **Metadata**: Agent metadata, task states, workflow definitions, context data, and LLM model information are persisted in PostgreSQL.
    *   **Caching**: Frequently accessed data (e.g., agent capabilities, popular model responses) is cached in Redis (`redis_client.py`) to improve performance.
    *   **Vector Storage**: Embeddings or other vector data might be stored in ChromaDB (`vector_store.py`).

5.  **Monitoring & Logging (Monitoring Layer)**:
    *   All layers generate logs and metrics.
    *   The `Monitoring Layer` (`metrics_collector.py`, etc.) collects operational data: API request rates, task execution times, agent performance, LLM response latencies, resource utilization.
    *   Logs are aggregated (e.g., ELK stack).
    *   Metrics are stored (e.g., Prometheus) and visualized (e.g., Grafana).
    *   Alerts (`alerting.py`) can be triggered based on predefined conditions.

6.  **Feedback & Adaptation**:
    *   Data from the `Monitoring Layer` can be used by the `Orchestrator` and `Agent Selector` to optimize future task assignments and resource allocation.
    *   Analysis results from the `Scanning Layer` continuously update agent metadata in the `Storage Layer`, influencing agent selection and integration.

This interconnected data flow enables Mark-1 to operate as an intelligent and adaptive orchestration platform.

## 4. Key Features

Mark-1 AI Orchestrator offers a suite of powerful features designed to manage complex AI systems efficiently and securely.

### 4.1. Multi-Agent Coordination

Mark-1 excels at managing and coordinating diverse groups of AI agents, enabling them to work together seamlessly.

-   **Advanced Agent Management**: The system provides sophisticated agent lifecycle management, including registration (manual or via auto-discovery through `src/mark1/agents/discovery.py`), adaptation using a `UniversalAgentAdapter` (`src/mark1/agents/adapter.py`), and pooling (`src/mark1/agents/pool.py`). It supports role-based coordination and integration with various agent frameworks like LangChain (ReAct, Plan-and-Execute, and crucially, `LangGraph` for stateful, graph-based agents as detailed in `project.md`'s `LangChainOrchestrator`), AutoGPT, and CrewAI. The `AdvancedCodebaseScanner` (`src/mark1/scanning/codebase_scanner.py`) assists in understanding agent capabilities and patterns for better integration.
-   **Inter-Agent Communication**: Mark-1 facilitates high-performance message passing using a dedicated `Communication Bus` (`src/mark1/agents/communication/bus.py`) and standardized `Protocols` (`src/mark1/agents/communication/protocols.py`). This allows for complex interactions and consensus mechanisms, essential for multi-agent systems that may require shared memory or intricate communication patterns (e.g., LangChain's multi-agent setups or CrewAI crews).
-   **Conflict Resolution**: The platform includes capabilities for intelligent conflict detection and automated resolution strategies, crucial when multiple agents might provide differing solutions or attempt to control the same resources.
-   **Load Balancing**: Workloads are dynamically distributed across agent clusters by the `Agent Pool` and potentially the `Agent Selector` (`src/mark1/core/agent_selector.py`), ensuring optimal resource utilization and preventing bottlenecks.

### 4.2. AI Model Management

The platform offers robust capabilities for managing the entire lifecycle of AI models, with a focus on flexibility and performance.

-   **Dynamic Model Loading**: Mark-1 supports on-demand model discovery and loading from multiple sources. This includes integration with local model providers like Ollama (via `src/mark1/llm/ollama_client.py` and `src/mark1/llm/providers/ollama_provider.py`) and a generic `ModelManager` (`src/mark1/llm/model_manager.py`). The `LLMCallDetector` (`src/mark1/scanning/llm_call_detector.py`) can identify LLM calls in agent code, and the `LLMCallReplacer` (`src/mark1/llm/call_replacer.py`) can modify these to use managed models.
-   **Intelligent Routing**: The `Agent Selector` or a dedicated model routing mechanism within the `Model Layer` can choose the optimal model for a given task based on performance metrics, specific capabilities (e.g., context window size, fine-tuning characteristics), and current operational costs.
-   **Performance Monitoring**: Real-time model performance is tracked (e.g., response times, accuracy if applicable), with data fed to the `Monitoring Layer`. This allows for continuous optimization and identification of underperforming models.
-   **Multi-Modal Support**: The architecture is designed to integrate and manage not only Large Language Models (LLMs) but also vision, audio, and other multimodal models, making it a versatile platform for diverse AI applications.

### 4.3. Advanced Workflow Orchestration

Mark-1 enables the creation, execution, and management of complex, multi-step AI workflows with sophisticated control flow.

-   **Complex Workflows**: The `Workflow Engine` (`src/mark1/core/workflow_engine.py`) supports the definition and execution of multi-step workflows, potentially represented as Directed Acyclic Graphs (DAGs). This includes conditional execution paths, parallel processing, and robust error handling within the workflow. The `LangChainOrchestrator` can manage complex LangGraph-based workflows, handling state transitions and node dependencies.
-   **Parallel Processing**: Tasks within a workflow can be executed concurrently, with intelligent synchronization mechanisms to manage dependencies and optimize throughput.
-   **Real-Time Adaptation**: The system is designed to allow for dynamic modification of workflows during execution based on intermediate results or changing external conditions, providing adaptability for long-running or evolving tasks.
-   **Performance Optimization**: Mark-1 incorporates features for automated workflow optimization, including identifying bottlenecks, suggesting resource adjustments, and potentially re-routing tasks based on agent/model performance data collected by the `Monitoring Layer`.

### 4.4. Enterprise Security

Security is a foundational aspect of Mark-1, with features designed to protect data, manage access, and ensure operational integrity.

-   **Authentication & Authorization**: The platform implements robust authentication mechanisms, potentially including Multi-Factor Authentication (MFA), JWT tokens, and API keys, managed by `src/mark1/security/authentication.py`. Authorization is handled via Role-Based Access Control (RBAC), as outlined in `src/mark1/security/authorization.py`, allowing granular control over permissions.
-   **End-to-End Encryption**: All sensitive communications, both internal and external, are protected using protocols like TLS 1.3. Secure key management practices are enforced for encryption keys.
-   **Vulnerability Assessment & Input Validation**: Automated security scanning and compliance checks are part of the CI/CD pipeline and ongoing operations. `src/mark1/security/input_validation.py` ensures that data entering the system is sanitized to prevent common vulnerabilities. Agent sandboxing (`src/mark1/security/sandbox.py`) is planned to isolate agent execution environments.
-   **Audit Logging**: Comprehensive audit trails are generated for all significant actions within the system, providing traceability for security analysis and compliance reporting.

### 4.5. Production Monitoring

Mark-1 provides extensive monitoring capabilities to ensure system health, performance, and observability in production environments.

-   **Real-Time Metrics**: The `Metrics Collector` (`src/mark1/monitoring/metrics_collector.py`) gathers a wide array of metrics, including application metrics (request rates, error rates), business metrics (agent performance, task completion rates), and infrastructure metrics (CPU, memory). Prometheus is the designated tool for metrics collection.
-   **Visual Dashboards**: Grafana is used to provide visual dashboards, offering real-time insights into system behavior and key performance indicators (KPIs).
-   **Centralized Logging**: A centralized logging system, likely leveraging the ELK stack (Elasticsearch, Logstash, Kibana), aggregates logs from all components. Structured JSON logging with correlation IDs aids in debugging and analysis.
-   **Health Monitoring & Alerting**: Automated health checks (`src/mark1/monitoring/health_checker.py`) monitor the status of critical components. An alerting system (`src/mark1/monitoring/alerting.py`) integrates with tools like Grafana to notify operators of issues or anomalies based on predefined rules (e.g., high error rates, agent downtime).

## 5. Directory and File Structure

This section outlines the main directories and key modules within the Mark-1 AI Orchestrator project, providing an overview of how the codebase is organized. The structure is designed for modularity and scalability.

### 5.1. Main Project Directories

The Mark-1 project (`mark1-orchestrator/`) is organized into several top-level directories:

-   **`src/`**: Contains the primary source code for the Mark-1 application.
    -   **`mark1/`**: The main Python package for the orchestrator. All core logic, modules, and sub-packages reside here.
-   **`tests/`**: Houses all types of tests for the project.
    -   **`unit/`**: Unit tests for individual components and functions.
    -   **`integration/`**: Integration tests verifying interactions between components.
    -   **`e2e/`**: End-to-end tests simulating complete workflows and user scenarios.
-   **`scripts/`**: Contains utility scripts for various development and operational tasks, such as database migrations (`migrate_db.py`), data seeding (`seed_data.py`), setup (`setup.py`), and performance benchmarking (`benchmark.py`).
-   **`docs/`**: This directory. Contains all project documentation, including this comprehensive guide, API references, architecture diagrams, and usage examples. The file you are currently reading (`docs/docs.md`) is the central documentation hub.
-   **`agents/`**: A designated directory for storing discovered or integrated external AI agents, potentially categorized by framework (e.g., `langchain/`, `autogpt/`, `crewai/`, `custom/`). This allows the system to dynamically load and manage these agents.
-   **`data/`**: Used for storing persistent and transient data required or generated by the application.
    -   **`models/`**: Intended for local storage of AI models if not managed by a dedicated model registry elsewhere.
    -   **`cache/`**: For system caches that don't fit into Redis (e.g., file-based caches).
    -   **`logs/`**: Default directory for log file output if not configured to a centralized logging system.
-   **`docker/`**: Contains Docker-related files for containerizing the application and its services, including `Dockerfile.dev`, `Dockerfile.prod`, and potentially `docker-compose.override.yml` for development overrides.
-   **Configuration Files (Root)**:
    -   `README.md`: Main project readme, providing a general overview and quick start guide.
    -   `pyproject.toml`: Python project configuration, including dependencies managed by Poetry.
    -   `requirements.txt`: A list of Python dependencies, potentially generated from `pyproject.toml` for broader compatibility.
    -   `docker-compose.yml`: Defines the services, networks, and volumes for deploying Mark-1 and its dependencies (like databases, monitoring tools) using Docker Compose.
    -   `.env.example`: An example template for environment variables required to run the application.
    -   `.gitignore`: Specifies intentionally untracked files that Git should ignore.

### 5.2. Key Module Explanations

Within the main `src/mark1/` package, several sub-directories represent core modules with specific responsibilities:

-   **`src/mark1/main.py`**: The main entry point for the application, particularly for the Command Line Interface (CLI) and potentially for launching the API server.

-   **`src/mark1/config/`**: Handles configuration management.
    -   `settings.py`: Manages application settings, potentially loading from environment variables or configuration files.
    -   `logging_config.py`: Sets up structured logging for the application.
    -   `database_config.py`: Contains configurations specific to database connections.

-   **`src/mark1/core/`**: Contains the central orchestration logic.
    -   `orchestrator.py`: The main orchestration engine, responsible for coordinating tasks and agents.
    -   `task_planner.py`: Handles the planning of complex tasks, potentially breaking them into sub-tasks or workflows.
    -   `agent_selector.py`: Implements algorithms for selecting the appropriate agent(s) for a given task based on capabilities, performance, and availability.
    -   `workflow_engine.py`: Manages the execution of multi-step workflows (potentially DAGs), including state transitions and error handling.
    -   `context_manager.py`: Responsible for managing and providing the necessary context for agents during task execution.

-   **`src/mark1/agents/`**: Manages AI agents.
    -   `registry.py`: System for registering and tracking available agents and their metadata.
    -   `discovery.py`: Engine for auto-discovering agents in specified locations.
    -   `adapter.py`: Provides a universal adapter to interface with different types of agents.
    -   `pool.py`: Manages a pool of active agents, handling their lifecycle and resource allocation.
    -   `communication/`: Sub-module for inter-agent and orchestrator-agent communication (e.g., `bus.py`, `protocols.py`).
    -   `integrations/`: Contains specific integration logic for different agent frameworks (e.g., `langchain_integration.py`, `autogpt_integration.py`, `crewai_integration.py`).

-   **`src/mark1/scanning/`**: Advanced codebase analysis capabilities.
    -   `codebase_scanner.py`: The primary component for deep scanning of agent codebases.
    -   `ast_analyzer.py`: Engine for Abstract Syntax Tree (AST) analysis across multiple languages.
    -   `pattern_recognizer.py`: Identifies common agent patterns (e.g., ReAct, AutoGPT) within code.
    -   `llm_call_detector.py`: Detects LLM API calls within code for analysis or replacement.
    -   `capability_extractor.py`: Extracts agent capabilities from code, documentation, and configuration.
    -   `parsers/`: Language-specific parsers (e.g., `python_parser.py`).

-   **`src/mark1/llm/`**: Manages interactions with Large Language Models.
    -   `ollama_client.py`: Specific client for integrating with Ollama for local LLM execution.
    -   `model_manager.py`: Manages the lifecycle and selection of different LLMs.
    -   `call_replacer.py`: Engine to replace detected LLM API calls with alternatives (e.g., routing to Ollama).
    -   `providers/`: Abstractions for different LLM providers (e.g., `ollama_provider.py`).

-   **`src/mark1/storage/`**: Handles data persistence and caching.
    -   `database.py`: Abstraction layer for database interactions (likely PostgreSQL).
    -   `redis_client.py`: Client for interacting with Redis for caching.
    -   `vector_store.py`: Integration with vector databases like ChromaDB for embedding storage.
    -   `models/`: Defines data models/schemas (e.g., `agent_model.py`, `task_model.py`) for the ORM.
    -   `repositories/`: Data Access Layer for interacting with the database (e.g., `agent_repository.py`).

-   **`src/mark1/api/`**: Defines the external interfaces (REST, WebSocket, CLI).
    -   `rest_api.py`: Implements RESTful API endpoints using a framework like FastAPI.
    -   `websocket_api.py`: Implements WebSocket communication for real-time updates.
    -   `cli.py`: Defines the Command Line Interface using a library like Typer or Click.
    -   `schemas/`: Contains Pydantic schemas for API request/response validation.

-   **`src/mark1/security/`**: Implements security features.
    -   `authentication.py`: Handles user and service authentication.
    -   `authorization.py`: Manages permissions and access control (RBAC).
    -   `sandbox.py`: (Planned) Provides sandboxed environments for agent execution.
    -   `input_validation.py`: Handles sanitization and validation of input data.

-   **`src/mark1/monitoring/`**: Manages system monitoring, metrics, and alerting.
    -   `metrics_collector.py`: Collects performance and operational metrics (e.g., for Prometheus).
    -   `health_checker.py`: Monitors the health of system components.
    -   `alerting.py`: Manages alerting based on predefined rules or anomalies.

-   **`src/mark1/utils/`**: Contains utility functions, custom exceptions, constants, and decorators used across the project.

## 6. Core Modules and Their Functionality

This section delves deeper into the primary modules within the `src/mark1/` directory, detailing their responsibilities, key components, and interactions.

### 6.1. `src/mark1/config`

-   **Responsibility**: Manages all aspects of application configuration, ensuring that the system can be easily adapted to different environments (development, testing, production) and that settings are accessible and consistent.
-   **Key Sub-components**:
    -   `settings.py`: Centralizes application-wide settings, likely using Pydantic for typed configurations loaded from environment variables, `.env` files, or external configuration files. This would include database URLs, API keys, log levels, LLM provider details, etc.
    -   `logging_config.py`: Configures the application's logging behavior, possibly using `structlog` for structured logging. It defines log formats, output handlers (console, file, centralized logging service), and log levels for different modules.
    -   `database_config.py`: Specifically handles database connection parameters and potentially ORM (e.g., SQLAlchemy) configurations if not managed directly within `settings.py`.
-   **Interactions**:
    -   Provides configuration data to all other modules in the system.
    -   Interacts with the environment (e.g., reading environment variables) to load settings.
    -   The `main.py` entry point would likely initialize configurations from this module early in the application startup.

### 6.2. `src/mark1/core`

-   **Responsibility**: The heart of the Mark-1 Orchestrator, responsible for the central logic of task processing, workflow management, and agent coordination.
-   **Key Sub-components**:
    -   `orchestrator.py`: The main engine that receives tasks (likely from the `api` module), decides how they should be processed, and coordinates the necessary resources (agents, models). It oversees the entire lifecycle of a task.
    -   `task_planner.py`: If a task is complex, this component breaks it down into smaller, manageable sub-tasks or constructs a plan (workflow) for execution. It determines dependencies between sub-tasks.
    -   `agent_selector.py`: Intelligently selects the most suitable agent or agents for a given task or sub-task. It uses criteria such as agent capabilities (derived from the `scanning` module), availability (from `agents/pool.py`), performance history, and current load.
    -   `workflow_engine.py`: Executes defined workflows (potentially DAGs from the `task_planner.py`). It manages the state of each step in the workflow, handles transitions, conditional logic, and ensures tasks are executed in the correct order. It likely interacts heavily with `orchestrator.py` and `agent_selector.py`.
    -   `context_manager.py`: Manages the contextual information required by agents to perform tasks. This includes gathering necessary data from the `storage` module (via `context_repository.py`), maintaining shared state for collaborative tasks, and ensuring agents have the right inputs.
-   **Interactions**:
    -   Receives task execution requests from the `api` module.
    -   Interacts with `agents` module (specifically `registry.py`, `pool.py`, `adapter.py`) to get agent information and dispatch tasks.
    -   Uses the `llm` module to facilitate model interactions for agents.
    -   Relies on the `storage` module to persist and retrieve task, workflow, and context data.
    -   Feeds data to the `monitoring` module about task/workflow status and performance.

### 6.3. `src/mark1/agents`

-   **Responsibility**: Manages all aspects related to AI agents, including their registration, discovery, lifecycle, communication, and integration of diverse agent frameworks.
-   **Key Sub-components**:
    -   `registry.py`: Maintains a catalog of all available agents, their types, capabilities (often populated by the `scanning` module), configurations, and status.
    -   `discovery.py`: Automatically discovers potential agents from specified directories or codebases, preparing them for registration.
    -   `adapter.py`: A crucial component (`UniversalAgentAdapter`) that provides a standardized interface for the `core` module to interact with different types of agents, abstracting away the specifics of their underlying frameworks.
    -   `pool.py`: Manages a pool of active agent instances, handling their instantiation, resource allocation, scaling, and health.
    -   `communication/`:
        -   `bus.py`: Implements an inter-agent communication bus or message queue, allowing agents to exchange information or coordinate actions.
        -   `protocols.py`: Defines the standardized message formats and protocols for communication.
    -   `integrations/`: Contains specialized modules for integrating with specific agent frameworks:
        -   `langchain_integration.py`: Handles LangChain agents, including `LangChainOrchestrator` for managing legacy chains and new `LangGraph` workflows, state management, and `LangChainToolAdapter` for tool integration.
        -   `autogpt_integration.py`, `crewai_integration.py`: (As per `project.md`) Similar integration points for AutoGPT and CrewAI agents.
        -   `custom_integration.py`: Framework for integrating custom-built agents.
-   **Interactions**:
    -   The `core` module (especially `orchestrator.py` and `agent_selector.py`) uses this module extensively to find, allocate, and interact with agents.
    -   The `scanning` module provides data to `registry.py` about discovered agents' capabilities and patterns.
    -   The `storage` module (via `agent_repository.py`) persists agent metadata.
    -   Agents from this module interact with the `llm` module to use AI models.

### 6.4. `src/mark1/scanning`

-   **Responsibility**: Provides advanced capabilities for analyzing codebases, primarily those of AI agents, to understand their structure, dependencies, embedded LLM calls, operational patterns, and extract their capabilities. This information is vital for agent integration and intelligent selection.
-   **Key Sub-components**:
    -   `codebase_scanner.py` (`AdvancedCodebaseScanner`): The main orchestrator for the scanning process. It takes a repository path, classifies files, and invokes various analysis sub-modules.
    -   `ast_analyzer.py` (`MultiLanguageASTAnalyzer`): Parses source code of multiple languages (Python, JS, TS, etc.) into Abstract Syntax Trees (ASTs) and enables traversal for deep analysis.
        -   `parsers/`: Contains language-specific parsers (e.g., `python_parser.py`, `javascript_parser.py`) likely utilizing tools like Python's `ast` module or `tree-sitter`.
    -   `pattern_recognizer.py` (`AgentPatternRecognizer`): Identifies common agent patterns (e.g., LangChain ReAct, AutoGPT structure, CrewAI setup) using a combination of rule-based detection (from a `pattern_db`) and ML classification.
    -   `dependency_mapper.py`: Analyzes import statements and project files to map out dependencies of an agent.
    -   `llm_call_detector.py` (`LLMCallDetector`): Detects direct and indirect (e.g., via LangChain abstractions) calls to LLM APIs within the agent's code. It uses predefined patterns for various providers (OpenAI, Anthropic) and can extract context for potential replacement.
    -   `capability_extractor.py` (`CapabilityExtractor`): Infers agent capabilities from multiple sources: function/class analysis in code, NLP processing of documentation, and parsing of configuration files.
-   **Interactions**:
    -   Provides detailed agent analysis (metadata, capabilities, LLM usage) to the `agents/registry.py` and `agents/integrations/` modules.
    -   May interact with the `llm` module if capability extraction involves understanding model-specific functionalities.
    -   Results from this module are used by `core/agent_selector.py` to make informed decisions.

### 6.5. `src/mark1/llm`

-   **Responsibility**: Manages all interactions with Large Language Models (LLMs) and other AI models. It provides an abstraction layer for different model providers and facilitates dynamic model usage.
-   **Key Sub-components**:
    -   `model_manager.py`: Oversees the available AI models, their configurations, and loading strategies. It might handle dynamic loading/unloading of models to manage resources.
    -   `ollama_client.py`: A specific client to interact with a local Ollama instance, enabling the use of self-hosted open-source models.
    -   `call_replacer.py`: Works in conjunction with `scanning/llm_call_detector.py`. If the system is configured to replace hardcoded LLM calls in an agent's code (e.g., to switch from OpenAI to a local Ollama model), this module handles the generation of the replacement code or adapts the call at runtime.
    -   `prompt_adapter.py`: (If needed) Adapts prompts to the specific formats required by different LLM providers or models.
    -   `providers/`: Contains abstractions for different LLM sources.
        -   `base_provider.py`: Defines a common interface for all LLM providers.
        -   `ollama_provider.py`: Implements the base provider interface for Ollama models.
        -   `local_provider.py`: A generic provider for other types of locally managed models.
-   **Interactions**:
    -   Used by agents (via `agents/adapter.py` or directly within integrated agent code) to perform model inference.
    -   The `scanning` module (specifically `llm_call_detector.py`) identifies areas where this module's `call_replacer.py` might be invoked.
    -   The `core/agent_selector.py` might consider model availability and performance (from `model_manager.py`) when selecting agents.
    -   Model performance data is sent to the `monitoring` module.

### 6.6. `src/mark1/storage`

-   **Responsibility**: Handles all data persistence and caching needs of the application. It provides an abstraction layer over different storage backends.
-   **Key Sub-components**:
    -   `database.py`: Provides the core database connection management and session handling, likely using SQLAlchemy as an ORM for PostgreSQL.
    -   `redis_client.py`: Manages connections and interactions with a Redis server for caching frequently accessed data (e.g., agent capabilities, task results for short periods).
    -   `vector_store.py`: Integrates with a vector database (e.g., ChromaDB, as per `pyproject.toml`) for storing and querying embeddings, which can be used for semantic search, context retrieval, or agent memory.
    -   `models/`: Defines the ORM data models (schemas) for various entities:
        -   `agent_model.py`: Schema for agent metadata, configurations, status.
        -   `task_model.py`: Schema for task definitions, states, inputs, outputs.
        -   `context_model.py`: Schema for storing persistent context related to tasks or workflows.
    -   `repositories/`: Implements the Data Access Layer (DAL) or repository pattern, providing structured methods to Create, Read, Update, Delete (CRUD) data for each entity.
        -   `agent_repository.py`, `task_repository.py`, `context_repository.py`: Specific repositories for their respective data models.
-   **Interactions**:
    -   Used by almost all other modules:
        -   `core` module stores and retrieves task, workflow, and context data.
        -   `agents` module stores and retrieves agent registration and state data.
        -   `config` module might store certain configurations if they are dynamic.
        -   `monitoring` module might store aggregated metrics or log summaries if not using external systems exclusively.
    -   The `api` module indirectly interacts via other modules to fetch or store data related to user requests.

### 6.7. `src/mark1/api`

-   **Responsibility**: Exposes the Mark-1 Orchestrator's functionalities to the outside world through various interfaces, enabling users and external systems to interact with the platform.
-   **Key Sub-components**:
    -   `rest_api.py`: Implements the RESTful API endpoints (e.g., using FastAPI). This is the primary way for programmatic interaction, covering operations like creating/managing tasks, agents, and workflows.
    -   `websocket_api.py`: Provides WebSocket endpoints for real-time bidirectional communication. Useful for streaming task logs, live agent status updates, or receiving real-time notifications.
    -   `cli.py`: Defines the Command Line Interface (CLI) for the orchestrator (e.g., using Typer/Click, invoked via `main.py`). This allows for administrative tasks, querying status, and managing the system from the terminal.
    -   `schemas/`: Contains Pydantic models (or similar) that define the structure and validation rules for API request and response payloads (e.g., `agent_schemas.py`, `task_schemas.py`).
-   **Interactions**:
    -   Acts as the primary entry point for user and external system requests.
    -   Forwards requests to the `core` module (specifically `orchestrator.py`) for processing.
    -   Retrieves data from various modules (via the `core` or `storage` modules) to formulate API responses.
    -   Interacts with the `security` module to enforce authentication and authorization on API endpoints.

### 6.8. `src/mark1/security`

-   **Responsibility**: Implements all security-related features and mechanisms to protect the platform, its data, and ensure controlled access.
-   **Key Sub-components**:
    -   `authentication.py`: Handles user and service authentication. This could involve verifying credentials, managing JWT tokens, or integrating with OAuth2/OIDC providers.
    -   `authorization.py`: Manages permissions and access control, likely implementing Role-Based Access Control (RBAC) to define what actions different users or roles can perform.
    -   `input_validation.py`: Ensures that all data received from external sources (especially via the `api` module) is validated and sanitized to prevent injection attacks and other vulnerabilities. This often leverages the schemas defined in `api/schemas/`.
    -   `sandbox.py`: (Planned/Future) A critical component for securely executing agent code, especially untrusted or third-party agents. It would create isolated environments to limit the agent's access to system resources.
-   **Interactions**:
    -   The `api` module uses `authentication.py` and `authorization.py` to protect its endpoints.
    -   The `core` module, when executing agents, would ideally use `sandbox.py` to ensure safe execution.
    -   All modules that accept external input should, directly or indirectly, benefit from `input_validation.py`.

### 6.9. `src/mark1/monitoring`

-   **Responsibility**: Collects, analyzes, and exposes data about the system's health, performance, and operational behavior. It provides observability into the orchestrator.
-   **Key Sub-components**:
    -   `metrics_collector.py`: Gathers various metrics from different parts of the application (e.g., task execution times, agent performance, API latencies, resource usage). It likely exposes these metrics in a format compatible with Prometheus.
    -   `performance_analyzer.py`: (Potentially) Analyzes collected metrics to identify performance trends, bottlenecks, or anomalies.
    -   `health_checker.py`: Implements health check endpoints or internal mechanisms to monitor the status of critical components and dependencies (e.g., database, LLM providers).
    -   `alerting.py`: Configures and manages alerts that are triggered when certain metric thresholds are breached or when health checks fail. This would integrate with systems like Grafana alerting.
-   **Interactions**:
    -   Collects data from all other major modules (`core`, `agents`, `llm`, `api`, `storage`).
    -   Exposes metrics to external monitoring systems like Prometheus.
    -   Works with visualization tools like Grafana.
    -   The `api` module might expose a health check endpoint managed by `health_checker.py`.

### 6.10. `src/mark1/utils`

-   **Responsibility**: Provides common utility functions, helper classes, custom decorators, project-specific exceptions, and constants that are used across multiple modules within the application. This helps to avoid code duplication and centralize shared logic.
-   **Key Sub-components**:
    -   `helpers.py`: A collection of general-purpose helper functions.
    -   `decorators.py`: Custom decorators (e.g., for logging, error handling, timing) used to enhance functions or methods in other modules.
    -   `exceptions.py`: Defines a hierarchy of custom exception classes specific to the Mark-1 application, allowing for more granular error handling.
    -   `constants.py`: Stores system-wide constants, enums, or fixed configuration values that are not environment-dependent.
-   **Interactions**:
    -   This module is imported and used by virtually all other modules within `src/mark1/` as needed. It does not typically initiate interactions but rather provides reusable tools.

## 7. Technology Stack

The Mark-1 AI Orchestrator leverages a modern Python-based technology stack, incorporating various frameworks and libraries to achieve its functionality. The primary dependencies are managed via Poetry and are listed in `pyproject.toml`.

Key technologies include:

-   **Core Framework & API**:
    -   **Python**: ^3.11
    -   **FastAPI**: ^0.104.1 (High-performance web framework for building RESTful APIs)
    -   **Uvicorn**: ^0.24.0 (ASGI server for FastAPI)
    -   **Pydantic**: ^2.5.0 (Data validation and settings management)
    -   **Typer**: ^0.9.0 (For building the Command Line Interface)

-   **LangChain Ecosystem (Agent & Workflow Integration)**:
    -   **LangChain**: ^0.1.0 (Core library for building applications with LLMs)
    -   **LangChain Community**: ^0.0.10 (Community-contributed components for LangChain)
    -   **LangGraph**: ^0.0.20 (Low-level library for creating stateful, multi-actor applications with LLMs, part of LangChain)
    -   **LangSmith**: ^0.0.70 (Platform for debugging, tracing, and monitoring LangChain applications)

-   **LLM Integration**:
    -   **Ollama**: ^0.1.7 (Client library for interacting with the Ollama local LLM runner)
    -   **HTTPX**: ^0.25.2 (Asynchronous HTTP client, used by Ollama and potentially for other API interactions)

-   **Database & Storage**:
    -   **SQLAlchemy**: ^2.0.23 (SQL toolkit and Object-Relational Mapper for database interactions, likely with PostgreSQL)
    -   **Alembic**: ^1.13.0 (Database migration tool for SQLAlchemy)
    -   **Redis**: ^5.0.1 (In-memory data store, used for caching)
    -   **aioredis**: ^2.0.1 (Async Redis client library)
    -   **ChromaDB**: ^0.4.18 (Vector database for embedding storage and similarity search)

-   **Code Analysis**:
    -   **ast**: \* (Built-in Python module for Abstract Syntax Tree manipulation)
    -   **tree-sitter**: ^0.20.4 (Parser generator tool and an incremental parsing library)
    -   **tree-sitter-python**: ^0.20.4 (Python grammar for Tree-sitter)
    -   **tree-sitter-javascript**: ^0.20.3 (JavaScript grammar for Tree-sitter)
    -   **libcst**: ^1.1.0 (Concrete Syntax Tree parser and serializer library for Python)

-   **Async & Concurrency**:
    -   **asyncio**: \* (Built-in Python library for writing concurrent code using async/await syntax)
    -   **aiofiles**: ^23.2.1 (Asynchronous file operations)
    -   **Celery**: ^5.3.4 (Distributed task queue, potentially for background jobs or long-running tasks)

-   **Monitoring & Logging**:
    -   **Prometheus Client**: ^0.19.0 (Python client for exposing metrics to Prometheus)
    -   **Structlog**: ^23.2.0 (Structured logging for Python)
    -   **Rich**: ^13.7.0 (For rich text and beautiful formatting in the terminal, useful for CLI outputs and logging)

-   **Security**:
    -   **Cryptography**: ^41.0.7 (Cryptographic recipes and primitives)
    -   **PyJWT**: ^2.8.0 (JSON Web Token implementation in Python)

-   **Utilities**:
    -   **Click**: ^8.1.7 (Composable command line interface toolkit, a dependency of Typer)
    -   **Python-dotenv**: ^1.0.0 (Reads key-value pairs from a `.env` file and sets them as environment variables)
    -   **Jinja2**: ^3.1.2 (Templating engine, could be used for generating configurations or reports)

This stack provides a robust foundation for building a scalable, maintainable, and feature-rich AI orchestration platform.

## 8. Setup and Installation

This section guides you through setting up the Mark-1 AI Orchestrator for both production and local development environments.

### 8.1. Prerequisites

Before you begin, ensure you have the following prerequisites installed and configured:

-   **Docker**: Version 20.10 or newer.
-   **Docker Compose**: Version 2.0 or newer.
-   **Git**: For cloning the repository.
-   **RAM**: A minimum of 4GB RAM available for Docker containers is recommended. For development or larger deployments, 8GB+ is preferable.
-   **Python**: For local development, Python 3.11+ is required (as specified in `project.md` and `README.md`'s software dependencies).
-   **Operating System**: While Docker enables cross-platform deployment, development instructions are generally provided with Linux/Mac in mind (e.g., `source venv/bin/activate`). Windows users might need to adapt certain shell commands (e.g., `venv\Scripts\activate`).

### 8.2. Installation Steps

#### Production Deployment (Docker Compose - Recommended)

This is the recommended method for deploying a complete production stack with all services.

1.  **Clone and Configure**:
    ```bash
    # Clone the repository
    git clone https://github.com/mark1-ai/orchestrator.git
    cd mark1-orchestrator # Or the name of your cloned directory
    ```
    ```bash
    # Copy production environment template
    cp production.env .env
    ```
    Next, you need to generate security keys for `SECRET_KEY`, `JWT_SECRET`, and `ENCRYPTION_KEY`. You can use `openssl rand -base64 32` for each:
    ```bash
    openssl rand -base64 32 # Output for SECRET_KEY
    openssl rand -base64 32 # Output for JWT_SECRET
    openssl rand -base64 32 # Output for ENCRYPTION_KEY
    ```
    Edit the `.env` file with your generated values and any other necessary configurations (e.g., database credentials if not using default Docker ones, external service endpoints).
    ```bash
    nano .env
    ```

2.  **Deploy Production Stack**:
    Start all services using Docker Compose:
    ```bash
    docker-compose up -d
    ```
    This command will pull the necessary images and start the Mark-1 Orchestrator along with its dependencies (e.g., PostgreSQL, Redis, Prometheus, Grafana, ELK stack) in detached mode.

3.  **Verify Deployment**:
    Check the status of the running containers:
    ```bash
    docker-compose ps
    ```
    You can also check the health endpoint of the orchestrator (assuming it runs on port 8000 by default):
    ```bash
    curl http://localhost:8000/health
    ```

4.  **Access Services**:
    Once deployed, you can access the various services at their respective URLs (default ports shown):
    -   **Mark-1 Orchestrator API**: `http://localhost:8000`
    -   **Grafana Dashboard**: `http://localhost:3000`
    -   **Prometheus Metrics**: `http://localhost:9090`
    -   **Kibana Logs**: `http://localhost:5601`

#### Local Development Setup

This setup is intended for developers who want to run and modify the Mark-1 code directly.

1.  **Clone Repository**:
    If you haven't already, clone the repository:
    ```bash
    git clone https://github.com/mark1-ai/orchestrator.git
    cd mark1-orchestrator # Or the name of your cloned directory
    ```

2.  **Create Virtual Environment**:
    It's highly recommended to use a Python virtual environment.
    ```bash
    python3 -m venv venv
    ```
    Activate the virtual environment:
    -   Linux/Mac:
        ```bash
        source venv/bin/activate
        ```
    -   Windows:
        ```bash
        venv\Scripts\activate
        ```

3.  **Install Dependencies**:
    Install the required Python packages using `pip` and the `requirements.txt` file. If you are using Poetry (as indicated by `pyproject.toml`), you might use `poetry install` instead. The `README.md` specifies `pip`:
    ```bash
    pip install -r requirements.txt
    pip install -e .  # Installs the project in editable mode
    ```

4.  **Set Up Development Database & Services**:
    For local development, you might still want to run dependencies like databases (PostgreSQL, Redis) using Docker. The `README.md` suggests a development Docker Compose file:
    ```bash
    docker-compose -f docker-compose.dev.yml up -d postgres redis
    ```
    Ensure your `.env` file (you might copy `production.env` or a specific `development.env` template if available) is configured to point to these local services (e.g., `DATABASE_URL=postgresql://user:pass@localhost:5432/db`, `REDIS_URL=redis://localhost:6379/0`).

5.  **Configure Environment Variables**:
    Ensure you have a `.env` file with the necessary configurations for development (similar to the production setup, but potentially with different values for keys, database URLs, etc.).

6.  **Run Development Server**:
    Launch the Mark-1 application using its main entry point (as indicated in `project.md` and `README.md`):
    ```bash
    python -m mark1.main
    ```
    This will typically start the FastAPI server using Uvicorn on a development port (e.g., 8000).

7.  **Running Tests**:
    To run the test suite:
    ```bash
    pytest tests/
    ```
    For coverage reports:
    ```bash
    pytest --cov=src tests/
    ```

## 9. Usage
   - 9.1. CLI (Command Line Interface)
   - 9.2. REST API
   - 9.3. WebSocket API

## 10. Deployment
    - 10.1. Docker Compose
    - 10.2. Kubernetes
    - 10.3. Cloud Deployments

## 11. Contributing

## 12. Roadmap
