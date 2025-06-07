#!/usr/bin/env python3
"""
Phase 2 Real Agent Testing Script

Tests our advanced scanning and analysis capabilities against
a real open-source AI agent project.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import structlog

# Import our Phase 2 components
from src.mark1.scanning.codebase_scanner import CodebaseScanner
from src.mark1.scanning.ast_analyzer import MultiLanguageASTAnalyzer
from src.mark1.scanning.llm_call_detector import LLMCallDetector


async def test_real_agent_analysis(agent_path: Path) -> Dict[str, Any]:
    """
    Comprehensive analysis of a real AI agent project
    
    Args:
        agent_path: Path to the agent project directory
        
    Returns:
        Complete analysis results
    """
    print(f"🤖 Testing Mark-1 Phase 2 Analysis on Real Agent Project")
    print(f"📁 Agent Path: {agent_path}")
    print("=" * 80)
    
    if not agent_path.exists():
        print(f"❌ Error: Path {agent_path} does not exist!")
        return {"error": "Path does not exist"}
    
    # Initialize our analyzers
    print("🔧 Initializing Mark-1 Phase 2 Components...")
    scanner = CodebaseScanner()
    ast_analyzer = MultiLanguageASTAnalyzer()
    llm_detector = LLMCallDetector()
    
    await scanner.initialize()
    
    print("✅ All components initialized successfully!")
    print()
    
    # Start comprehensive analysis
    start_time = time.time()
    
    print("🔍 Starting Comprehensive Codebase Analysis...")
    print("-" * 50)
    
    try:
        # Perform full scan with all features enabled
        scan_results = await scanner.scan_directory(
            directory=agent_path,
            recursive=True,
            include_ast_analysis=True,
            include_llm_detection=True
        )
        
        analysis_time = time.time() - start_time
        
        # Generate detailed report
        report = await generate_detailed_report(scan_results, analysis_time)
        
        # Print results
        print_analysis_results(report)
        
        return report
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def generate_detailed_report(scan_results, analysis_time: float) -> Dict[str, Any]:
    """Generate a comprehensive analysis report"""
    
    # Basic metrics
    total_files = scan_results.total_files_scanned
    agents_found = len(scan_results.discovered_agents)
    ast_results_count = len(scan_results.ast_analysis_results)
    llm_results_count = len(scan_results.llm_detection_results)
    
    # Framework analysis
    frameworks = scan_results.framework_distribution
    languages = scan_results.language_distribution
    
    # Quality metrics
    quality_metrics = scan_results.code_quality_metrics
    
    # Migration opportunities
    migration_data = scan_results.migration_opportunities
    
    # Detailed agent analysis
    agent_details = []
    for agent in scan_results.discovered_agents:
        agent_details.append({
            "name": agent.name,
            "file": str(agent.file_path),
            "framework": agent.framework,
            "capabilities": agent.capabilities,
            "confidence": agent.confidence,
            "class_name": agent.class_name,
            "metadata": agent.metadata
        })
    
    # AST Analysis summary
    ast_summary = {
        "files_analyzed": ast_results_count,
        "total_elements": sum(len(r.elements) for r in scan_results.ast_analysis_results),
        "total_patterns": sum(len(r.agent_patterns) for r in scan_results.ast_analysis_results),
        "frameworks_detected": set(),
        "capabilities_found": set()
    }
    
    for result in scan_results.ast_analysis_results:
        ast_summary["frameworks_detected"].update(result.framework_indicators)
        ast_summary["capabilities_found"].update(result.capabilities)
    
    ast_summary["frameworks_detected"] = list(ast_summary["frameworks_detected"])
    ast_summary["capabilities_found"] = list(ast_summary["capabilities_found"])
    
    # LLM Detection summary
    llm_summary = {
        "files_with_llm_calls": llm_results_count,
        "total_llm_calls": sum(r.total_calls for r in scan_results.llm_detection_results),
        "providers_found": set(),
        "estimated_monthly_cost": sum(r.estimated_monthly_cost for r in scan_results.llm_detection_results),
        "replacement_opportunities": sum(len(r.replacement_suggestions) for r in scan_results.llm_detection_results)
    }
    
    for result in scan_results.llm_detection_results:
        llm_summary["providers_found"].update(p.value for p in result.providers_found)
    
    llm_summary["providers_found"] = list(llm_summary["providers_found"])
    
    return {
        "analysis_summary": {
            "total_analysis_time": round(analysis_time, 2),
            "scan_path": str(scan_results.scan_path),
            "total_files_scanned": total_files,
            "agents_discovered": agents_found,
            "errors": len(scan_results.errors)
        },
        "distribution": {
            "frameworks": frameworks,
            "languages": languages
        },
        "code_quality": quality_metrics,
        "agent_details": agent_details,
        "ast_analysis": ast_summary,
        "llm_detection": llm_summary,
        "migration_opportunities": migration_data,
        "errors": scan_results.errors
    }


def print_analysis_results(report: Dict[str, Any]):
    """Print formatted analysis results"""
    
    print("📊 ANALYSIS RESULTS")
    print("=" * 80)
    
    # Summary
    summary = report["analysis_summary"]
    print(f"⏱️  Analysis Time: {summary['total_analysis_time']} seconds")
    print(f"📁 Files Scanned: {summary['total_files_scanned']}")
    print(f"🤖 Agents Found: {summary['agents_discovered']}")
    print(f"❌ Errors: {summary['errors']}")
    print()
    
    # Distribution
    distribution = report["distribution"]
    if distribution["frameworks"]:
        print("📚 FRAMEWORK DISTRIBUTION")
        print("-" * 30)
        for framework, count in distribution["frameworks"].items():
            print(f"  {framework}: {count} agents")
        print()
    
    if distribution["languages"]:
        print("💻 LANGUAGE DISTRIBUTION") 
        print("-" * 30)
        for language, count in distribution["languages"].items():
            print(f"  {language}: {count} files")
        print()
    
    # Agent Details
    agents = report["agent_details"]
    if agents:
        print("🤖 DISCOVERED AGENTS")
        print("-" * 30)
        for i, agent in enumerate(agents, 1):
            print(f"  {i}. {agent['name']}")
            print(f"     📁 File: {Path(agent['file']).name}")
            print(f"     🔧 Framework: {agent['framework']}")
            print(f"     🎯 Confidence: {agent['confidence']:.2f}")
            if agent['capabilities']:
                print(f"     ⚡ Capabilities: {', '.join(agent['capabilities'][:3])}{'...' if len(agent['capabilities']) > 3 else ''}")
            print()
    
    # AST Analysis
    ast_data = report["ast_analysis"]
    print("🔍 AST ANALYSIS RESULTS")
    print("-" * 30)
    print(f"  Files Analyzed: {ast_data['files_analyzed']}")
    print(f"  Code Elements: {ast_data['total_elements']}")
    print(f"  Agent Patterns: {ast_data['total_patterns']}")
    if ast_data['frameworks_detected']:
        print(f"  Frameworks: {', '.join(ast_data['frameworks_detected'])}")
    print(f"  Capabilities: {len(ast_data['capabilities_found'])} unique")
    print()
    
    # LLM Detection
    llm_data = report["llm_detection"]
    if llm_data['total_llm_calls'] > 0:
        print("🔗 LLM CALL DETECTION")
        print("-" * 30)
        print(f"  Files with LLM calls: {llm_data['files_with_llm_calls']}")
        print(f"  Total LLM calls: {llm_data['total_llm_calls']}")
        print(f"  Providers found: {', '.join(llm_data['providers_found'])}")
        print(f"  Est. monthly cost: ${llm_data['estimated_monthly_cost']:.2f}")
        print(f"  Replacement opportunities: {llm_data['replacement_opportunities']}")
        print()
    
    # Code Quality
    quality = report["code_quality"]
    if quality:
        print("📈 CODE QUALITY METRICS")
        print("-" * 30)
        print(f"  Documentation ratio: {quality.get('average_documentation_ratio', 0):.2f}")
        print(f"  Average complexity: {quality.get('average_complexity', 0):.2f}")
        print(f"  Analysis success rate: {quality.get('analysis_success_rate', 0):.2f}")
        print()
    
    # Migration Opportunities
    migration = report["migration_opportunities"]
    if migration and migration.get('total_opportunities', 0) > 0:
        print("🚀 MIGRATION OPPORTUNITIES")
        print("-" * 30)
        complexity = migration.get('migration_complexity', 'unknown')
        print(f"  Migration complexity: {complexity}")
        
        if 'summary' in migration:
            summary = migration['summary']
            print(f"  Potential savings: ${summary.get('potential_savings', 0):.2f}/month")
            print(f"  Total LLM calls: {summary.get('total_llm_calls', 0)}")
        print()
    
    print("✅ Analysis Complete!")
    print("=" * 80)


async def test_with_sample_agents():
    """Test with some sample agent projects"""
    
    # Test projects to try (you can download these)
    test_projects = [
        {
            "name": "CrewAI Examples",
            "url": "https://github.com/joaomdmoura/crewAI-examples",
            "local_path": "test_agents/crewai-examples"
        },
        {
            "name": "LangChain Agents",
            "url": "https://github.com/langchain-ai/langchain/tree/master/templates",
            "local_path": "test_agents/langchain-templates"
        },
        {
            "name": "AutoGPT",
            "url": "https://github.com/Significant-Gravitas/AutoGPT",
            "local_path": "test_agents/autogpt"
        }
    ]
    
    print("🧪 SUGGESTED TEST PROJECTS")
    print("=" * 50)
    for i, project in enumerate(test_projects, 1):
        print(f"{i}. {project['name']}")
        print(f"   URL: {project['url']}")
        print(f"   Local path: {project['local_path']}")
        print()
    
    # Look for existing test projects
    test_dir = Path("test_agents")
    if test_dir.exists():
        existing_projects = [d for d in test_dir.iterdir() if d.is_dir()]
        if existing_projects:
            print("📁 FOUND EXISTING TEST PROJECTS:")
            for project in existing_projects:
                print(f"   - {project.name}")
            print()
            
            # Test the first one found
            first_project = existing_projects[0]
            print(f"🧪 Testing with: {first_project.name}")
            await test_real_agent_analysis(first_project)
    else:
        print("💡 To test with real agents, download one of the above projects to test_agents/")


async def create_sample_agent_for_testing():
    """Create a sample agent project for testing if no real ones are available"""
    
    sample_dir = Path("test_agents/sample_agent")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a sample LangChain agent
    sample_agent_code = '''"""
Sample LangChain Agent for Testing Mark-1 Analysis
"""

import os
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


class SampleLangChainAgent:
    """A sample LangChain agent for testing purposes"""
    
    def __init__(self):
        # Initialize OpenAI client (this should be detected by LLM scanner)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.memory = ConversationBufferMemory()
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self):
        """Create agent tools"""
        def search_tool(query: str) -> str:
            """Search for information online"""
            return f"Search results for: {query}"
        
        def calculator_tool(expression: str) -> str:
            """Calculate mathematical expressions"""
            try:
                result = eval(expression)
                return f"Result: {result}"
            except:
                return "Invalid expression"
        
        return [
            Tool(
                name="search",
                description="Search for information online",
                func=search_tool
            ),
            Tool(
                name="calculator", 
                description="Calculate mathematical expressions",
                func=calculator_tool
            )
        ]
    
    def _create_agent(self):
        """Create the ReAct agent"""
        prompt = PromptTemplate.from_template("""
        You are a helpful assistant. Use the following tools to answer questions:
        
        {tools}
        
        Use this format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Question: {input}
        Thought: {agent_scratchpad}
        """)
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    async def run(self, query: str) -> str:
        """Run the agent with a query"""
        try:
            result = await self.agent.ainvoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat(self, message: str) -> str:
        """Synchronous chat interface"""
        # This will be detected as another LLM call
        response = self.llm.invoke(message)
        return response.content


async def main():
    """Main function for testing"""
    agent = SampleLangChainAgent()
    
    # Test queries
    queries = [
        "What is 15 * 24?",
        "Search for information about Python programming",
        "Hello, how are you?"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        result = await agent.run(query)
        print(f"Result: {result}")
        print("-" * 50)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
'''
    
    # Write the sample agent
    with open(sample_dir / "agent.py", "w") as f:
        f.write(sample_agent_code)
    
    # Create requirements.txt
    requirements = '''langchain>=0.1.0
langchain-openai>=0.1.0
openai>=1.0.0
'''
    
    with open(sample_dir / "requirements.txt", "w") as f:
        f.write(requirements)
    
    # Create a simple config file
    config = '''{
    "agent_name": "SampleLangChainAgent",
    "framework": "langchain",
    "model": "gpt-3.5-turbo",
    "tools": ["search", "calculator"],
    "memory_type": "conversation_buffer"
}'''
    
    with open(sample_dir / "config.json", "w") as f:
        f.write(config)
    
    print(f"✅ Created sample agent at: {sample_dir}")
    return sample_dir


async def main():
    """Main test execution"""
    print("🚀 Mark-1 Phase 2 Real Agent Testing")
    print("=" * 80)
    
    # Check for existing test agents
    await test_with_sample_agents()
    
    # If no real agents found, create and test with sample
    test_dir = Path("test_agents")
    if not test_dir.exists() or not any(test_dir.iterdir()):
        print("📝 Creating sample agent for testing...")
        sample_path = await create_sample_agent_for_testing()
        
        print("\n🧪 Testing with sample agent...")
        await test_real_agent_analysis(sample_path)
    
    print("\n" + "=" * 80)
    print("🎯 NEXT STEPS:")
    print("1. Download a real open-source agent project")
    print("2. Place it in test_agents/ directory") 
    print("3. Run this script again for comprehensive analysis")
    print("4. Review results and proceed to Phase 3!")


if __name__ == "__main__":
    asyncio.run(main()) 