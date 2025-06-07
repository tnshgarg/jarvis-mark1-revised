#!/usr/bin/env python3
"""
Debug AST Analysis - Test our analyzer on a specific file
"""

import asyncio
from pathlib import Path
import ast

# Import our analyzer
from src.mark1.scanning.ast_analyzer import MultiLanguageASTAnalyzer, PythonASTAnalyzer


async def debug_ast_analysis():
    """Debug AST analysis on a specific CrewAI file"""
    
    test_file = Path("test_agents/crewAI-examples/starter_template/agents.py")
    
    print(f"🔍 Debugging AST Analysis on: {test_file}")
    print("=" * 60)
    
    if not test_file.exists():
        print(f"❌ File not found: {test_file}")
        return
    
    # Read the file content
    with open(test_file, 'r') as f:
        content = f.read()
    
    print("📄 FILE CONTENT:")
    print("-" * 30)
    print(content[:500] + "..." if len(content) > 500 else content)
    print()
    
    # Test basic AST parsing
    print("🔧 Testing basic AST parsing...")
    try:
        tree = ast.parse(content, filename=str(test_file))
        print("✅ AST parsing successful")
        
        # Print AST nodes
        print(f"AST root: {type(tree)}")
        print(f"Body length: {len(tree.body)}")
        
        for i, node in enumerate(tree.body):
            print(f"  Node {i}: {type(node).__name__}")
            if isinstance(node, ast.ClassDef):
                print(f"    Class: {node.name}")
                print(f"    Methods: {len([n for n in node.body if isinstance(n, ast.FunctionDef)])}")
            elif isinstance(node, ast.FunctionDef):
                print(f"    Function: {node.name}")
            elif isinstance(node, ast.Import):
                print(f"    Import: {[alias.name for alias in node.names]}")
            elif isinstance(node, ast.ImportFrom):
                print(f"    ImportFrom: {node.module} -> {[alias.name for alias in node.names]}")
        print()
        
    except Exception as e:
        print(f"❌ AST parsing failed: {e}")
        return
    
    # Test our Python analyzer
    print("🔧 Testing our PythonASTAnalyzer...")
    analyzer = PythonASTAnalyzer()
    
    try:
        result = await analyzer.parse_file(test_file)
        
        print(f"✅ Analysis completed")
        print(f"Elements found: {len(result.elements)}")
        print(f"Imports: {result.imports}")
        print(f"Framework indicators: {result.framework_indicators}")
        print(f"Agent patterns: {len(result.agent_patterns)}")
        print(f"Capabilities: {result.capabilities}")
        print(f"Errors: {result.errors}")
        
        # Print detailed elements
        if result.elements:
            print("\n📋 DETECTED ELEMENTS:")
            for i, element in enumerate(result.elements):
                print(f"  {i+1}. {element.name} ({element.node_type.value})")
                if element.decorators:
                    print(f"     Decorators: {element.decorators}")
                if element.base_classes:
                    print(f"     Base classes: {element.base_classes}")
                if element.parameters:
                    print(f"     Parameters: {element.parameters}")
        
        # Print agent patterns
        if result.agent_patterns:
            print("\n🤖 AGENT PATTERNS:")
            for i, pattern in enumerate(result.agent_patterns):
                print(f"  {i+1}. {pattern}")
        
    except Exception as e:
        print(f"❌ Analyzer failed: {e}")
        import traceback
        traceback.print_exc()


async def test_simple_ast_visitor():
    """Test a simple AST visitor to understand the issue"""
    
    test_file = Path("test_agents/crewAI-examples/starter_template/agents.py")
    
    with open(test_file, 'r') as f:
        content = f.read()
    
    print("🧪 Testing simple AST visitor...")
    
    try:
        tree = ast.parse(content)
        
        class SimpleVisitor(ast.NodeVisitor):
            def __init__(self):
                self.classes = []
                self.functions = []
                self.imports = []
                self.calls = []
            
            def visit_ClassDef(self, node):
                self.classes.append(node.name)
                print(f"Found class: {node.name}")
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                self.functions.append(node.name)
                print(f"Found function: {node.name}")
                self.generic_visit(node)
            
            def visit_Import(self, node):
                for alias in node.names:
                    self.imports.append(alias.name)
                    print(f"Found import: {alias.name}")
            
            def visit_ImportFrom(self, node):
                if node.module:
                    for alias in node.names:
                        full_import = f"{node.module}.{alias.name}"
                        self.imports.append(full_import)
                        print(f"Found import from: {full_import}")
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    self.calls.append(node.func.id)
                    print(f"Found call: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    call_name = f"...{node.func.attr}"
                    self.calls.append(call_name)
                    print(f"Found call: {call_name}")
                
                self.generic_visit(node)
        
        visitor = SimpleVisitor()
        visitor.visit(tree)
        
        print(f"\nSUMMARY:")
        print(f"Classes: {visitor.classes}")
        print(f"Functions: {visitor.functions}")
        print(f"Imports: {visitor.imports}")
        print(f"Calls: {visitor.calls}")
        
    except Exception as e:
        print(f"❌ Simple visitor failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main debug function"""
    await debug_ast_analysis()
    print("\n" + "=" * 60)
    await test_simple_ast_visitor()


if __name__ == "__main__":
    asyncio.run(main()) 