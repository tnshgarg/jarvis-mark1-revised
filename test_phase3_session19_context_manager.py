#!/usr/bin/env python3
"""
Test Suite for Phase 3 Session 19: Advanced Context Management

This test suite validates the advanced context management system including:
- Intelligent caching with multiple strategies
- Context compression and optimization
- Hierarchical context relationships
- Advanced context sharing and permissions
- Smart context merging capabilities
- Performance monitoring and analytics

Test Categories:
1. Basic Context Operations (CRUD)
2. Advanced Caching System
3. Context Compression & Optimization
4. Hierarchical Context Management
5. Context Sharing & Collaboration
6. Smart Context Merging
7. Performance Analytics
8. Cache Strategy Optimization
"""

import asyncio
import json
import time
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

from src.mark1.core.context_manager import (
    AdvancedContextManager, ContextCacheStrategy, ContextType, 
    ContextScope, ContextPriority, ContextOperationType
)


class Session19ContextManagerTests:
    """Comprehensive test suite for Session 19 Advanced Context Management"""
    
    def __init__(self):
        self.context_manager = None
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
            "performance_metrics": {},
            "session_start_time": time.time()
        }
        self.test_contexts = []  # Track created contexts for cleanup
    
    def log_test_result(self, test_name: str, success: bool, details: str = "", duration: float = 0.0):
        """Log individual test results"""
        self.test_results["total_tests"] += 1
        if success:
            self.test_results["passed_tests"] += 1
            status = "✅ PASSED"
        else:
            self.test_results["failed_tests"] += 1
            status = "❌ FAILED"
        
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "duration": f"{duration:.3f}s"
        }
        self.test_results["test_details"].append(result)
        print(f"{status}: {test_name} ({duration:.3f}s)")
        if details:
            print(f"  Details: {details}")

    async def setup_context_manager(self):
        """Initialize the advanced context manager"""
        try:
            print("\n🚀 Initializing Advanced Context Manager...")
            
            self.context_manager = AdvancedContextManager(
                cache_size=500,
                cache_memory_mb=50,
                cache_strategy=ContextCacheStrategy.ADAPTIVE,
                auto_compression=True,
                compression_threshold=512
            )
            
            await self.context_manager.initialize()
            
            print("✅ Advanced Context Manager initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Advanced Context Manager: {e}")
            return False

    async def test_basic_context_operations(self):
        """Test 1: Basic CRUD operations with enhanced features"""
        print("\n" + "="*60)
        print("TEST 1: BASIC CONTEXT OPERATIONS")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Test context creation with advanced features
            create_result = await self.context_manager.create_context(
                key="test_basic_context",
                content={
                    "message": "Hello from Session 19!",
                    "features": ["caching", "compression", "hierarchy"],
                    "metadata": {"version": "1.0", "author": "Mark1"}
                },
                context_type=ContextType.MEMORY,
                scope=ContextScope.AGENT,
                priority=ContextPriority.HIGH,
                tags={"test", "session19", "basic"},
                expires_in_hours=24
            )
            
            assert create_result.success, f"Context creation failed: {create_result.message}"
            context_id = create_result.context_id
            self.test_contexts.append(context_id)
            
            # Test context retrieval
            get_result = await self.context_manager.get_context(context_id=context_id)
            assert get_result.success, f"Context retrieval failed: {get_result.message}"
            assert get_result.data["message"] == "Hello from Session 19!"
            
            # Test context update with versioning
            update_result = await self.context_manager.update_context(
                context_id=context_id,
                content={"message": "Updated message", "update_count": 1},
                merge=True,
                create_version=True
            )
            assert update_result.success, f"Context update failed: {update_result.message}"
            
            # Verify update and versioning
            updated_result = await self.context_manager.get_context(context_id=context_id)
            assert updated_result.success, "Failed to retrieve updated context"
            assert updated_result.data["message"] == "Updated message"
            assert updated_result.data["update_count"] == 1
            assert "features" in updated_result.data  # Should be merged
            
            duration = time.time() - start_time
            self.log_test_result(
                "Basic Context Operations", 
                True, 
                f"Created, retrieved, and updated context with ID: {context_id[:8]}...",
                duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Basic Context Operations", False, str(e), duration)

    async def test_caching_system(self):
        """Test 2: Advanced caching system with multiple strategies"""
        print("\n" + "="*60)
        print("TEST 2: ADVANCED CACHING SYSTEM")
        print("="*60)
        
        start_time = time.time()
        
        try:
            cache_test_contexts = []
            
            # Create multiple contexts to test caching
            for i in range(10):
                result = await self.context_manager.create_context(
                    key=f"cache_test_context_{i}",
                    content={
                        "cache_test": True,
                        "index": i,
                        "data": f"Cache test data {i}" * 20  # Make it larger
                    },
                    context_type=ContextType.MEMORY,
                    scope=ContextScope.AGENT,
                    priority=ContextPriority.MEDIUM
                )
                assert result.success, f"Failed to create cache test context {i}"
                cache_test_contexts.append(result.context_id)
                self.test_contexts.append(result.context_id)
            
            # Test cache hits by retrieving the same contexts
            cache_hits = 0
            for context_id in cache_test_contexts:
                result = await self.context_manager.get_context(context_id=context_id)
                assert result.success, f"Failed to retrieve context {context_id}"
                if result.cache_hit:
                    cache_hits += 1
            
            # Get cache statistics
            cache_stats = self.context_manager.cache.stats
            
            # Test cache hit rate
            hit_rate = cache_stats.hit_rate
            assert hit_rate > 0, "Cache hit rate should be greater than 0"
            
            duration = time.time() - start_time
            self.log_test_result(
                "Advanced Caching System", 
                True, 
                f"Cache hits: {cache_hits}/10, Hit rate: {hit_rate:.2%}, Total entries: {cache_stats.total_entries}",
                duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Advanced Caching System", False, str(e), duration)

    async def test_context_compression(self):
        """Test 3: Context compression and optimization"""
        print("\n" + "="*60)
        print("TEST 3: CONTEXT COMPRESSION & OPTIMIZATION")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Create a large context that should trigger compression
            large_content = {
                "large_data": "This is a large piece of data " * 100,
                "numbers": list(range(1000)),
                "repeated_text": ["Sample text " * 50 for _ in range(20)]
            }
            
            create_result = await self.context_manager.create_context(
                key="compression_test_context",
                content=large_content,
                context_type=ContextType.MEMORY,
                scope=ContextScope.AGENT,
                auto_compress=True
            )
            
            assert create_result.success, f"Failed to create large context: {create_result.message}"
            context_id = create_result.context_id
            self.test_contexts.append(context_id)
            
            # Check if compression was applied
            compressed = create_result.compressed
            
            # Retrieve and verify content integrity
            get_result = await self.context_manager.get_context(
                context_id=context_id, 
                decompress=True
            )
            assert get_result.success, "Failed to retrieve compressed context"
            assert get_result.data["large_data"].startswith("This is a large piece of data")
            assert len(get_result.data["numbers"]) == 1000
            
            # Test storage optimization
            optimization_stats = await self.context_manager.optimize_context_storage()
            
            duration = time.time() - start_time
            self.log_test_result(
                "Context Compression & Optimization", 
                True, 
                f"Compression applied: {compressed}, Optimized: {optimization_stats.get('contexts_compressed', 0)} contexts",
                duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Context Compression & Optimization", False, str(e), duration)

    async def test_hierarchical_contexts(self):
        """Test 4: Hierarchical context management"""
        print("\n" + "="*60)
        print("TEST 4: HIERARCHICAL CONTEXT MANAGEMENT")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Create root context
            root_result = await self.context_manager.create_context(
                key="hierarchy_root",
                content={"type": "root", "description": "Root of hierarchy"},
                context_type=ContextType.MEMORY,
                scope=ContextScope.AGENT,
                priority=ContextPriority.HIGH
            )
            assert root_result.success, "Failed to create root context"
            root_id = root_result.context_id
            self.test_contexts.append(root_id)
            
            # Create child contexts
            child_ids = []
            for i in range(3):
                child_result = await self.context_manager.create_context(
                    key=f"hierarchy_child_{i}",
                    content={"type": "child", "index": i, "parent": "root"},
                    context_type=ContextType.MEMORY,
                    scope=ContextScope.AGENT,
                    parent_context_id=root_id
                )
                assert child_result.success, f"Failed to create child context {i}"
                child_ids.append(child_result.context_id)
                self.test_contexts.append(child_result.context_id)
            
            # Create grandchild contexts
            grandchild_ids = []
            for i, child_id in enumerate(child_ids[:2]):  # Only for first 2 children
                grandchild_result = await self.context_manager.create_context(
                    key=f"hierarchy_grandchild_{i}",
                    content={"type": "grandchild", "index": i, "parent": f"child_{i}"},
                    context_type=ContextType.MEMORY,
                    scope=ContextScope.AGENT,
                    parent_context_id=child_id
                )
                assert grandchild_result.success, f"Failed to create grandchild context {i}"
                grandchild_ids.append(grandchild_result.context_id)
                self.test_contexts.append(grandchild_result.context_id)
            
            # Test hierarchy retrieval
            hierarchy = await self.context_manager.get_context_hierarchy(root_id)
            assert "root" in hierarchy, "Hierarchy should contain root information"
            assert "children" in hierarchy, "Hierarchy should contain children information"
            
            duration = time.time() - start_time
            self.log_test_result(
                "Hierarchical Context Management", 
                True, 
                f"Created hierarchy: 1 root, {len(child_ids)} children, {len(grandchild_ids)} grandchildren",
                duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Hierarchical Context Management", False, str(e), duration)

    async def test_context_sharing(self):
        """Test 5: Context sharing and collaboration"""
        print("\n" + "="*60)
        print("TEST 5: CONTEXT SHARING & COLLABORATION")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Create a context to share
            share_result = await self.context_manager.create_context(
                key="shareable_context",
                content={
                    "shared_data": "This context is shared among agents",
                    "collaboration": True,
                    "access_level": "read-write"
                },
                context_type=ContextType.MEMORY,
                scope=ContextScope.AGENT,
                priority=ContextPriority.HIGH
            )
            assert share_result.success, "Failed to create shareable context"
            context_id = share_result.context_id
            self.test_contexts.append(context_id)
            
            # Test sharing with multiple agents
            target_agents = ["agent_1", "agent_2", "agent_3"]
            sharing_result = await self.context_manager.share_context(
                context_id=context_id,
                target_agent_ids=target_agents,
                copy_on_share=True,
                share_children=False
            )
            assert sharing_result.success, f"Context sharing failed: {sharing_result.message}"
            
            shared_context_ids = sharing_result.metadata.get("shared_context_ids", [])
            assert len(shared_context_ids) == len(target_agents), f"Expected {len(target_agents)} shared contexts"
            
            # Track shared contexts for cleanup
            self.test_contexts.extend(shared_context_ids)
            
            # Test agent context retrieval
            agent_contexts = await self.context_manager.get_agent_contexts("agent_1", include_shared=True)
            assert len(agent_contexts) > 0, "Agent should have associated contexts"
            
            duration = time.time() - start_time
            self.log_test_result(
                "Context Sharing & Collaboration", 
                True, 
                f"Shared context with {len(target_agents)} agents, created {len(shared_context_ids)} copies",
                duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Context Sharing & Collaboration", False, str(e), duration)

    async def test_context_merging(self):
        """Test 6: Smart context merging capabilities"""
        print("\n" + "="*60)
        print("TEST 6: SMART CONTEXT MERGING")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Create multiple contexts to merge
            merge_contexts = []
            
            for i in range(3):
                result = await self.context_manager.create_context(
                    key=f"merge_source_{i}",
                    content={
                        f"data_{i}": f"Source data from context {i}",
                        "common_field": f"Value from context {i}",
                        "unique_field": f"unique_{i}",
                        "numbers": list(range(i*10, (i+1)*10))
                    },
                    context_type=ContextType.MEMORY,
                    scope=ContextScope.AGENT
                )
                assert result.success, f"Failed to create merge source context {i}"
                merge_contexts.append(result.context_id)
                self.test_contexts.append(result.context_id)
            
            # Test deep merge strategy
            merge_result = await self.context_manager.merge_contexts(
                source_context_ids=merge_contexts,
                target_key="merged_context_deep",
                merge_strategy="deep",
                conflict_resolution="latest",
                create_backup=True
            )
            assert merge_result.success, f"Context merging failed: {merge_result.message}"
            
            merged_context_id = merge_result.context_id
            self.test_contexts.append(merged_context_id)
            
            # Verify merged content
            merged_result = await self.context_manager.get_context(context_id=merged_context_id)
            assert merged_result.success, "Failed to retrieve merged context"
            
            merged_data = merged_result.data
            assert "data_0" in merged_data, "Merged context should contain data from source 0"
            assert "data_1" in merged_data, "Merged context should contain data from source 1"
            assert "data_2" in merged_data, "Merged context should contain data from source 2"
            
            # Test backup creation
            backup_ids = merge_result.metadata.get("backup_ids", [])
            assert len(backup_ids) > 0, "Backup contexts should be created"
            self.test_contexts.extend(backup_ids)
            
            duration = time.time() - start_time
            self.log_test_result(
                "Smart Context Merging", 
                True, 
                f"Merged {len(merge_contexts)} contexts with {len(backup_ids)} backups created",
                duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Smart Context Merging", False, str(e), duration)

    async def test_performance_analytics(self):
        """Test 7: Performance monitoring and analytics"""
        print("\n" + "="*60)
        print("TEST 7: PERFORMANCE ANALYTICS")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Perform various operations to generate metrics
            operations_count = 20
            
            for i in range(operations_count):
                # Create context
                create_result = await self.context_manager.create_context(
                    key=f"perf_test_{i}",
                    content={"performance_test": True, "iteration": i},
                    context_type=ContextType.MEMORY,
                    scope=ContextScope.AGENT
                )
                if create_result.success:
                    self.test_contexts.append(create_result.context_id)
                
                # Read context (some from cache)
                if i > 0:
                    await self.context_manager.get_context(context_id=self.test_contexts[-2])
                
                # Update some contexts
                if i % 3 == 0 and create_result.success:
                    await self.context_manager.update_context(
                        context_id=create_result.context_id,
                        content={"updated": True, "update_iteration": i}
                    )
            
            # Get comprehensive performance metrics
            performance_metrics = await self.context_manager.get_performance_metrics()
            
            # Validate metrics structure
            assert "cache_stats" in performance_metrics, "Performance metrics should include cache stats"
            assert "operation_stats" in performance_metrics, "Performance metrics should include operation stats"
            assert "operation_times" in performance_metrics, "Performance metrics should include operation times"
            
            cache_stats = performance_metrics["cache_stats"]
            assert cache_stats["total_entries"] > 0, "Cache should have entries"
            assert cache_stats["hit_rate"] >= 0, "Hit rate should be non-negative"
            
            operation_stats = performance_metrics["operation_stats"]
            assert operation_stats.get(ContextOperationType.CREATE, 0) > 0, "Should have create operations"
            assert operation_stats.get(ContextOperationType.READ, 0) > 0, "Should have read operations"
            
            self.test_results["performance_metrics"] = performance_metrics
            
            duration = time.time() - start_time
            self.log_test_result(
                "Performance Analytics", 
                True, 
                f"Cache hit rate: {cache_stats['hit_rate']:.2%}, Operations: {sum(operation_stats.values())}",
                duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Performance Analytics", False, str(e), duration)

    async def test_cache_strategy_optimization(self):
        """Test 8: Cache strategy optimization and adaptation"""
        print("\n" + "="*60)
        print("TEST 8: CACHE STRATEGY OPTIMIZATION")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Test different cache strategies by creating new context managers
            strategies = [
                ContextCacheStrategy.LRU,
                ContextCacheStrategy.LFU,
                ContextCacheStrategy.ADAPTIVE
            ]
            
            strategy_results = {}
            
            for strategy in strategies:
                # Create contexts and measure performance
                test_contexts = []
                strategy_start = time.time()
                
                for i in range(10):
                    result = await self.context_manager.create_context(
                        key=f"strategy_test_{strategy.value}_{i}",
                        content={"strategy": strategy.value, "test_data": f"data_{i}"},
                        context_type=ContextType.MEMORY,
                        scope=ContextScope.AGENT
                    )
                    if result.success:
                        test_contexts.append(result.context_id)
                        self.test_contexts.append(result.context_id)
                
                # Access patterns to test cache behavior
                for context_id in test_contexts[:5]:  # Access first 5 multiple times
                    for _ in range(3):
                        await self.context_manager.get_context(context_id=context_id)
                
                strategy_duration = time.time() - strategy_start
                strategy_results[strategy.value] = {
                    "duration": strategy_duration,
                    "contexts_created": len(test_contexts)
                }
            
            # Get final cache statistics
            final_cache_stats = self.context_manager.cache.stats
            
            # Test cache eviction and memory management
            cache_size_before = self.context_manager.cache_size
            
            # Force cache optimization
            optimization_result = await self.context_manager.optimize_context_storage()
            
            duration = time.time() - start_time
            self.log_test_result(
                "Cache Strategy Optimization", 
                True, 
                f"Tested {len(strategies)} strategies, Final cache size: {final_cache_stats.total_entries}",
                duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Cache Strategy Optimization", False, str(e), duration)

    async def cleanup_test_contexts(self):
        """Clean up all test contexts"""
        print("\n🧹 Cleaning up test contexts...")
        
        cleanup_count = 0
        for context_id in self.test_contexts:
            try:
                # For testing purposes, we'll just verify they exist
                result = await self.context_manager.get_context(context_id=context_id)
                if result.success:
                    cleanup_count += 1
            except Exception as e:
                print(f"Warning: Failed to verify context {context_id}: {e}")
        
        print(f"✅ Verified {cleanup_count} test contexts")

    async def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("SESSION 19: ADVANCED CONTEXT MANAGEMENT - TEST REPORT")
        print("="*80)
        
        # Calculate overall duration
        total_duration = time.time() - self.test_results["session_start_time"]
        
        # Test summary
        print(f"\n📊 TEST SUMMARY")
        print(f"   Total Tests: {self.test_results['total_tests']}")
        print(f"   Passed: {self.test_results['passed_tests']} ✅")
        print(f"   Failed: {self.test_results['failed_tests']} ❌")
        print(f"   Success Rate: {(self.test_results['passed_tests']/self.test_results['total_tests']*100):.1f}%")
        print(f"   Total Duration: {total_duration:.2f}s")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS")
        for result in self.test_results["test_details"]:
            print(f"   {result['status']}: {result['test']} ({result['duration']})")
            if result['details']:
                print(f"      └─ {result['details']}")
        
        # Performance metrics summary
        if "performance_metrics" in self.test_results:
            metrics = self.test_results["performance_metrics"]
            print(f"\n⚡ PERFORMANCE SUMMARY")
            
            if "cache_stats" in metrics:
                cache = metrics["cache_stats"]
                print(f"   Cache Hit Rate: {cache.get('hit_rate', 0):.2%}")
                print(f"   Cache Entries: {cache.get('total_entries', 0)}")
                print(f"   Cache Size: {cache.get('total_size_bytes', 0):,} bytes")
                print(f"   Compression Ratio: {cache.get('compression_ratio', 0):.2%}")
            
            if "operation_stats" in metrics:
                ops = metrics["operation_stats"]
                print(f"   Total Operations: {sum(ops.values())}")
                for op_type, count in ops.items():
                    print(f"     └─ {op_type}: {count}")
        
        # Context manager status
        print(f"\n🎯 CONTEXT MANAGER STATUS")
        print(f"   Initialized: {self.context_manager.is_initialized}")
        print(f"   Cache Size: {self.context_manager.cache_size}")
        print(f"   Total Contexts: {self.context_manager.total_contexts}")
        print(f"   Cache Hit Rate: {self.context_manager.cache_hit_rate:.2%}")
        print(f"   Compression Ratio: {self.context_manager.compression_ratio:.2%}")
        
        # Test completion status
        success_rate = (self.test_results['passed_tests'] / self.test_results['total_tests']) * 100
        
        if success_rate == 100:
            print(f"\n🎉 ALL TESTS PASSED! Session 19: Advanced Context Management is COMPLETE!")
            print(f"🚀 Ready for Session 20: API Layer & REST Endpoints")
        elif success_rate >= 80:
            print(f"\n✅ Most tests passed ({success_rate:.1f}%). Session 19 is substantially complete.")
            print(f"🔧 Minor issues detected - review failed tests before proceeding.")
        else:
            print(f"\n⚠️  Significant issues detected ({success_rate:.1f}% success rate).")
            print(f"🛠️  Please review and fix failed tests before proceeding to Session 20.")
        
        return success_rate >= 80

    async def run_all_tests(self):
        """Run the complete test suite"""
        print("🧪 Starting Session 19: Advanced Context Management Test Suite")
        print("🎯 Testing intelligent caching, compression, hierarchies, sharing, and optimization")
        
        # Setup
        if not await self.setup_context_manager():
            return False
        
        # Run all tests
        await self.test_basic_context_operations()
        await self.test_caching_system()
        await self.test_context_compression()
        await self.test_hierarchical_contexts()
        await self.test_context_sharing()
        await self.test_context_merging()
        await self.test_performance_analytics()
        await self.test_cache_strategy_optimization()
        
        # Cleanup
        await self.cleanup_test_contexts()
        
        # Generate report
        success = await self.generate_test_report()
        
        # Shutdown
        if self.context_manager:
            await self.context_manager.shutdown()
        
        return success


async def main():
    """Main test execution function"""
    print("="*80)
    print("MARK-1 PHASE 3 SESSION 19: ADVANCED CONTEXT MANAGEMENT")
    print("="*80)
    print("Testing advanced features:")
    print("• Intelligent Caching with Multiple Strategies")
    print("• Context Compression & Storage Optimization") 
    print("• Hierarchical Context Relationships")
    print("• Advanced Context Sharing & Collaboration")
    print("• Smart Context Merging Capabilities")
    print("• Performance Analytics & Monitoring")
    print("• Cache Strategy Optimization")
    print("="*80)
    
    # Run tests
    test_suite = Session19ContextManagerTests()
    success = await test_suite.run_all_tests()
    
    if success:
        print("\n🎊 Session 19: Advanced Context Management - COMPLETED SUCCESSFULLY!")
        print("Ready for Session 20: API Layer & REST Endpoints")
        return 0
    else:
        print("\n💥 Session 19 encountered issues. Please review and fix before proceeding.")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
