"""
Test script to verify Identity Graph integration with PhilosopherAgent.
Tests ID-RAG functionality and graph loading.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.philosopher_agents import PhilosopherAgent
from config.settings import PHILOSOPHERS

def test_identity_graph_loading():
    """Test that all 8 philosophers can load their identity graphs."""
    print("=" * 80)
    print("TESTING IDENTITY GRAPH AUTO-LOADING")
    print("=" * 80)
    
    results = {}
    
    for philosopher_key in PHILOSOPHERS.keys():
        print(f"\n{'='*80}")
        print(f"Testing: {philosopher_key.upper()}")
        print(f"{'='*80}")
        
        try:
            agent = PhilosopherAgent(philosopher_key)
            
            # Check if identity graph loaded
            if agent.identity_graph:
                core_beliefs = agent.identity_graph.get_core_beliefs()
                all_nodes = agent.identity_graph.graph.number_of_nodes()
                all_edges = agent.identity_graph.graph.number_of_edges()
                
                results[philosopher_key] = {
                    "status": "[OK] LOADED",
                    "nodes": all_nodes,
                    "edges": all_edges,
                    "core_beliefs": len(core_beliefs)
                }
                
                print(f"[OK] Identity graph loaded successfully")
                print(f"  Nodes: {all_nodes}")
                print(f"  Edges: {all_edges}")
                print(f"  Core beliefs: {len(core_beliefs)}")
                
                # Show sample core belief
                if core_beliefs:
                    print(f"\n  Sample core belief:")
                    print(f"  |- {core_beliefs[0]['content'][:100]}...")
            else:
                results[philosopher_key] = {
                    "status": "[FAIL] NOT LOADED",
                    "nodes": 0,
                    "edges": 0,
                    "core_beliefs": 0
                }
                print(f"[FAIL] No identity graph loaded")
                
        except Exception as e:
            results[philosopher_key] = {
                "status": f"[FAIL] ERROR: {e}",
                "nodes": 0,
                "edges": 0,
                "core_beliefs": 0
            }
            print(f"[FAIL] Error creating agent: {e}")
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Identity Graph Loading")
    print(f"{'='*80}")
    print(f"{'Philosopher':<15} {'Status':<20} {'Nodes':<8} {'Edges':<8} {'Core Beliefs':<15}")
    print("-" * 80)
    
    for philosopher_key, data in results.items():
        print(f"{philosopher_key:<15} {data['status']:<20} {data['nodes']:<8} {data['edges']:<8} {data['core_beliefs']:<15}")
    
    # Count successes
    loaded = sum(1 for d in results.values() if "[OK]" in d['status'])
    total = len(results)
    print(f"\n{'='*80}")
    print(f"RESULT: {loaded}/{total} philosophers successfully loaded identity graphs")
    print(f"{'='*80}")
    
    return loaded == total

def test_identity_context_retrieval():
    """Test that ID-RAG context retrieval works."""
    print(f"\n{'='*80}")
    print("TESTING ID-RAG CONTEXT RETRIEVAL")
    print(f"{'='*80}")
    
    # Test with Kant
    philosopher_key = "kant"
    print(f"\nTesting with: {philosopher_key.upper()}")
    
    try:
        agent = PhilosopherAgent(philosopher_key)
        
        if not agent.identity_graph:
            print(f"[FAIL] No identity graph loaded for {philosopher_key}")
            return False
        
        # Test query
        test_query = "What is the basis of moral duty?"
        print(f"\nTest query: '{test_query}'")
        
        # Retrieve identity context
        identity_context = agent._retrieve_identity_context(test_query, top_k=3)
        
        if identity_context:
            print(f"\n[OK] Identity context retrieved ({len(identity_context)} chars):")
            print("-" * 80)
            print(identity_context)
            print("-" * 80)
            return True
        else:
            print(f"[FAIL] No identity context retrieved")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error testing context retrieval: {e}")
        return False

def test_full_response_with_idrag():
    """Test full response generation with ID-RAG."""
    print(f"\n{'='*80}")
    print("TESTING FULL RESPONSE WITH ID-RAG")
    print(f"{'='*80}")
    
    philosopher_key = "mill"
    print(f"\nTesting with: {philosopher_key.upper()}")
    
    try:
        agent = PhilosopherAgent(philosopher_key)
        
        if not agent.identity_graph:
            print(f"[WARN] Warning: No identity graph loaded for {philosopher_key}")
        
        if not agent.chain:
            print(f"[FAIL] No RAG chain available for {philosopher_key}")
            return False
        
        # Simple test query
        test_query = "Is it morally permissible to lie to save a life?"
        print(f"\nTest query: '{test_query}'")
        
        # Generate response (with ID-RAG if available)
        print("\nGenerating response...")
        response = agent.respond(test_query, chat_history=[])
        
        print(f"\n[OK] Response generated ({len(response)} chars):")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing full response: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("IDENTITY GRAPH INTEGRATION TEST SUITE")
    print("=" * 80)
    
    # Run tests
    test1_pass = test_identity_graph_loading()
    test2_pass = test_identity_context_retrieval()
    test3_pass = test_full_response_with_idrag()
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL TEST RESULTS")
    print(f"{'='*80}")
    print(f"1. Auto-loading test:      {'[OK] PASS' if test1_pass else '[FAIL] FAIL'}")
    print(f"2. Context retrieval test: {'[OK] PASS' if test2_pass else '[FAIL] FAIL'}")
    print(f"3. Full response test:     {'[OK] PASS' if test3_pass else '[FAIL] FAIL'}")
    print(f"{'='*80}")
    
    if test1_pass and test2_pass and test3_pass:
        print("[OK] ALL TESTS PASSED - ID-RAG integration successful!")
    else:
        print("[FAIL] SOME TESTS FAILED - check output above for details")
    print(f"{'='*80}\n")
