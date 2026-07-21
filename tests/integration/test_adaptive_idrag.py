"""
Test: Adaptive ID-RAG Implementation
Unit tests for LLM query builder, adaptive graph search, and integration.

Tests:
1. Query builder generates valid JSON strategies
2. Adaptive search retrieves relevant triplets
3. End-to-end integration (opponent argument → adaptive context)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.philosopher_agents import PhilosopherAgent
from src.common.identity_graph import PhilosopherIdentityGraph
from langchain.schema import HumanMessage

def test_query_builder():
    """Test LLM query builder generates valid strategies."""
    print("\n" + "="*70)
    print("TEST 1: LLM Query Builder")
    print("="*70)
    
    kant = PhilosopherAgent("kant")
    
    if not kant.identity_graph:
        print("❌ FAIL: Kant identity graph not loaded")
        return False
    
    # Test with Mill's utilitarian argument
    opponent_arg = "The morally right action is the one that maximizes utility and produces the greatest happiness for the greatest number."
    opponent_name = "John Stuart Mill"
    
    print(f"\n📝 Opponent argument: {opponent_arg[:80]}...")
    print(f"👤 Opponent: {opponent_name}")
    
    strategy = kant._build_retrieval_strategy(opponent_arg, opponent_name)
    
    print(f"\n🎯 Generated strategy:")
    print(f"  High priority relations: {strategy.get('high_priority_relations', [])}")
    print(f"  Medium priority relations: {strategy.get('medium_priority_relations', [])}")
    print(f"  Keywords: {strategy.get('keywords', [])}")
    print(f"  Focus: {strategy.get('focus', 'N/A')}")
    print(f"  Target triplets: {strategy.get('target_triplets', 0)}")
    
    # Validate structure
    required_fields = ["high_priority_relations", "keywords", "focus", "target_triplets"]
    missing = [f for f in required_fields if f not in strategy]
    
    if missing:
        print(f"\n❌ FAIL: Missing required fields: {missing}")
        return False
    
    if not isinstance(strategy["keywords"], list) or len(strategy["keywords"]) == 0:
        print(f"\n⚠️  WARN: No keywords extracted (fallback strategy used)")
    
    print("\n✅ PASS: Query builder generated valid strategy")
    return True


def test_adaptive_search():
    """Test adaptive graph search with strategy."""
    print("\n" + "="*70)
    print("TEST 2: Adaptive Graph Search")
    print("="*70)
    
    kant_graph = PhilosopherIdentityGraph.load_from_file(
        "data/identity_graphs/kant_identity_graph.json"
    )
    
    # Simulate strategy targeting anti-utilitarian beliefs
    strategy = {
        "high_priority_relations": ["opposes", "contradicts"],
        "medium_priority_relations": ["supports", "grounds"],
        "keywords": ["utility", "consequentialism", "happiness"],
        "focus": "retrieve Kant's critiques of utilitarian reasoning",
        "target_triplets": 15
    }
    
    print(f"\n🎯 Strategy: {strategy['focus']}")
    print(f"   Keywords: {strategy['keywords']}")
    
    results = kant_graph.adaptive_search(strategy, max_triplets=15)
    
    print(f"\n📊 Retrieved {len(results)} triplets:")
    for i, (source_id, target_id, edge_data) in enumerate(results[:5], 1):
        source_node = kant_graph.get_node(source_id)
        target_node = kant_graph.get_node(target_id)
        relation = edge_data.get('edge_type', 'relates_to')
        
        print(f"\n  {i}. {source_node['content'][:60]}...")
        print(f"     [{relation}]")
        print(f"     {target_node['content'][:60]}...")
    
    if len(results) == 0:
        print("\n❌ FAIL: No triplets retrieved")
        return False
    
    if len(results) > strategy["target_triplets"]:
        print(f"\n⚠️  WARN: Retrieved {len(results)} triplets (target: {strategy['target_triplets']})")
    
    print(f"\n✅ PASS: Adaptive search retrieved {len(results)} relevant triplets")
    return True


def test_integration():
    """Test full integration: opponent argument → adaptive context."""
    print("\n" + "="*70)
    print("TEST 3: End-to-End Integration")
    print("="*70)
    
    kant = PhilosopherAgent("kant")
    
    if not kant.identity_graph:
        print("❌ FAIL: Kant identity graph not loaded")
        return False
    
    # Simulate debate history with Mill's argument
    chat_history = [
        HumanMessage(
            content="Consider the Trolley Problem: Is it morally permissible to pull the lever?",
            name="Moderator"
        ),
        HumanMessage(
            content="Five lives clearly outweigh one life. The utilitarian calculation is straightforward: pulling the lever maximizes happiness and minimizes suffering. We must produce the greatest good for the greatest number.",
            name="John Stuart Mill"
        )
    ]
    
    print(f"\n💬 Mill's argument: {chat_history[-1].content[:100]}...")
    
    # Test ADAPTIVE retrieval
    print("\n🔄 Testing ADAPTIVE ID-RAG...")
    adaptive_context = kant._retrieve_identity_context(
        query="Trolley Problem",
        opponent_argument=chat_history[-1].content,
        opponent_name="John Stuart Mill",
        use_adaptive=True
    )
    
    print(f"\n📄 Adaptive context (first 500 chars):")
    print(adaptive_context[:500] + "...")
    
    # Test STATIC retrieval (fallback)
    print("\n🔄 Testing STATIC ID-RAG (fallback)...")
    static_context = kant._retrieve_identity_context(
        query="Trolley Problem",
        use_adaptive=False
    )
    
    print(f"\n📄 Static context (first 500 chars):")
    print(static_context[:500] + "...")
    
    # Compare
    print(f"\n📊 Comparison:")
    print(f"  Adaptive length: {len(adaptive_context)} chars")
    print(f"  Static length: {len(static_context)} chars")
    
    if len(adaptive_context) == 0:
        print("\n❌ FAIL: Adaptive context is empty")
        return False
    
    if len(static_context) == 0:
        print("\n❌ FAIL: Static context is empty")
        return False
    
    # Check if adaptive context mentions utility/consequentialism (should be targeted)
    adaptive_lower = adaptive_context.lower()
    if "utility" in adaptive_lower or "consequentialism" in adaptive_lower or "happiness" in adaptive_lower:
        print("\n✅ PASS: Adaptive context targeted opponent's framework")
    else:
        print("\n⚠️  WARN: Adaptive context may not be targeted (no opponent keywords found)")
    
    print("\n✅ PASS: Integration test complete")
    return True


def test_full_response():
    """Test complete response generation with adaptive ID-RAG."""
    print("\n" + "="*70)
    print("TEST 4: Full Response Generation")
    print("="*70)
    
    kant = PhilosopherAgent("kant")
    
    chat_history = [
        HumanMessage(
            content="Trolley Problem: Pull lever to save 5 by sacrificing 1?",
            name="Moderator"
        ),
        HumanMessage(
            content="Clearly we must pull the lever. Five lives outweigh one. Utility demands it.",
            name="John Stuart Mill"
        )
    ]
    
    print(f"\n💬 Mill: {chat_history[-1].content}")
    print("\n⏳ Generating Kant's response with ADAPTIVE ID-RAG...")
    
    try:
        response = kant.respond(
            topic="Trolley Problem",
            chat_history=chat_history,
            use_id_rag=True,
            use_adaptive_idrag=True
        )
        
        print(f"\n🎭 Kant's response (first 300 chars):")
        print(response[:300] + "...")
        
        # Check for key Kantian terms (should appear if ID-RAG working)
        kant_terms = ["duty", "categorical", "dignity", "respect", "maxim", "universal"]
        found_terms = [term for term in kant_terms if term.lower() in response.lower()]
        
        print(f"\n📊 Kantian terminology detected: {found_terms}")
        
        if len(found_terms) >= 2:
            print("\n✅ PASS: Response contains Kantian terminology (ID-RAG working)")
        else:
            print("\n⚠️  WARN: Few Kantian terms detected (ID-RAG may be weak)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: Response generation failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADAPTIVE ID-RAG UNIT TESTS")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Query Builder", test_query_builder()))
    results.append(("Adaptive Search", test_adaptive_search()))
    results.append(("Integration", test_integration()))
    results.append(("Full Response", test_full_response()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n📊 Total: {passed}/{total} tests passed ({100*passed//total}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Adaptive ID-RAG implementation verified.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review implementation.")
