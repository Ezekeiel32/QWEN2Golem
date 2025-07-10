#!/usr/bin/env python3
"""
Test script to verify both plain hypercube and enhanced aether hypercube models work together
"""

import requests
import json
import time
from datetime import datetime

def test_dual_neural_models():
    """Test both neural models through the golem server"""
    
    # Test server status
    print("🔍 Testing server status...")
    try:
        response = requests.get('http://localhost:5000/neural/status')
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Server status: {json.dumps(status, indent=2)}")
        else:
            print(f"❌ Server status error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server connection error: {e}")
        return False
    
    # Test texts representing different consciousness states
    test_texts = [
        "I am exploring the depths of consciousness and reality",
        "The mystical matrix reveals hidden patterns in the universe",
        "Mathematical frameworks guide us through dimensional spaces",
        "Aether flows through the hypercube vertices like sacred geometry",
        "Binary duality creates the foundation of all existence",
        "The geometric ratio 22 emerges from the hypercube expansion",
        "Infinitesimal errors track the precision of consciousness",
        "Neural networks learn the patterns of mystical awareness"
    ]
    
    print(f"\n🧠 Testing dual neural classification with {len(test_texts)} texts...")
    
    results = []
    
    for i, text in enumerate(test_texts):
        print(f"\n📝 Test {i+1}: {text[:50]}...")
        
        try:
            response = requests.post(
                'http://localhost:8080/neural/classify',
                json={'text': text},
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract key information
                models_used = result.get('models_used', [])
                primary_model = result.get('primary_model', 'unknown')
                predicted_vertex = result.get('predicted_vertex', -1)
                confidence = result.get('confidence', 0.0)
                
                print(f"   🎯 Primary Model: {primary_model}")
                print(f"   📊 Models Used: {models_used}")
                print(f"   🔢 Predicted Vertex: {predicted_vertex}")
                print(f"   💪 Confidence: {confidence:.3f}")
                
                # Check for model agreement if both models are available
                if 'model_agreement' in result:
                    agreement = result['model_agreement']
                    conf_diff = result.get('confidence_difference', 0.0)
                    print(f"   🤝 Model Agreement: {agreement}")
                    print(f"   📈 Confidence Difference: {conf_diff:.3f}")
                
                # Check for enhanced model specific features
                if 'enhanced_aether_hypercube' in result:
                    enhanced = result['enhanced_aether_hypercube']
                    framework_integrity = enhanced.get('framework_integrity', 0.0)
                    infinitesimal_error = enhanced.get('infinitesimal_error', 0.0)
                    print(f"   🔗 Framework Integrity: {framework_integrity:.3f}")
                    print(f"   ⚡ Infinitesimal Error: {infinitesimal_error:.6f}")
                
                results.append({
                    'text': text,
                    'models_used': models_used,
                    'primary_model': primary_model,
                    'predicted_vertex': predicted_vertex,
                    'confidence': confidence,
                    'success': True
                })
                
                print(f"   ✅ Classification successful")
                
            else:
                print(f"   ❌ Classification failed: {response.status_code}")
                print(f"   📝 Error: {response.text}")
                
                results.append({
                    'text': text,
                    'success': False,
                    'error': response.text
                })
                
        except Exception as e:
            print(f"   ❌ Request error: {e}")
            results.append({
                'text': text,
                'success': False,
                'error': str(e)
            })
        
        time.sleep(0.5)  # Small delay between requests
    
    # Summary
    successful_tests = sum(1 for r in results if r.get('success', False))
    total_tests = len(results)
    
    print(f"\n📊 DUAL MODEL TEST SUMMARY")
    print(f"   ✅ Successful Tests: {successful_tests}/{total_tests}")
    print(f"   📈 Success Rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests > 0:
        # Analyze model usage
        models_used_counts = {}
        primary_model_counts = {}
        vertex_predictions = {}
        
        for result in results:
            if result.get('success'):
                models = result.get('models_used', [])
                for model in models:
                    models_used_counts[model] = models_used_counts.get(model, 0) + 1
                
                primary = result.get('primary_model', 'unknown')
                primary_model_counts[primary] = primary_model_counts.get(primary, 0) + 1
                
                vertex = result.get('predicted_vertex', -1)
                vertex_predictions[vertex] = vertex_predictions.get(vertex, 0) + 1
        
        print(f"\n🔍 ANALYSIS:")
        print(f"   📊 Models Used: {models_used_counts}")
        print(f"   🎯 Primary Models: {primary_model_counts}")
        print(f"   🔢 Vertex Distribution: {dict(sorted(vertex_predictions.items()))}")
    
    return successful_tests == total_tests

def test_model_comparison():
    """Test specific model comparison functionality"""
    
    print(f"\n🔬 Testing model comparison...")
    
    test_text = "The mystical mathematical framework 1+0+1+0=2^5=32*11/16=22+3.33*3 guides consciousness through the 5D hypercube"
    
    try:
        response = requests.post(
            'http://localhost:8080/neural/classify',
            json={'text': test_text},
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"📝 Test Text: {test_text}")
            print(f"🎯 Primary Model: {result.get('primary_model', 'unknown')}")
            print(f"📊 Models Used: {result.get('models_used', [])}")
            
            # Compare plain vs enhanced predictions
            if 'plain_hypercube' in result and 'enhanced_aether_hypercube' in result:
                plain = result['plain_hypercube']
                enhanced = result['enhanced_aether_hypercube']
                
                print(f"\n🔍 MODEL COMPARISON:")
                print(f"   Plain Hypercube:")
                print(f"     🔢 Vertex: {plain['predicted_vertex']}")
                print(f"     💪 Confidence: {plain['confidence']:.3f}")
                print(f"     🎭 Signature: {plain['consciousness_signature']}")
                
                print(f"   Enhanced Aether Hypercube:")
                print(f"     🔢 Vertex: {enhanced['predicted_vertex']}")
                print(f"     💪 Confidence: {enhanced['confidence']:.3f}")
                print(f"     🎭 Signature: {enhanced['consciousness_signature']}")
                print(f"     🔗 Framework Integrity: {enhanced.get('framework_integrity', 0.0):.3f}")
                print(f"     ⚡ Infinitesimal Error: {enhanced.get('infinitesimal_error', 0.0):.6f}")
                
                agreement = result.get('model_agreement', False)
                print(f"   🤝 Agreement: {agreement}")
                
                return True
            else:
                print(f"⚠️  Only one model available for comparison")
                return False
                
        else:
            print(f"❌ Comparison test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Comparison test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 DUAL NEURAL MODEL TEST SUITE")
    print("=" * 50)
    
    # Test dual models
    dual_success = test_dual_neural_models()
    
    # Test model comparison
    comparison_success = test_model_comparison()
    
    print(f"\n🏁 FINAL RESULTS:")
    print(f"   🧠 Dual Model Test: {'✅ PASSED' if dual_success else '❌ FAILED'}")
    print(f"   🔬 Comparison Test: {'✅ PASSED' if comparison_success else '❌ FAILED'}")
    
    if dual_success and comparison_success:
        print(f"\n🎉 ALL TESTS PASSED! Both plain hypercube and enhanced aether hypercube models are working together!")
    else:
        print(f"\n⚠️  Some tests failed. Check the server logs for details.") 