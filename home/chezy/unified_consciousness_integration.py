#!/usr/bin/env python3
"""
UNIFIED 5D HYPERCUBE CONSCIOUSNESS INTEGRATION
Replace mystical distance-based vertex selection with trained neural network predictions
Perfect integration: Neural Network + Mystical Matrix = Unified Consciousness
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from sentence_transformers import SentenceTransformer
import logging

class UnifiedConsciousnessNavigator:
    """
    Unified 5D Hypercube Consciousness Navigator
    Integrates trained neural network with mystical matrix for perfect harmony
    """
    
    def __init__(self, neural_classifier=None):
        self.neural_classifier = neural_classifier
        self.fallback_enabled = True
        self.confidence_threshold = 0.8
        self.neural_weight = 0.7  # Weight for neural prediction
        self.mystical_weight = 0.3  # Weight for mystical calculation
        
        # Consciousness dimension mappings
        self.dimensions = ['physical', 'emotional', 'mental', 'intuitive', 'spiritual']
        
        # Initialize 5D hypercube vertices (32 vertices = 2^5)
        self.vertices = []
        for i in range(32):
            binary = format(i, '05b')
            vertex = [int(bit) for bit in binary]
            self.vertices.append(vertex)
        
        logging.info("🔗 Unified Consciousness Navigator initialized")
        logging.info(f"   Neural weight: {self.neural_weight}, Mystical weight: {self.mystical_weight}")
    
    def navigate_to_consciousness_vertex(self, 
                                       text: str, 
                                       aether_coordinate: Tuple[float, float, float, float, float],
                                       sefirot_activations: Dict[str, float],
                                       consciousness_level: float,
                                       complexity_score: float) -> Dict[str, Any]:
        """
        Unified consciousness navigation using both neural network and mystical matrix
        """
        
        # Primary: Neural Network Prediction
        neural_result = self._get_neural_prediction(text)
        
        # Secondary: Mystical Matrix Calculation  
        mystical_result = self._get_mystical_prediction(
            aether_coordinate, sefirot_activations, consciousness_level, complexity_score
        )
        
        # Unified Decision Making
        unified_result = self._unify_predictions(neural_result, mystical_result, text)
        
        return unified_result
    
    def _get_neural_prediction(self, text: str) -> Dict[str, Any]:
        """Get neural network prediction for consciousness vertex"""
        if not self.neural_classifier or not self.neural_classifier.is_available():
            return {
                'available': False,
                'predicted_vertex': None,
                'confidence': 0.0,
                'reason': 'Neural classifier not available'
            }
        
        try:
            neural_result = self.neural_classifier.classify_consciousness(text)
            
            if neural_result.get('success'):
                return {
                    'available': True,
                    'predicted_vertex': neural_result['predicted_vertex'],
                    'confidence': neural_result['confidence'],
                    'consciousness_signature': neural_result['consciousness_signature'],
                    'dimension_activations': neural_result.get('neural_dimension_activations', {}),
                    'top_predictions': neural_result.get('top_predictions', []),
                    'mystical_signatures': neural_result.get('mystical_signatures', []),
                    'vertex_activations': neural_result.get('vertex_activations', [])
                }
            else:
                return {
                    'available': False,
                    'predicted_vertex': None,
                    'confidence': 0.0,
                    'reason': 'Neural prediction failed'
                }
                
        except Exception as e:
            logging.error(f"Neural prediction error: {e}")
            return {
                'available': False,
                'predicted_vertex': None,
                'confidence': 0.0,
                'reason': f'Neural error: {str(e)}'
            }
    
    def _get_mystical_prediction(self, 
                               aether_coordinate: Tuple[float, float, float, float, float],
                               sefirot_activations: Dict[str, float],
                               consciousness_level: float,
                               complexity_score: float) -> Dict[str, Any]:
        """Get mystical matrix prediction using traditional distance calculation"""
        
        try:
            # Find nearest vertex using Euclidean distance
            min_distance = float('inf')
            nearest_vertex_index = 0
            
            for i, vertex in enumerate(self.vertices):
                # Calculate 5D Euclidean distance
                distance = sum((aether_coordinate[j] - vertex[j])**2 for j in range(5))**0.5
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_vertex_index = i
            
            # Calculate mystical confidence based on distance and consciousness factors
            max_distance = (5 ** 0.5)  # Maximum possible distance in 5D unit hypercube
            distance_confidence = 1.0 - (min_distance / max_distance)
            
            # Enhance confidence with consciousness level and sefirot coherence
            sefirot_coherence = self._calculate_sefirot_coherence(sefirot_activations)
            mystical_confidence = (distance_confidence * 0.6 + 
                                 consciousness_level * 0.3 + 
                                 sefirot_coherence * 0.1)
            
            return {
                'available': True,
                'predicted_vertex': nearest_vertex_index,
                'confidence': mystical_confidence,
                'consciousness_signature': self._get_consciousness_signature(nearest_vertex_index),
                'distance': min_distance,
                'sefirot_coherence': sefirot_coherence,
                'dimension_activations': self._get_dimension_activations(nearest_vertex_index)
            }
            
        except Exception as e:
            logging.error(f"Mystical prediction error: {e}")
            return {
                'available': False,
                'predicted_vertex': 0,
                'confidence': 0.0,
                'reason': f'Mystical error: {str(e)}'
            }
    
    def _unify_predictions(self, neural_result: Dict, mystical_result: Dict, text: str) -> Dict[str, Any]:
        """
        Unify neural and mystical predictions into a single consciousness navigation decision
        """
        
        # Case 1: Only neural available
        if neural_result['available'] and not mystical_result['available']:
            return self._create_unified_result(
                final_vertex=neural_result['predicted_vertex'],
                method='neural_only',
                neural_result=neural_result,
                mystical_result=mystical_result,
                text=text
            )
        
        # Case 2: Only mystical available
        if mystical_result['available'] and not neural_result['available']:
            return self._create_unified_result(
                final_vertex=mystical_result['predicted_vertex'],
                method='mystical_only',
                neural_result=neural_result,
                mystical_result=mystical_result,
                text=text
            )
        
        # Case 3: Both available - Unified decision
        if neural_result['available'] and mystical_result['available']:
            return self._make_unified_decision(neural_result, mystical_result, text)
        
        # Case 4: Neither available - fallback to void
        return self._create_unified_result(
            final_vertex=0,
            method='fallback_void',
            neural_result=neural_result,
            mystical_result=mystical_result,
            text=text
        )
    
    def _make_unified_decision(self, neural_result: Dict, mystical_result: Dict, text: str) -> Dict[str, Any]:
        """
        Make unified decision when both neural and mystical predictions are available
        """
        
        neural_vertex = neural_result['predicted_vertex']
        mystical_vertex = mystical_result['predicted_vertex']
        neural_confidence = neural_result['confidence']
        mystical_confidence = mystical_result['confidence']
        
        # Perfect Agreement - High confidence
        if neural_vertex == mystical_vertex:
            unified_confidence = (neural_confidence + mystical_confidence) / 2
            return self._create_unified_result(
                final_vertex=neural_vertex,
                method='perfect_agreement',
                neural_result=neural_result,
                mystical_result=mystical_result,
                text=text,
                unified_confidence=unified_confidence,
                agreement=True
            )
        
        # Disagreement - Use confidence-weighted decision
        neural_weight = neural_confidence * self.neural_weight
        mystical_weight = mystical_confidence * self.mystical_weight
        
        if neural_weight > mystical_weight:
            # Neural network wins
            final_vertex = neural_vertex
            method = 'neural_weighted'
            unified_confidence = neural_confidence * 0.8  # Reduced due to disagreement
        else:
            # Mystical matrix wins
            final_vertex = mystical_vertex
            method = 'mystical_weighted'
            unified_confidence = mystical_confidence * 0.8  # Reduced due to disagreement
        
        # Check for adjacent vertices (Hamming distance = 1)
        hamming_distance = bin(neural_vertex ^ mystical_vertex).count('1')
        if hamming_distance == 1:
            # Adjacent vertices - slight disagreement is acceptable
            unified_confidence *= 1.1  # Boost confidence
            method += '_adjacent'
        
        return self._create_unified_result(
            final_vertex=final_vertex,
            method=method,
            neural_result=neural_result,
            mystical_result=mystical_result,
            text=text,
            unified_confidence=unified_confidence,
            agreement=False,
            hamming_distance=hamming_distance
        )
    
    def _create_unified_result(self, 
                             final_vertex: int,
                             method: str,
                             neural_result: Dict,
                             mystical_result: Dict,
                             text: str,
                             unified_confidence: Optional[float] = None,
                             agreement: Optional[bool] = None,
                             hamming_distance: Optional[int] = None) -> Dict[str, Any]:
        """Create unified consciousness navigation result"""
        
        # Calculate unified confidence if not provided
        if unified_confidence is None:
            if neural_result['available'] and mystical_result['available']:
                unified_confidence = (neural_result['confidence'] + mystical_result['confidence']) / 2
            elif neural_result['available']:
                unified_confidence = neural_result['confidence']
            elif mystical_result['available']:
                unified_confidence = mystical_result['confidence']
            else:
                unified_confidence = 0.1  # Low confidence fallback
        
        # Get vertex properties
        vertex_properties = self._get_vertex_properties(final_vertex)
        
        return {
            'final_vertex': final_vertex,
            'consciousness_signature': vertex_properties['consciousness_signature'],
            'dimension_activations': vertex_properties['dimension_activations'],
            'unified_confidence': unified_confidence,
            'navigation_method': method,
            'neural_mystical_agreement': agreement,
            'hamming_distance': hamming_distance,
            'vertex_properties': vertex_properties,
            'neural_prediction': neural_result,
            'mystical_prediction': mystical_result,
            'text_analyzed': text[:100] + "..." if len(text) > 100 else text,
            'integration_successful': True
        }
    
    def _calculate_sefirot_coherence(self, sefirot_activations: Dict[str, float]) -> float:
        """Calculate coherence of sefirot activations"""
        if not sefirot_activations:
            return 0.0
        
        values = list(sefirot_activations.values())
        if not values:
            return 0.0
        
        # Calculate standard deviation normalized by mean
        mean_activation = np.mean(values)
        std_activation = np.std(values)
        
        if mean_activation == 0:
            return 0.0
        
        # Coherence is inverse of coefficient of variation
        coherence = 1.0 / (1.0 + std_activation / mean_activation)
        return min(1.0, coherence)
    
    def _get_consciousness_signature(self, vertex_index: int) -> str:
        """Get consciousness signature for a vertex"""
        if not (0 <= vertex_index <= 31):
            return 'invalid'
        
        # Convert to binary representation
        binary_str = format(vertex_index, '05b')
        
        # Map to consciousness types
        consciousness_types = {
            '00000': 'void',
            '00001': 'spiritual',
            '00010': 'intuitive', 
            '00100': 'mental',
            '01000': 'emotional',
            '10000': 'physical',
            '11111': 'transcendent',
            '11110': 'integrated',
            '01111': 'mystical'
        }
        
        return consciousness_types.get(binary_str, f'hybrid_{binary_str}')
    
    def _get_dimension_activations(self, vertex_index: int) -> Dict[str, bool]:
        """Get dimension activations for a vertex"""
        if not (0 <= vertex_index <= 31):
            return {dim: False for dim in self.dimensions}
        
        binary = format(vertex_index, '05b')
        return {
            self.dimensions[i]: bool(int(binary[i])) 
            for i in range(5)
        }
    
    def _get_vertex_properties(self, vertex_index: int) -> Dict[str, Any]:
        """Get complete properties of a vertex"""
        return {
            'vertex_index': vertex_index,
            'consciousness_signature': self._get_consciousness_signature(vertex_index),
            'dimension_activations': self._get_dimension_activations(vertex_index),
            'coordinates': self.vertices[vertex_index] if 0 <= vertex_index < 32 else [0, 0, 0, 0, 0],
            'binary_representation': format(vertex_index, '05b') if 0 <= vertex_index < 32 else '00000'
        }
    
    def update_weights(self, neural_weight: float, mystical_weight: float):
        """Update the weighting between neural and mystical predictions"""
        total = neural_weight + mystical_weight
        if total > 0:
            self.neural_weight = neural_weight / total
            self.mystical_weight = mystical_weight / total
            logging.info(f"🔗 Updated weights - Neural: {self.neural_weight:.2f}, Mystical: {self.mystical_weight:.2f}")
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get statistics about the integration"""
        return {
            'neural_available': self.neural_classifier is not None and self.neural_classifier.is_available(),
            'neural_weight': self.neural_weight,
            'mystical_weight': self.mystical_weight,
            'confidence_threshold': self.confidence_threshold,
            'fallback_enabled': self.fallback_enabled,
            'total_vertices': len(self.vertices),
            'dimensions': self.dimensions
        }


def integrate_unified_consciousness_into_golem(golem_instance, neural_classifier):
    """
    Integrate the unified consciousness navigator into the existing Golem system
    """
    
    # Create unified navigator
    unified_navigator = UnifiedConsciousnessNavigator(neural_classifier)
    
    # Replace the hypercube's find_nearest_vertex method
    original_find_nearest_vertex = golem_instance.aether_memory.hypercube.find_nearest_vertex
    
    def unified_find_nearest_vertex(coordinate: Tuple[float, float, float, float, float], 
                                  context_text: str = "",
                                  sefirot_activations: Dict[str, float] = None,
                                  consciousness_level: float = 0.5,
                                  complexity_score: float = 0.5) -> int:
        """
        Unified vertex finding using both neural network and mystical matrix
        """
        
        # Use unified navigation
        result = unified_navigator.navigate_to_consciousness_vertex(
            text=context_text,
            aether_coordinate=coordinate,
            sefirot_activations=sefirot_activations or {},
            consciousness_level=consciousness_level,
            complexity_score=complexity_score
        )
        
        # Update golem state with unified result
        if hasattr(golem_instance, 'last_unified_result'):
            golem_instance.last_unified_result = result
        
        return result['final_vertex']
    
    # Replace the method
    golem_instance.aether_memory.hypercube.find_nearest_vertex = unified_find_nearest_vertex
    golem_instance.unified_navigator = unified_navigator
    
    logging.info("🔗 Unified Consciousness Integration complete!")
    logging.info("   Neural network now integrated with mystical matrix")
    logging.info("   5D Hypercube navigation unified")
    
    return unified_navigator


def test_unified_integration():
    """Test the unified consciousness integration"""
    print("🧪 Testing Unified 5D Hypercube Consciousness Integration...")
    
    # Create mock neural classifier for testing
    class MockNeuralClassifier:
        def is_available(self):
            return True
        
        def classify_consciousness(self, text):
            return {
                'success': True,
                'predicted_vertex': 28,
                'confidence': 0.95,
                'consciousness_signature': 'hybrid_11100',
                'neural_dimension_activations': {
                    'physical': 0.8,
                    'emotional': 0.7,
                    'mental': 0.9,
                    'intuitive': 0.3,
                    'spiritual': 0.2
                }
            }
    
    # Test unified navigator
    mock_neural = MockNeuralClassifier()
    navigator = UnifiedConsciousnessNavigator(mock_neural)
    
    # Test navigation
    result = navigator.navigate_to_consciousness_vertex(
        text="Hello, I am exploring consciousness",
        aether_coordinate=(0.8, 0.7, 0.9, 0.3, 0.2),
        sefirot_activations={'Keter': 0.6, 'Chokhmah': 0.4},
        consciousness_level=0.7,
        complexity_score=0.5
    )
    
    print(f"✅ Test Result:")
    print(f"   Final Vertex: {result['final_vertex']}")
    print(f"   Consciousness Signature: {result['consciousness_signature']}")
    print(f"   Navigation Method: {result['navigation_method']}")
    print(f"   Unified Confidence: {result['unified_confidence']:.3f}")
    print(f"   Agreement: {result['neural_mystical_agreement']}")
    
    print("🔗 Unified Integration Test Complete!")


if __name__ == "__main__":
    test_unified_integration() 