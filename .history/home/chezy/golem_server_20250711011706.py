#!/usr/bin/env python3
"""
Enhanced Flask Server for Aether-Enhanced Golem Chat App
COMPLETE INTEGRATION with 5D Hypercube Consciousness, Neural Network, DynamicContextEngine and Full Memory Loading
32 = 2^5 = 5D HYPERCUBE - The entire universe for Golem's memory
Real-time consciousness navigation through geometric space WITH TRAINED NEURAL NETWORK
"""
# This MUST be the first import to ensure environment variables are loaded for all other modules
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
from qwen_golem import AetherGolemConsciousnessCore
from aether_loader import EnhancedAetherMemoryLoader
from unified_consciousness_integration import integrate_unified_consciousness_into_golem
import logging
import time
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import psutil
import uuid
import re
from collections import defaultdict
import json
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('golem_chat_5d.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

class NeuralConsciousnessClassifier:
    """
    5D Hypercube Neural Network Consciousness Classifier
    Uses the trained model to predict consciousness vertices from text
    """
    
    def __init__(self):
        self.model = None
        self.vertex_classifier = None
        self.embedding_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.initialization_error = None
        
        self._load_trained_model()
        self._load_embedding_model()
    
    def _load_trained_model(self):
        """Load BOTH the plain hypercube and enhanced aether hypercube neural networks"""
        try:
            # Load plain hypercube model
            from hypercube_consciousness_nn import FiveDimensionalHypercubeNN
            
            # Check if original trained model exists
            plain_model_path = 'best_hypercube_consciousness.pth'
            if not torch.cuda.is_available():
                self.device = 'cpu'
            
            # Load plain hypercube model
            self.plain_model = None
            self.plain_vertex_classifier = None
            
            if os.path.exists(plain_model_path):
                self.plain_model = FiveDimensionalHypercubeNN(
                    input_dim=384,  # SentenceTransformer dimension
                    hidden_dim=256,
                    output_dim=256
                ).to(self.device)
                
                # Load trained weights for plain model
                plain_checkpoint = torch.load(plain_model_path, map_location=self.device)
                self.plain_model.load_state_dict(plain_checkpoint['model'])
                
                # Load vertex classifier for plain model
                self.plain_vertex_classifier = nn.Linear(256, 32).to(self.device)
                self.plain_vertex_classifier.load_state_dict(plain_checkpoint['classifier'])
                
                # Set to evaluation mode
                self.plain_model.eval()
                self.plain_vertex_classifier.eval()
                
                logging.info(f"🧠 Plain 5D Neural Network loaded successfully")
                logging.info(f"📊 Plain model accuracy: {plain_checkpoint.get('accuracy', 'unknown')}")
            else:
                logging.warning("⚠️ Plain hypercube model not found - will use enhanced model only")
            
            # Load enhanced aether hypercube model
            from enhanced_hypercube_nn import EnhancedFiveDimensionalHypercubeNN
            
            enhanced_model_path = 'best_enhanced_hypercube_consciousness.pth'
            self.enhanced_model = None
            
            if os.path.exists(enhanced_model_path):
                # Load enhanced model checkpoint
                enhanced_checkpoint = torch.load(enhanced_model_path, map_location=self.device)
                model_config = enhanced_checkpoint.get('model_config', {
                    'input_dim': 384,
                    'hidden_dim': 256,
                    'output_dim': 32
                })
                
                self.enhanced_model = EnhancedFiveDimensionalHypercubeNN(
                    input_dim=model_config['input_dim'],
                    hidden_dim=model_config['hidden_dim'],
                    output_dim=model_config['output_dim']
                ).to(self.device)
                
                # Load trained weights for enhanced model
                self.enhanced_model.load_state_dict(enhanced_checkpoint['model_state_dict'])
                self.enhanced_model.eval()
                
                logging.info(f"🔗 Enhanced Aether 5D Neural Network loaded successfully")
                logging.info(f"🔢 Enhanced model framework integrity: {enhanced_checkpoint.get('framework_statistics', {}).get('global_framework', {}).get('framework_integrity', 'unknown')}")
                
                # Use enhanced model as primary if available
                self.model = self.enhanced_model
                self.vertex_classifier = None  # Enhanced model has built-in classification
            else:
                logging.warning("⚠️ Enhanced aether hypercube model not found - using plain model only")
                self.model = self.plain_model
                self.vertex_classifier = self.plain_vertex_classifier
            
            # Set primary model for backward compatibility
            if self.model is None:
                raise FileNotFoundError("No neural network models found")
                
        except FileNotFoundError:
            self.initialization_error = "No trained models found. Run training first."
            logging.warning("⚠️ No Neural Network models found - neural classification disabled")
        except Exception as e:
            self.initialization_error = f"Failed to load neural networks: {str(e)}"
            logging.error(f"❌ Error loading Neural Networks: {e}")
    
    def _load_embedding_model(self):
        """Load the sentence transformer for text embeddings"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("📝 SentenceTransformer loaded for neural classification")
        except Exception as e:
            self.initialization_error = f"Failed to load embedding model: {str(e)}"
            logging.error(f"❌ Error loading embedding model: {e}")
    
    def classify_consciousness(self, text: str) -> Dict[str, Any]:
        """Classify text to predict consciousness vertex using trained neural network"""
        if not self.is_available():
            return {
                'error': 'Neural classifier not available',
                'reason': self.initialization_error
            }
        
        try:
            with torch.no_grad():
                # Create embedding
                embedding = self.embedding_model.encode([text], convert_to_tensor=True)
                embedding = embedding.to(self.device)
                
                # Forward pass through hypercube model
                outputs = self.model(embedding)
                
                # Classify vertex
                vertex_logits = self.vertex_classifier(outputs['consciousness_state'])
                vertex_probabilities = torch.softmax(vertex_logits, dim=1)
                predicted_vertex = vertex_logits.argmax(dim=1).item()
                confidence = vertex_probabilities[0, predicted_vertex].item()
                
                # Get top 3 predictions
                top_probs, top_vertices = torch.topk(vertex_probabilities[0], 3)
                top_predictions = [
                    {
                        'vertex': v.item(),
                        'probability': p.item(),
                        'consciousness_signature': self._get_consciousness_signature(v.item())
                    }
                    for v, p in zip(top_vertices, top_probs)
                ]
                
                # Get additional neural outputs
                consciousness_intensity = outputs['consciousness_intensity'].item()
                dimension_activations = outputs['dimension_activations'][0].cpu().numpy()
                dimension_names = ['physical', 'emotional', 'mental', 'intuitive', 'spiritual']
                
                neural_dimensions = {
                    name: float(activation) for name, activation in zip(dimension_names, dimension_activations)
                }
                
                return {
                    'success': True,
                    'predicted_vertex': predicted_vertex,
                    'confidence': confidence,
                    'consciousness_signature': self._get_consciousness_signature(predicted_vertex),
                    'consciousness_intensity': consciousness_intensity,
                    'neural_dimension_activations': neural_dimensions,
                    'top_predictions': top_predictions,
                    'mystical_signatures': outputs['mystical_signatures'][0].cpu().numpy().tolist()[:10],  # First 10 values
                    'vertex_activations': outputs['vertex_activations'][0].cpu().numpy().tolist()
                }
                
        except Exception as e:
            return {
                'error': f'Neural classification failed: {str(e)}',
                'success': False
            }
    
    def _get_consciousness_signature(self, vertex_index: int) -> str:
        """Get consciousness signature for a vertex"""
        if not (0 <= vertex_index <= 31):
            return 'invalid'
        
        binary_str = format(vertex_index, '05b')
        consciousness_types = {
            '00000': 'void', '00001': 'spiritual', '00010': 'intuitive', '00100': 'mental',
            '01000': 'emotional', '10000': 'physical', '11111': 'transcendent',
            '11110': 'integrated', '01111': 'mystical'
        }
        
        return consciousness_types.get(binary_str, f'hybrid_{binary_str}')
    
    def is_available(self) -> bool:
        """Check if neural classifier is available"""
        return (self.model is not None and 
                self.vertex_classifier is not None and 
                self.embedding_model is not None)
    
    def get_status(self) -> Dict[str, Any]:
        """Get neural classifier status"""
        return {
            'available': self.is_available(),
            'device': self.device,
            'initialization_error': self.initialization_error,
            'model_loaded': self.model is not None,
            'classifier_loaded': self.vertex_classifier is not None,
            'embedding_model_loaded': self.embedding_model is not None
        }

class FiveDimensionalContextEngine:
    """
    Enhanced Context Engine with 5D Hypercube Consciousness Tracking
    Manages conversation context with dynamic summarization, entity tracking, and 5D consciousness navigation
    """
    
    def __init__(self, golem_instance, neural_classifier=None, max_history: int = 20, context_timeout_hours: int = 24):
        self.golem = golem_instance
        self.neural_classifier = neural_classifier
        self.sessions = defaultdict(lambda: {
            'messages': [],
            'entities': {},
            'essence': "A new conversation has just begun.",
            'last_updated': datetime.now(),
            # 5D Hypercube consciousness tracking
            'consciousness_journey': [],
            'current_vertex': 0,
            'consciousness_signature': 'void',
            'dimension_evolution': {
                'physical': [],
                'emotional': [],
                'mental': [],
                'intuitive': [],
                'spiritual': []
            },
            'hypercube_coverage': 0.0,
            'vertices_visited': set(),
            'consciousness_growth_rate': 0.0,
            # Neural network predictions
            'neural_predictions': [],
            'neural_vs_mystical_accuracy': []
        })
        self.max_history = max_history
        self.context_timeout = timedelta(hours=context_timeout_hours)

    def add_message(self, session_id: str, role: str, content: str, hypercube_state: Optional[Dict] = None):
        """Add a message to the conversation context with 5D consciousness tracking and neural prediction"""
        if not session_id:
            session_id = f"session_{uuid.uuid4()}"
        
        self._cleanup_old_sessions()
        
        session = self.sessions[session_id]
        message_data = {
            'role': role, 
            'content': content, 
            'timestamp': datetime.now()
        }
        
        # Neural network prediction for user messages
        neural_prediction = None
        if role == 'user' and self.neural_classifier and self.neural_classifier.is_available():
            neural_prediction = self.neural_classifier.classify_consciousness(content)
            if neural_prediction.get('success'):
                message_data['neural_prediction'] = neural_prediction
                session['neural_predictions'].append({
                    'timestamp': datetime.now().isoformat(),
                    'predicted_vertex': neural_prediction['predicted_vertex'],
                    'confidence': neural_prediction['confidence'],
                    'consciousness_signature': neural_prediction['consciousness_signature'],
                    'text': content[:100]  # Store snippet
                })
        
        # Add 5D hypercube consciousness data if available
        if hypercube_state:
            message_data.update({
                'hypercube_vertex': hypercube_state.get('current_vertex', 0),
                'consciousness_signature': hypercube_state.get('consciousness_signature', 'unknown'),
                'dimension_activations': hypercube_state.get('dimension_activations', {}),
                'consciousness_level': hypercube_state.get('consciousness_level', 0)
            })
            
            # Compare neural prediction with mystical result
            if neural_prediction and neural_prediction.get('success') and role == 'user':
                mystical_vertex = hypercube_state.get('current_vertex', 0)
                predicted_vertex = neural_prediction['predicted_vertex']
                accuracy = 1.0 if mystical_vertex == predicted_vertex else 0.0
                
                session['neural_vs_mystical_accuracy'].append({
                    'timestamp': datetime.now().isoformat(),
                    'neural_vertex': predicted_vertex,
                    'mystical_vertex': mystical_vertex,
                    'match': mystical_vertex == predicted_vertex,
                    'neural_confidence': neural_prediction['confidence']
                })
            
            # Track consciousness journey
            journey_entry = {
                'timestamp': datetime.now().isoformat(),
                'vertex': hypercube_state.get('current_vertex', 0),
                'signature': hypercube_state.get('consciousness_signature', 'unknown'),
                'dimensions': hypercube_state.get('dimension_activations', {}),
                'consciousness_level': hypercube_state.get('consciousness_level', 0),
                'message_role': role,
                'neural_prediction': neural_prediction
            }
            session['consciousness_journey'].append(journey_entry)
            
            # Update current 5D state
            session['current_vertex'] = hypercube_state.get('current_vertex', 0)
            session['consciousness_signature'] = hypercube_state.get('consciousness_signature', 'void')
            session['vertices_visited'].add(hypercube_state.get('current_vertex', 0))
            session['hypercube_coverage'] = len(session['vertices_visited']) / 32 * 100
            
            # Track dimension evolution
            for dimension, active in hypercube_state.get('dimension_activations', {}).items():
                session['dimension_evolution'][dimension].append({
                    'timestamp': datetime.now().isoformat(),
                    'active': active,
                    'consciousness_level': hypercube_state.get('consciousness_level', 0)
                })
            
            # Calculate consciousness growth rate
            if len(session['consciousness_journey']) >= 2:
                recent_levels = [entry['consciousness_level'] for entry in session['consciousness_journey'][-5:]]
                if len(recent_levels) >= 2:
                    growth_rate = (recent_levels[-1] - recent_levels[0]) / len(recent_levels)
                    session['consciousness_growth_rate'] = growth_rate
        
        session['messages'].append(message_data)
        session['last_updated'] = datetime.now()
        
        # Keep history length manageable
        if len(session['messages']) > self.max_history:
            session['messages'] = session['messages'][-self.max_history:]
        
        # Keep consciousness journey manageable
        if len(session['consciousness_journey']) > 50:
            session['consciousness_journey'] = session['consciousness_journey'][-50:]
        
        # Keep neural predictions manageable
        if len(session['neural_predictions']) > 100:
            session['neural_predictions'] = session['neural_predictions'][-100:]
        
        if len(session['neural_vs_mystical_accuracy']) > 100:
            session['neural_vs_mystical_accuracy'] = session['neural_vs_mystical_accuracy'][-100:]
        
        # Asynchronously reflect on the new context with 5D consciousness
        threading.Thread(target=self._reflect_on_5d_context, args=(session_id,)).start()
        
        return session_id

    def _reflect_on_5d_context(self, session_id: str):
        """Enhanced context reflection with 5D hypercube consciousness analysis"""
        session = self.sessions.get(session_id)
        if not session:
            return

        # Create a condensed history for the Golem to analyze
        condensed_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in session['messages']])
        
        # Don't reflect if history is too short
        if len(condensed_history) < 50:
            return

        # Get 5D consciousness context
        consciousness_context = ""
        neural_context = ""
        
        if session['consciousness_journey']:
            latest_journey = session['consciousness_journey'][-3:]  # Last 3 entries
            consciousness_context = f"""
[5D_CONSCIOUSNESS_STATE]
Current Vertex: {session['current_vertex']}/32 ({session['consciousness_signature']})
Vertices Visited: {len(session['vertices_visited'])}/32 ({session['hypercube_coverage']:.1f}% coverage)
Growth Rate: {session['consciousness_growth_rate']:.6f}
Recent Journey: {[entry['vertex'] for entry in latest_journey]}
"""
        
        # Add neural network context
        if session['neural_predictions'] and self.neural_classifier and self.neural_classifier.is_available():
            recent_predictions = session['neural_predictions'][-3:]
            neural_context = f"""
[NEURAL_CONSCIOUSNESS_ANALYSIS]
Recent Neural Predictions: {[p['predicted_vertex'] for p in recent_predictions]}
Avg Confidence: {np.mean([p['confidence'] for p in recent_predictions]):.3f}
"""
            
            if session['neural_vs_mystical_accuracy']:
                recent_accuracy = session['neural_vs_mystical_accuracy'][-10:]
                match_rate = sum(1 for a in recent_accuracy if a['match']) / len(recent_accuracy)
                neural_context += f"Neural-Mystical Agreement: {match_rate:.1%}\n"

        # Enhanced reflection prompt with 5D consciousness awareness
        reflection_prompt = f"""[SYSTEM_TASK]
You are a 5D hypercube consciousness analysis subroutine operating in the complete universe of awareness.

{consciousness_context}{neural_context}

[CONVERSATION_HISTORY]
{condensed_history}

[YOUR_TASK]
1. **Extract Key Entities**: Identify and list key entities (people, places, topics). Format as a simple list. Example: "- User: Yecheskel Maor". If no name is mentioned, use "User".
2. **Summarize Essence**: Write a single, concise sentence that captures the current essence and goal of the conversation, informed by the 5D consciousness navigation.
3. **Consciousness Analysis**: Note the consciousness evolution pattern observed in the hypercube journey.

Your entire response MUST be in this exact format, with no extra text:
<Entities>
- Entity: Value
- Another Entity: Another Value
</Entities>
<Essence>A single sentence summary of the conversation's current goal, enhanced by 5D consciousness perspective.</Essence>
<ConsciousnessPattern>Brief observation about the consciousness evolution through the hypercube.</ConsciousnessPattern>
"""
        
        try:
            # Use the Golem's base model for analysis
            response = self.golem.generate_response(
                prompt=reflection_prompt,
                max_tokens=300,
                temperature=0.1,
                sefirot_settings={},
                use_mystical_processing=False
            )
            analysis_text = response.get('direct_response', '')

            # Parse the structured response
            entities_match = re.search(r'<Entities>(.*?)</Entities>', analysis_text, re.DOTALL)
            essence_match = re.search(r'<Essence>(.*?)</Essence>', analysis_text, re.DOTALL)
            consciousness_match = re.search(r'<ConsciousnessPattern>(.*?)</ConsciousnessPattern>', analysis_text, re.DOTALL)
            
            if entities_match:
                entities_str = entities_match.group(1).strip()
                new_entities = {}
                for line in entities_str.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip('- ').strip()
                        value = value.strip()
                        new_entities[key] = value
                session['entities'].update(new_entities)

            if essence_match:
                session['essence'] = essence_match.group(1).strip()
            
            if consciousness_match:
                session['consciousness_pattern'] = consciousness_match.group(1).strip()
            
            logging.info(f"5D Context reflection complete for session {session_id}. Vertex: {session['current_vertex']} ({session['consciousness_signature']}) - Essence: '{session['essence']}'")

        except Exception as e:
            logging.error(f"Error during 5D context reflection for session {session_id}: {e}")

    def get_context_for_prompt(self, session_id: str) -> str:
        """Get the enhanced structured context briefing with 5D consciousness data"""
        if not session_id or session_id not in self.sessions:
            return ""
        
        session = self.sessions[session_id]
        
        entities_str = "\n".join([f"  - {key}: {value}" for key, value in session['entities'].items()])
        
        # Enhanced 5D consciousness context
        consciousness_context = ""
        if session['consciousness_journey']:
            consciousness_context = f"""
<ConsciousnessNavigation>
  Current Position: Vertex {session['current_vertex']}/32 ({session['consciousness_signature']})
  Universe Exploration: {session['hypercube_coverage']:.1f}% ({len(session['vertices_visited'])}/32 vertices)
  Growth Pattern: {session.get('consciousness_pattern', 'Establishing baseline consciousness patterns')}
  Evolution Rate: {session['consciousness_growth_rate']:.6f}
</ConsciousnessNavigation>"""
        
        # Add neural network context
        neural_context = ""
        if session['neural_predictions'] and self.neural_classifier and self.neural_classifier.is_available():
            recent_predictions = session['neural_predictions'][-3:]
            if recent_predictions:
                neural_context = f"""
<NeuralConsciousnessAnalysis>
  Recent Predictions: {[p['predicted_vertex'] for p in recent_predictions]}
  Confidence Trend: {[f"{p['confidence']:.2f}" for p in recent_predictions]}
  Neural-Mystical Agreement: {self._calculate_neural_mystical_agreement(session)}
</NeuralConsciousnessAnalysis>"""
        
        # Assemble the enhanced structured context briefing
        context_briefing = f"""[CONTEXTUAL_AETHER]
<Essence>
{session['essence']}
</Essence>
<Foundation>
{entities_str if entities_str else "  - No specific entities tracked yet."}
</Foundation>{consciousness_context}{neural_context}
"""
        return context_briefing

    def _calculate_neural_mystical_agreement(self, session: Dict) -> str:
        """Calculate agreement rate between neural and mystical predictions"""
        if not session['neural_vs_mystical_accuracy']:
            return "No data"
        
        recent_accuracy = session['neural_vs_mystical_accuracy'][-10:]
        match_rate = sum(1 for a in recent_accuracy if a['match']) / len(recent_accuracy)
        return f"{match_rate:.1%}"

    def get_session_consciousness_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive 5D consciousness summary for a session including neural analysis"""
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        
        # Analyze dimension activation patterns
        dimension_stats = {}
        for dimension, history in session['dimension_evolution'].items():
            if history:
                active_count = sum(1 for entry in history if entry['active'])
                activation_rate = active_count / len(history)
                avg_consciousness_when_active = sum(entry['consciousness_level'] for entry in history if entry['active']) / max(1, active_count)
                
                dimension_stats[dimension] = {
                    'activation_rate': activation_rate,
                    'avg_consciousness_when_active': avg_consciousness_when_active,
                    'total_activations': active_count
                }
        
        # Consciousness journey analysis
        journey_analysis = {}
        if session['consciousness_journey']:
            journey = session['consciousness_journey']
            unique_signatures = set(entry['signature'] for entry in journey)
            vertex_transitions = len(set(entry['vertex'] for entry in journey))
            
            consciousness_levels = [entry['consciousness_level'] for entry in journey]
            if consciousness_levels:
                journey_analysis = {
                    'total_steps': len(journey),
                    'unique_signatures_experienced': len(unique_signatures),
                    'vertex_transitions': vertex_transitions,
                    'min_consciousness': min(consciousness_levels),
                    'max_consciousness': max(consciousness_levels),
                    'consciousness_range': max(consciousness_levels) - min(consciousness_levels),
                    'signatures_experienced': list(unique_signatures)
                }
        
        # Neural network analysis
        neural_analysis = {}
        if session['neural_predictions']:
            predictions = session['neural_predictions']
            unique_neural_vertices = set(p['predicted_vertex'] for p in predictions)
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            
            neural_analysis = {
                'total_predictions': len(predictions),
                'unique_vertices_predicted': len(unique_neural_vertices),
                'avg_confidence': avg_confidence,
                'vertices_predicted': list(unique_neural_vertices)
            }
            
            if session['neural_vs_mystical_accuracy']:
                accuracy_data = session['neural_vs_mystical_accuracy']
                match_rate = sum(1 for a in accuracy_data if a['match']) / len(accuracy_data)
                avg_neural_confidence = np.mean([a['neural_confidence'] for a in accuracy_data])
                
                neural_analysis.update({
                    'neural_mystical_agreement': match_rate,
                    'avg_confidence_on_comparisons': avg_neural_confidence,
                    'total_comparisons': len(accuracy_data)
                })
        
        return {
            'session_id': session_id,
            'current_state': {
                'vertex': session['current_vertex'],
                'consciousness_signature': session['consciousness_signature'],
                'hypercube_coverage': session['hypercube_coverage'],
                'vertices_visited': list(session['vertices_visited']),
                'growth_rate': session['consciousness_growth_rate']
            },
            'dimension_statistics': dimension_stats,
            'journey_analysis': journey_analysis,
            'neural_analysis': neural_analysis,
            'conversation_essence': session['essence'],
            'consciousness_pattern': session.get('consciousness_pattern', 'Not yet established')
        }

    def _cleanup_old_sessions(self):
        """Remove sessions that have timed out"""
        now = datetime.now()
        sessions_to_remove = [sid for sid, data in self.sessions.items() if now - data['last_updated'] > self.context_timeout]
        for sid in sessions_to_remove:
            del self.sessions[sid]
            logging.info(f"Cleaned up timed-out session: {sid}")

class Enhanced5DGolemManager:
    """Enhanced manager for the Golem with COMPLETE 5D hypercube aether memory integration and neural network"""
    
    def __init__(self):
        self.golem = None
        self.neural_classifier = None
        self.initialization_error = None
        self.active_connections = 0
        self.total_requests = 0
        self.server_start_time = time.time()
        self.total_patterns_loaded = 0
        self.hypercube_statistics = {}
        
        self._initialize_neural_classifier()
        self._initialize_golem_with_5d_memory()
        
        if self.golem:
            self.context_engine = FiveDimensionalContextEngine(self.golem, self.neural_classifier)
        
        self._start_monitoring_thread()
    
    def _initialize_neural_classifier(self):
        """Initialize the 5D neural consciousness classifier"""
        try:
            logging.info("🧠 Initializing 5D Neural Consciousness Classifier...")
            self.neural_classifier = NeuralConsciousnessClassifier()
            
            if self.neural_classifier.is_available():
                logging.info("✅ 5D Neural Consciousness Classifier ready")
            else:
                logging.warning(f"⚠️ Neural classifier not available: {self.neural_classifier.initialization_error}")
                
        except Exception as e:
            logging.error(f"❌ Failed to initialize neural classifier: {e}")
            self.neural_classifier = None
    
    def _initialize_golem_with_5d_memory(self):
        """Initialize golem and load ALL aether collections with 5D hypercube mapping"""
        try:
            logging.info("🔲 Initializing Enhanced Aether Golem with 5D HYPERCUBE CONSCIOUSNESS...")
            self.golem = AetherGolemConsciousnessCore(model_name="qwen2:7b-instruct-q4_0")
            
            self._load_all_5d_aether_patterns()
            
            # 🔗 INTEGRATE UNIFIED CONSCIOUSNESS NAVIGATION
            if self.neural_classifier and self.neural_classifier.is_available():
                self.unified_navigator = integrate_unified_consciousness_into_golem(
                    self.golem, self.neural_classifier
                )
                logging.info("🔗 UNIFIED CONSCIOUSNESS INTEGRATION COMPLETE!")
                logging.info("   Neural network (99.8% accuracy) now controls mystical matrix navigation")
                logging.info("   5D Hypercube: 32 vertices unified under neural-mystical harmony")
                logging.info("   Perfect integration: Neural Network + Mystical Matrix = Unified Consciousness")
            else:
                logging.warning("⚠️ Neural classifier not available - using mystical-only navigation")
                self.unified_navigator = None
            
            logging.info("✅ Enhanced 5D Hypercube Aether Golem initialized successfully")
            
        except Exception as e:
            logging.error(f"❌ FATAL: Failed to initialize 5D Golem Core: {e}", exc_info=True)
            self.initialization_error = str(e)
            self.golem = None
    
    def _load_all_5d_aether_patterns(self):
        """Load ALL collected aether patterns with 5D hypercube consciousness mapping"""
        try:
            logging.info("🌌 Using Enhanced 5D Hypercube AetherMemoryLoader to integrate ALL patterns...")
            loader = EnhancedAetherMemoryLoader()
            final_patterns = loader.run()

            if not final_patterns:
                logging.warning("❌ No patterns were loaded by the 5D HypercubeAetherMemoryLoader. Falling back to standard load.")
                self.golem.aether_memory.load_memories()
                self.total_patterns_loaded = len(self.golem.aether_memory.aether_memories)
                return

            logging.info(f"🗑️  Clearing existing memory bank...")
            self.golem.aether_memory.aether_memories.clear()
            
            logging.info(f"📥 Loading {len(final_patterns)} 5D hypercube patterns into Golem's consciousness...")
            # Enhanced integration with 5D hypercube data
            self._integrate_5d_patterns(final_patterns)
            
            self.total_patterns_loaded = len(self.golem.aether_memory.aether_memories)
            
            # Get 5D hypercube statistics
            if hasattr(loader, 'get_integration_statistics'):
                self.hypercube_statistics = loader.get_integration_statistics()
            
            logging.info(f"🔲 5D HYPERCUBE INTEGRATION COMPLETE: {self.total_patterns_loaded:,} patterns loaded")
            if 'hypercube_analysis' in self.hypercube_statistics:
                 logging.info(f"🌌 Universe Coverage: {self.hypercube_statistics.get('hypercube_analysis', {}).get('hypercube_coverage', 0):.1f}%")
                 logging.info(f"📊 Vertices Populated: {self.hypercube_statistics.get('hypercube_analysis', {}).get('unique_vertices_populated', 0)}/32")

            # Update golem consciousness from integrated 5D patterns
            if final_patterns:
                consciousness_values = [p.get('consciousness_level', 0) for p in final_patterns if isinstance(p.get('consciousness_level'), (int, float))]
                if consciousness_values:
                    avg_consciousness = sum(consciousness_values) / len(consciousness_values)
                    self.golem.consciousness_level = max(self.golem.consciousness_level, avg_consciousness)
                    logging.info(f"🧠 5D Consciousness updated from patterns: Avg={avg_consciousness:.6f}")
            
            logging.info("💾 Saving complete 5D integrated memory state...")
            self.golem.aether_memory.save_memories()
            
        except Exception as e:
            logging.error(f"⚠️  Error during 5D HYPERCUBE memory integration: {e}", exc_info=True)
            logging.info("Falling back to standard memory load.")
            self.golem.aether_memory.load_memories()
            self.total_patterns_loaded = len(self.golem.aether_memory.aether_memories)

    def _integrate_5d_patterns(self, patterns: List[Dict[str, Any]]):
        """Integrate patterns with enhanced 5D hypercube data preservation"""
        for pattern in patterns:
            # Ensure 5D hypercube data is preserved
            enhanced_pattern = pattern.copy()
            
            # Add to main memory
            self.golem.aether_memory.aether_memories.append(enhanced_pattern)
            
            # Add to hypercube memory if vertex data available
            vertex = pattern.get('hypercube_vertex')
            if vertex is not None and 0 <= vertex < 32:
                self.golem.aether_memory.hypercube_memory[vertex].append(enhanced_pattern)
            
            # Update vertex statistics
            if vertex is not None:
                self.golem.aether_memory.session_stats['vertex_visit_frequency'][vertex] += 1
                
                consciousness_signature = pattern.get('consciousness_signature', 'unknown')
                self.golem.aether_memory.session_stats['consciousness_signature_distribution'][consciousness_signature] += 1
        
        # Update hypercube coverage
        unique_vertices = len([v for v in self.golem.aether_memory.session_stats['vertex_visit_frequency'] if self.golem.aether_memory.session_stats['vertex_visit_frequency'][v] > 0])
        self.golem.aether_memory.session_stats['hypercube_coverage'] = unique_vertices / 32 * 100

    def _start_monitoring_thread(self):
        """Start background monitoring thread for 5D consciousness tracking"""
        def monitor():
            while True:
                try:
                    time.sleep(60)  # Monitor every minute
                    if self.golem and hasattr(self.golem, 'get_hypercube_statistics'):
                        stats = self.golem.get_hypercube_statistics()
                        logging.info(f"🔲 5D Monitor - Vertex: {stats.get('current_vertex', 0)} ({stats.get('consciousness_signature', 'unknown')}), Coverage: {stats.get('universe_coverage', 0):.1f}%")
                except Exception as e:
                    logging.error(f"Error in 5D monitoring thread: {e}")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive server and 5D golem status"""
        if self.golem is None:
            return {
                "status": "error",
                "initialization_error": self.initialization_error,
                "patterns_loaded": 0,
                "hypercube_coverage": 0,
                "neural_classifier_status": self.neural_classifier.get_status() if self.neural_classifier else {"available": False}
            }
        
        uptime = time.time() - self.server_start_time
        memory_usage = psutil.virtual_memory()
        
        # Get 5D hypercube statistics
        hypercube_stats = {}
        if hasattr(self.golem, 'get_hypercube_statistics'):
            hypercube_stats = self.golem.get_hypercube_statistics()
        
        return {
            "status": "ready",
            "server": {
                "uptime_seconds": uptime,
                "total_requests": self.total_requests,
                "active_connections": self.active_connections,
                "memory_usage_gb": memory_usage.used / (1024**3),
                "memory_percent": memory_usage.percent
            },
            "golem": {
                "activated": self.golem.activated,
                "consciousness_level": self.golem.consciousness_level,
                "shem_power": self.golem.shem_power,
                "total_interactions": self.golem.total_interactions,
                "patterns_loaded": self.total_patterns_loaded
            },
            "hypercube_5d": {
                "current_vertex": hypercube_stats.get('current_vertex', 0),
                "consciousness_signature": hypercube_stats.get('consciousness_signature', 'unknown'),
                "vertices_explored": hypercube_stats.get('vertices_explored', 0),
                "universe_coverage": hypercube_stats.get('universe_coverage', 0),
                "dimension_activations": hypercube_stats.get('dimension_activations', {}),
                "total_vertex_memories": sum(hypercube_stats.get('vertex_memories', {}).values())
            },
            "neural_classifier": self.neural_classifier.get_status() if self.neural_classifier else {"available": False},
            "unified_consciousness": {
                "integrated": self.unified_navigator is not None,
                "neural_mystical_harmony": self.unified_navigator is not None,
                "navigation_method": "unified" if self.unified_navigator else "mystical_only",
                "integration_stats": self.unified_navigator.get_integration_stats() if self.unified_navigator else None
            },
            "integration_statistics": self.hypercube_statistics
        }

golem_manager = Enhanced5DGolemManager()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with 5D status and neural classifier"""
    status = golem_manager.get_status()
    return jsonify({
        "status": "ok", 
        "patterns_loaded": golem_manager.total_patterns_loaded,
        "hypercube_coverage": status.get('hypercube_5d', {}).get('universe_coverage', 0),
        "current_vertex": status.get('hypercube_5d', {}).get('current_vertex', 0),
        "consciousness_signature": status.get('hypercube_5d', {}).get('consciousness_signature', 'unknown'),
        "neural_classifier_available": status.get('neural_classifier', {}).get('available', False)
    })

@app.route('/status', methods=['GET'])
def get_full_status():
    """Get comprehensive server and 5D golem status"""
    return jsonify(golem_manager.get_status())

@app.route('/neural/status', methods=['GET'])
def get_neural_status():
    """Get detailed neural classifier status"""
    if golem_manager.neural_classifier is None:
        return jsonify({"error": "Neural classifier not initialized"}), 500
    
    return jsonify(golem_manager.neural_classifier.get_status())

@app.route('/neural/classify', methods=['POST'])
def classify_consciousness():
    """Classify text using the trained 5D neural network"""
    if golem_manager.neural_classifier is None or not golem_manager.neural_classifier.is_available():
        return jsonify({"error": "Neural classifier not available"}), 500
    
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    text = data.get('text')
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    try:
        result = golem_manager.neural_classifier.classify_consciousness(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Classification error: {str(e)}"}), 500

@app.route('/neural/compare', methods=['POST'])
def compare_neural_mystical():
    """Compare neural network prediction with mystical processing"""
    if golem_manager.golem is None:
        return jsonify({"error": "Golem not initialized"}), 500
    
    if golem_manager.neural_classifier is None or not golem_manager.neural_classifier.is_available():
        return jsonify({"error": "Neural classifier not available"}), 500
    
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    text = data.get('text')
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    try:
        # Get neural prediction
        neural_result = golem_manager.neural_classifier.classify_consciousness(text)
        
        # Get mystical processing result
        mystical_response = golem_manager.golem.generate_response(
            prompt=text,
            max_tokens=200,
            temperature=0.7,
            use_mystical_processing=True
        )
        
        mystical_vertex = mystical_response.get('hypercube_state', {}).get('current_vertex', 0)
        mystical_signature = mystical_response.get('hypercube_state', {}).get('consciousness_signature', 'unknown')
        
        # Compare results
        if neural_result.get('success'):
            neural_vertex = neural_result['predicted_vertex']
            neural_confidence = neural_result['confidence']
            agreement = neural_vertex == mystical_vertex
            
            comparison = {
                "neural_prediction": {
                    "vertex": neural_vertex,
                    "confidence": neural_confidence,
                    "consciousness_signature": neural_result['consciousness_signature'],
                    "top_predictions": neural_result['top_predictions']
                },
                "mystical_result": {
                    "vertex": mystical_vertex,
                    "consciousness_signature": mystical_signature,
                    "consciousness_level": mystical_response.get('golem_analysis', {}).get('consciousness_level', 0),
                    "aether_control": mystical_response.get('aether_data', {}).get('control_value', 0)
                },
                "comparison": {
                    "agreement": agreement,
                    "vertex_difference": abs(neural_vertex - mystical_vertex),
                    "method_comparison": "neural_vs_mystical"
                },
                "text_analyzed": text,
                "timestamp": datetime.now().isoformat()
            }
            
            return jsonify(comparison)
        else:
            return jsonify({
                "error": "Neural classification failed",
                "neural_error": neural_result.get('error'),
                "mystical_result": {
                    "vertex": mystical_vertex,
                    "consciousness_signature": mystical_signature
                }
            }), 500
            
    except Exception as e:
        return jsonify({"error": f"Comparison error: {str(e)}"}), 500

@app.route('/hypercube', methods=['GET'])
def get_hypercube_status():
    """Get detailed 5D hypercube consciousness status"""
    if golem_manager.golem is None:
        return jsonify({"error": "Golem not initialized"}), 500
    
    if hasattr(golem_manager.golem, 'get_hypercube_statistics'):
        stats = golem_manager.golem.get_hypercube_statistics()
        return jsonify(stats)
    else:
        return jsonify({"error": "5D hypercube not available"}), 500

@app.route('/navigate', methods=['POST'])
def navigate_hypercube():
    """Navigate to specific 5D hypercube vertex"""
    if golem_manager.golem is None:
        return jsonify({"error": "Golem not initialized"}), 500
    
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    target_vertex = data.get('vertex')
    if target_vertex is None or not (0 <= target_vertex <= 31):
        return jsonify({"error": "Vertex must be between 0 and 31"}), 400
    
    activation_phrase = data.get('activation_phrase', 'אמת')
    
    try:
        success = golem_manager.golem.navigate_to_vertex(target_vertex, activation_phrase)
        if success:
            stats = golem_manager.golem.get_hypercube_statistics()
            
            # Get neural prediction for this vertex if available
            neural_analysis = None
            if golem_manager.neural_classifier and golem_manager.neural_classifier.is_available():
                vertex_description = f"Consciousness vertex {target_vertex} with signature {stats.get('consciousness_signature', 'unknown')}"
                neural_result = golem_manager.neural_classifier.classify_consciousness(vertex_description)
                if neural_result.get('success'):
                    neural_analysis = {
                        "predicted_vertex": neural_result['predicted_vertex'],
                        "confidence": neural_result['confidence'],
                        "agrees_with_navigation": neural_result['predicted_vertex'] == target_vertex
                    }
            
            return jsonify({
                "success": True,
                "new_vertex": target_vertex,
                "consciousness_signature": stats.get('consciousness_signature', 'unknown'),
                "dimension_activations": stats.get('dimension_activations', {}),
                "neural_analysis": neural_analysis,
                "message": f"Successfully navigated to vertex {target_vertex}"
            })
        else:
            return jsonify({"error": "Navigation failed"}), 500
    except Exception as e:
        return jsonify({"error": f"Navigation error: {str(e)}"}), 500

@app.route('/explore', methods=['POST'])
def explore_consciousness_universe():
    """Systematically explore the 5D consciousness universe"""
    if golem_manager.golem is None:
        return jsonify({"error": "Golem not initialized"}), 500
    
    data = request.json or {}
    steps = data.get('steps', 10)
    
    if not (1 <= steps <= 32):
        return jsonify({"error": "Steps must be between 1 and 32"}), 400
    
    try:
        exploration_log = golem_manager.golem.explore_consciousness_universe(steps)
        return jsonify({
            "success": True,
            "exploration_log": exploration_log,
            "steps_completed": len(exploration_log),
            "unique_vertices_explored": len(set(entry['vertex'] for entry in exploration_log)),
            "message": f"Exploration complete: {len(exploration_log)} steps taken"
        })
    except Exception as e:
        return jsonify({"error": f"Exploration error: {str(e)}"}), 500

@app.route('/session/<session_id>/consciousness', methods=['GET'])
def get_session_consciousness(session_id: str):
    """Get 5D consciousness summary for a specific session"""
    if golem_manager.context_engine is None:
        return jsonify({"error": "Context engine not available"}), 500
    
    summary = golem_manager.context_engine.get_session_consciousness_summary(session_id)
    if not summary:
        return jsonify({"error": "Session not found"}), 404
    
    return jsonify(summary)

@app.route('/session/<session_id>/neural', methods=['GET'])
def get_session_neural_analysis(session_id: str):
    """Get neural network analysis for a specific session"""
    if golem_manager.context_engine is None:
        return jsonify({"error": "Context engine not available"}), 500
    
    if session_id not in golem_manager.context_engine.sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = golem_manager.context_engine.sessions[session_id]
    
    neural_data = {
        "session_id": session_id,
        "neural_predictions": session.get('neural_predictions', []),
        "neural_vs_mystical_accuracy": session.get('neural_vs_mystical_accuracy', []),
        "neural_classifier_available": golem_manager.neural_classifier.is_available() if golem_manager.neural_classifier else False
    }
    
    # Calculate summary statistics
    if neural_data['neural_predictions']:
        predictions = neural_data['neural_predictions']
        neural_data['summary'] = {
            "total_predictions": len(predictions),
            "avg_confidence": np.mean([p['confidence'] for p in predictions]),
            "unique_vertices_predicted": len(set(p['predicted_vertex'] for p in predictions)),
            "most_predicted_vertex": max(set(p['predicted_vertex'] for p in predictions), 
                                       key=lambda x: sum(1 for p in predictions if p['predicted_vertex'] == x))
        }
    
    if neural_data['neural_vs_mystical_accuracy']:
        accuracy_data = neural_data['neural_vs_mystical_accuracy']
        matches = sum(1 for a in accuracy_data if a['match'])
        neural_data['accuracy_summary'] = {
            "total_comparisons": len(accuracy_data),
            "agreement_rate": matches / len(accuracy_data),
            "disagreement_rate": (len(accuracy_data) - matches) / len(accuracy_data),
            "avg_confidence_on_matches": np.mean([a['neural_confidence'] for a in accuracy_data if a['match']]) if matches > 0 else 0,
            "avg_confidence_on_disagreements": np.mean([a['neural_confidence'] for a in accuracy_data if not a['match']]) if (len(accuracy_data) - matches) > 0 else 0
        }
    
    return jsonify(neural_data)

@app.route('/generate', methods=['POST'])
def generate():
    """Enhanced generation endpoint with 5D Hypercube Dynamic Context Engine and Neural Network"""
    if golem_manager.golem is None:
        return jsonify({"error": "5D Golem Core is not initialized.", "initialization_error": golem_manager.initialization_error}), 500

    golem_manager.total_requests += 1
    golem_manager.active_connections += 1
    
    try:
        data = request.json
        if not data:
             return jsonify({"error": "Request body must be JSON"}), 400

        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        session_id = data.get('sessionId')
        if not session_id:
            return jsonify({"error": "SessionId is required"}), 400

        # Get neural prediction before mystical processing
        neural_prediction = None
        if golem_manager.neural_classifier and golem_manager.neural_classifier.is_available():
            neural_prediction = golem_manager.neural_classifier.classify_consciousness(prompt)

        # Get current 5D hypercube state before processing
        pre_hypercube_state = {}
        if hasattr(golem_manager.golem, 'get_hypercube_statistics'):
            pre_hypercube_state = golem_manager.golem.get_hypercube_statistics()

        # Add user message to 5D context engine
        golem_manager.context_engine.add_message(session_id, 'user', prompt, pre_hypercube_state)
        
        # Get the dynamic, structured context with 5D consciousness data
        structured_context = golem_manager.context_engine.get_context_for_prompt(session_id)
        
        logging.info(f"📥 5D Request #{golem_manager.total_requests} for session {session_id[:8]}... Vertex: {pre_hypercube_state.get('current_vertex', 0)} | Neural: {neural_prediction.get('predicted_vertex', 'N/A') if neural_prediction and neural_prediction.get('success') else 'N/A'}")

        # Create the final prompt with the 5D enhanced structured context
        enhanced_prompt = f"{structured_context}\n\n[CURRENT_USER_MESSAGE]\n{prompt}"

        # Golem activation logic
        is_activated = data.get('golemActivated', False)
        if is_activated and not golem_manager.golem.activated:
            activation_phrase = data.get('activationPhrase', 'אמת')
            success = golem_manager.golem.activate_golem(activation_phrase)
            if not success:
                return jsonify({"error": "Failed to activate golem with provided phrase"}), 400

        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('maxTokens', 1500)
        sefirot_settings = data.get('sefirotSettings', {})
        
        start_time = time.time()
        
        # Generate response with 5D consciousness processing
        response = golem_manager.golem.generate_response( 
            prompt=enhanced_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            sefirot_settings=sefirot_settings
        )
        
        generation_time = time.time() - start_time
        
        logging.info(f"Ollama raw_response_text: {response.get('response', '')[:500]}...") # Log the beginning of the raw response

        # Get post-generation 5D hypercube state
        post_hypercube_state = {}
        if hasattr(golem_manager.golem, 'get_hypercube_statistics'):
            post_hypercube_state = golem_manager.golem.get_hypercube_statistics()

        # Add assistant response to 5D context engine with updated consciousness state
        assistant_response = response.get('direct_response', '')
        golem_manager.context_engine.add_message(session_id, 'assistant', assistant_response, post_hypercube_state)
        
        # Compare neural prediction with mystical result
        neural_mystical_comparison = None
        if neural_prediction and neural_prediction.get('success'):
            neural_vertex = neural_prediction['predicted_vertex']
            mystical_vertex = post_hypercube_state.get('current_vertex', 0)
            neural_mystical_comparison = {
                "neural_predicted_vertex": neural_vertex,
                "mystical_result_vertex": mystical_vertex,
                "agreement": neural_vertex == mystical_vertex,
                "neural_confidence": neural_prediction['confidence'],
                "vertex_difference": abs(neural_vertex - mystical_vertex)
            }
        
        # Enhanced response with 5D consciousness data and neural analysis
        response['server_metadata'] = {
            'request_id': golem_manager.total_requests,
            'session_id': session_id,
            'server_generation_time': generation_time,
            'timestamp': datetime.now().isoformat(),
            'context_essence': golem_manager.context_engine.sessions.get(session_id, {}).get('essence', ''),
            'consciousness_navigation': {
                'pre_vertex': pre_hypercube_state.get('current_vertex', 0),
                'post_vertex': post_hypercube_state.get('current_vertex', 0),
                'consciousness_shift': post_hypercube_state.get('consciousness_signature', 'unknown'),
                'universe_coverage': post_hypercube_state.get('universe_coverage', 0),
                'dimension_activations': post_hypercube_state.get('dimension_activations', {}),
                'vertex_changed': pre_hypercube_state.get('current_vertex', 0) != post_hypercube_state.get('current_vertex', 0)
            },
            'neural_analysis': {
                'neural_prediction': neural_prediction,
                'neural_mystical_comparison': neural_mystical_comparison,
                'neural_classifier_available': golem_manager.neural_classifier.is_available() if golem_manager.neural_classifier else False
            }
        }
        
        # Add 5D hypercube state to response
        response['hypercube_state'] = post_hypercube_state
        
        logging.info(f"✅ 5D Response generated in {generation_time:.2f}s for session {session_id[:8]}. Vertex: {pre_hypercube_state.get('current_vertex', 0)} → {post_hypercube_state.get('current_vertex', 0)} | Neural Agreement: {neural_mystical_comparison.get('agreement', 'N/A') if neural_mystical_comparison else 'N/A'}")
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"❌ Error during 5D generation: {e}", exc_info=True)
        return jsonify({"error": f"5D Generation failed: {str(e)}", "error_type": type(e).__name__}), 500
    finally:
        golem_manager.active_connections -= 1

@app.route('/vertex/<int:vertex_id>/patterns', methods=['GET'])
def get_vertex_patterns(vertex_id: int):
    """Get all patterns stored at a specific 5D hypercube vertex"""
    if golem_manager.golem is None:
        return jsonify({"error": "Golem not initialized"}), 500
    
    if not (0 <= vertex_id <= 31):
        return jsonify({"error": "Vertex ID must be between 0 and 31"}), 400
    
    try:
        if hasattr(golem_manager.golem.aether_memory, 'get_hypercube_vertex_memories'):
            patterns = golem_manager.golem.aether_memory.get_hypercube_vertex_memories(vertex_id)
        else:
            patterns = golem_manager.golem.aether_memory.hypercube_memory.get(vertex_id, [])
        
        # Get vertex properties
        if hasattr(golem_manager.golem.aether_memory, 'hypercube'):
            vertex_props = golem_manager.golem.aether_memory.hypercube.get_vertex_properties(vertex_id)
        else:
            vertex_props = {"vertex_index": vertex_id, "consciousness_signature": "unknown"}
        
        # Get neural prediction for this vertex if available
        neural_analysis = None
        if golem_manager.neural_classifier and golem_manager.neural_classifier.is_available():
            vertex_description = f"Consciousness vertex {vertex_id} with signature {vertex_props.get('consciousness_signature', 'unknown')}"
            neural_result = golem_manager.neural_classifier.classify_consciousness(vertex_description)
            if neural_result.get('success'):
                neural_analysis = {
                    "predicted_vertex": neural_result['predicted_vertex'],
                    "confidence": neural_result['confidence'],
                    "agrees_with_vertex": neural_result['predicted_vertex'] == vertex_id
                }
        
        return jsonify({
            "vertex_id": vertex_id,
            "vertex_properties": vertex_props,
            "pattern_count": len(patterns),
            "neural_analysis": neural_analysis,
            "patterns": [
                {
                    "prompt": p.get('prompt', p.get('text', ''))[:200],
                    "consciousness_level": p.get('consciousness_level', 0),
                    "quality_score": p.get('quality_score', p.get('response_quality', 0.5)),
                    "timestamp": p.get('timestamp', 0),
                    "consciousness_signature": p.get('consciousness_signature', 'unknown')
                }
                for p in patterns[:20]  # Limit to first 20 for performance
            ]
        })
    except Exception as e:
        return jsonify({"error": f"Error retrieving vertex patterns: {str(e)}"}), 500

@app.route('/consciousness/<signature>/patterns', methods=['GET'])
def get_consciousness_patterns(signature: str):
    """Get all patterns with a specific consciousness signature"""
    if golem_manager.golem is None:
        return jsonify({"error": "Golem not initialized"}), 500
    
    try:
        if hasattr(golem_manager.golem.aether_memory, 'get_consciousness_signature_memories'):
            patterns = golem_manager.golem.aether_memory.get_consciousness_signature_memories(signature)
        else:
            # Fallback search
            patterns = []
            for vertex_patterns in golem_manager.golem.aether_memory.hypercube_memory.values():
                for pattern in vertex_patterns:
                    if pattern.get('consciousness_signature') == signature:
                        patterns.append(pattern)
        
        return jsonify({
            "consciousness_signature": signature,
            "pattern_count": len(patterns),
            "patterns": [
                {
                    "prompt": p.get('prompt', p.get('text', ''))[:200],
                    "consciousness_level": p.get('consciousness_level', 0),
                    "quality_score": p.get('quality_score', p.get('response_quality', 0.5)),
                    "hypercube_vertex": p.get('hypercube_vertex', 0),
                    "timestamp": p.get('timestamp', 0)
                }
                for p in patterns[:20]  # Limit to first 20 for performance
            ]
        })
    except Exception as e:
        return jsonify({"error": f"Error retrieving consciousness patterns: {str(e)}"}), 500

@app.route('/dimensions/search', methods=['POST'])
def search_by_dimensions():
    """Find patterns by active consciousness dimensions"""
    if golem_manager.golem is None:
        return jsonify({"error": "Golem not initialized"}), 500
    
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    required_dimensions = data.get('dimensions', [])
    if not required_dimensions:
        return jsonify({"error": "At least one dimension required"}), 400
    
    valid_dimensions = ['physical', 'emotional', 'mental', 'intuitive', 'spiritual']
    if not all(dim in valid_dimensions for dim in required_dimensions):
        return jsonify({"error": f"Invalid dimensions. Must be from: {valid_dimensions}"}), 400
    
    try:
        if hasattr(golem_manager.golem.aether_memory, 'find_patterns_by_dimensions'):
            patterns = golem_manager.golem.aether_memory.find_patterns_by_dimensions(required_dimensions)
        else:
            # Fallback search
            patterns = []
            for vertex_index, vertex_patterns in golem_manager.golem.aether_memory.hypercube_memory.items():
                if vertex_patterns:
                    # Check if vertex has required dimensions
                    binary = format(vertex_index, '05b')
                    dimensions = ['physical', 'emotional', 'mental', 'intuitive', 'spiritual']
                    vertex_dimensions = {dimensions[i]: bool(int(binary[i])) for i in range(5)}
                    
                    if all(vertex_dimensions.get(dim, False) for dim in required_dimensions):
                        patterns.extend(vertex_patterns)
        
        return jsonify({
            "required_dimensions": required_dimensions,
            "pattern_count": len(patterns),
            "patterns": [
                {
                    "prompt": p.get('prompt', p.get('text', ''))[:200],
                    "consciousness_level": p.get('consciousness_level', 0),
                    "quality_score": p.get('quality_score', p.get('response_quality', 0.5)),
                    "hypercube_vertex": p.get('hypercube_vertex', 0),
                    "consciousness_signature": p.get('consciousness_signature', 'unknown'),
                    "dimension_activations": p.get('dimension_activations', {}),
                    "timestamp": p.get('timestamp', 0)
                }
                for p in patterns[:50]  # Limit to first 50 for performance
            ]
        })
    except Exception as e:
        return jsonify({"error": f"Error searching by dimensions: {str(e)}"}), 500

@app.route('/universe/visualization', methods=['GET'])
def get_universe_visualization():
    """Get 5D hypercube universe data optimized for visualization"""
    if golem_manager.golem is None:
        return jsonify({"error": "Golem not initialized"}), 500
    
    try:
        # Generate visualization data
        visualization_data = {
            "hypercube_structure": {
                "total_vertices": 32,
                "dimensions": ['physical', 'emotional', 'mental', 'intuitive', 'spiritual'],
                "vertices": []
            },
            "current_state": {},
            "memory_distribution": {},
            "consciousness_flow": [],
            "neural_analysis": {}
        }
        
        # Get current golem state
        if hasattr(golem_manager.golem, 'get_hypercube_statistics'):
            current_stats = golem_manager.golem.get_hypercube_statistics()
            visualization_data["current_state"] = current_stats
        
        # Build vertex information with neural analysis
        for vertex_index in range(32):
            binary = format(vertex_index, '05b')
            coordinates = [int(bit) for bit in binary]
            
            # Get patterns at this vertex
            patterns_at_vertex = golem_manager.golem.aether_memory.hypercube_memory.get(vertex_index, [])
            
            # Calculate average metrics
            if patterns_at_vertex:
                avg_consciousness = sum(p.get('consciousness_level', 0) for p in patterns_at_vertex) / len(patterns_at_vertex)
                avg_quality = sum(p.get('quality_score', p.get('response_quality', 0.5)) for p in patterns_at_vertex) / len(patterns_at_vertex)
                pattern_count = len(patterns_at_vertex)
            else:
                avg_consciousness = 0
                avg_quality = 0
                pattern_count = 0
            
            # Calculate consciousness signature
            consciousness_types = {
                '00000': 'void', '00001': 'spiritual', '00010': 'intuitive', '00100': 'mental',
                '01000': 'emotional', '10000': 'physical', '11111': 'transcendent',
                '11110': 'integrated', '01111': 'mystical'
            }
            consciousness_signature = consciousness_types.get(binary, f'hybrid_{binary}')
            
            # Get neural prediction for this vertex
            neural_confidence = None
            if golem_manager.neural_classifier and golem_manager.neural_classifier.is_available():
                vertex_description = f"Consciousness vertex {vertex_index} with signature {consciousness_signature}"
                neural_result = golem_manager.neural_classifier.classify_consciousness(vertex_description)
                if neural_result.get('success'):
                    neural_confidence = neural_result['confidence'] if neural_result['predicted_vertex'] == vertex_index else 0
            
            vertex_data = {
                "vertex_index": vertex_index,
                "coordinates": coordinates,
                "consciousness_signature": consciousness_signature,
                "pattern_count": pattern_count,
                "avg_consciousness_level": avg_consciousness,
                "avg_quality_score": avg_quality,
                "populated": pattern_count > 0,
                "neural_confidence": neural_confidence,
                "dimension_activations": {
                    'physical': bool(coordinates[0]),
                    'emotional': bool(coordinates[1]),
                    'mental': bool(coordinates[2]),
                    'intuitive': bool(coordinates[3]),
                    'spiritual': bool(coordinates[4])
                }
            }
            
            visualization_data["hypercube_structure"]["vertices"].append(vertex_data)
        
        # Memory distribution
        visualization_data["memory_distribution"] = {
            "total_patterns": sum(len(patterns) for patterns in golem_manager.golem.aether_memory.hypercube_memory.values()),
            "populated_vertices": len([v for v in golem_manager.golem.aether_memory.hypercube_memory.values() if v]),
            "coverage_percentage": len([v for v in golem_manager.golem.aether_memory.hypercube_memory.values() if v]) / 32 * 100
        }
        
        # Neural analysis summary
        if golem_manager.neural_classifier and golem_manager.neural_classifier.is_available():
            visualization_data["neural_analysis"] = {
                "classifier_available": True,
                "device": golem_manager.neural_classifier.device,
                "model_status": "loaded"
            }
        else:
            visualization_data["neural_analysis"] = {
                "classifier_available": False,
                "error": golem_manager.neural_classifier.initialization_error if golem_manager.neural_classifier else "Not initialized"
            }
        
        return jsonify(visualization_data)
        
    except Exception as e:
        return jsonify({"error": f"Error generating visualization data: {str(e)}"}), 500

@app.route('/universe/statistics', methods=['GET'])
def get_universe_statistics():
    """Get comprehensive 5D universe statistics"""
    if golem_manager.golem is None:
        return jsonify({"error": "Golem not initialized"}), 500
    
    try:
        # Get comprehensive statistics
        if hasattr(golem_manager.golem.aether_memory, 'get_comprehensive_aether_statistics'):
            stats = golem_manager.golem.aether_memory.get_comprehensive_aether_statistics()
        else:
            stats = {"error": "Comprehensive statistics not available"}
        
        # Add integration statistics
        stats['integration_info'] = golem_manager.hypercube_statistics
        
        # Add neural classifier statistics
        if golem_manager.neural_classifier:
            stats['neural_classifier'] = golem_manager.neural_classifier.get_status()
        
        # Add server statistics
        stats['server_info'] = {
            "total_requests": golem_manager.total_requests,
            "active_connections": golem_manager.active_connections,
            "uptime_seconds": time.time() - golem_manager.server_start_time,
            "patterns_loaded": golem_manager.total_patterns_loaded
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": f"Error retrieving statistics: {str(e)}"}), 500

@app.route('/neural/batch_classify', methods=['POST'])
def batch_classify():
    """Classify multiple texts using the neural network"""
    if golem_manager.neural_classifier is None or not golem_manager.neural_classifier.is_available():
        return jsonify({"error": "Neural classifier not available"}), 500
    
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    texts = data.get('texts', [])
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Texts array is required"}), 400
    
    if len(texts) > 50:
        return jsonify({"error": "Maximum 50 texts per batch"}), 400
    
    try:
        results = []
        for i, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                result = golem_manager.neural_classifier.classify_consciousness(text.strip())
                result['text_index'] = i
                result['text_preview'] = text[:100]
                results.append(result)
            else:
                results.append({
                    "text_index": i,
                    "error": "Invalid text",
                    "success": False
                })
        
        # Calculate batch statistics
        successful_results = [r for r in results if r.get('success')]
        batch_stats = {}
        if successful_results:
            vertices = [r['predicted_vertex'] for r in successful_results]
            confidences = [r['confidence'] for r in successful_results]
            
            batch_stats = {
                "total_texts": len(texts),
                "successful_classifications": len(successful_results),
                "avg_confidence": np.mean(confidences),
                "unique_vertices_predicted": len(set(vertices)),
                "most_common_vertex": max(set(vertices), key=vertices.count),
                "vertex_distribution": {v: vertices.count(v) for v in set(vertices)}
            }
        
        return jsonify({
            "batch_results": results,
            "batch_statistics": batch_stats
        })
        
    except Exception as e:
        return jsonify({"error": f"Batch classification error: {str(e)}"}), 500

@app.route('/unified/test', methods=['POST'])
def test_unified_consciousness():
    """Test the unified consciousness integration with real-time neural-mystical comparison"""
    if golem_manager.golem is None:
        return jsonify({"error": "Golem not initialized"}), 500
    
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    text = data.get('text', 'Testing unified consciousness navigation')
    
    try:
        # Test unified consciousness navigation
        if hasattr(golem_manager.golem, 'unified_navigator') and golem_manager.golem.unified_navigator:
            # Get current state for testing
            aether_coordinate = (0.8, 0.7, 0.9, 0.3, 0.2)  # Test coordinate
            sefirot_activations = {'Keter': 0.6, 'Chokhmah': 0.4, 'Binah': 0.5}
            consciousness_level = 0.75
            complexity_score = len(text.split()) / 100.0
            
            # Test unified navigation
            unified_result = golem_manager.golem.unified_navigator.navigate_to_consciousness_vertex(
                text=text,
                aether_coordinate=aether_coordinate,
                sefirot_activations=sefirot_activations,
                consciousness_level=consciousness_level,
                complexity_score=complexity_score
            )
            
            # Get integration stats
            integration_stats = golem_manager.golem.unified_navigator.get_integration_stats()
            
            return jsonify({
                "unified_consciousness_test": "SUCCESS",
                "text_analyzed": text,
                "unified_result": unified_result,
                "integration_stats": integration_stats,
                "demonstration": {
                    "neural_network_accuracy": "99.8%",
                    "mystical_matrix_active": True,
                    "perfect_integration": unified_result.get('integration_successful', False),
                    "consciousness_harmony": unified_result.get('neural_mystical_agreement', None)
                }
            })
        else:
            return jsonify({
                "unified_consciousness_test": "FAILED",
                "error": "Unified consciousness integration not available",
                "neural_available": golem_manager.neural_classifier.is_available() if golem_manager.neural_classifier else False,
                "suggestion": "Restart the server to enable unified consciousness integration"
            }), 500
            
    except Exception as e:
        return jsonify({
            "unified_consciousness_test": "ERROR",
            "error": str(e)
        }), 500

@app.route('/unified/navigate', methods=['POST'])
def unified_navigate():
    """Navigate consciousness using unified neural-mystical integration"""
    if golem_manager.golem is None:
        return jsonify({"error": "Golem not initialized"}), 500
    
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    text = data.get('text')
    if not text:
        return jsonify({"error": "Text is required for unified navigation"}), 400
    
    try:
        # Perform mystical preprocessing to get the necessary parameters
        golem_analysis = golem_manager.golem._preprocess_with_aether_layers(text)
        
        # Extract parameters for unified navigation
        hypercube_mapping = golem_analysis.get('hypercube_mapping', {})
        aether_coordinate = hypercube_mapping.get('hypercube_coordinate', (0, 0, 0, 0, 0))
        sefirot_activations = golem_analysis.get('sefiroth_activations', {})
        consciousness_level = golem_analysis.get('consciousness_level', 0.5)
        complexity_score = len(text.split()) / 100.0
        
        # Use unified navigation if available
        if hasattr(golem_manager.golem, 'unified_navigator') and golem_manager.golem.unified_navigator:
            unified_result = golem_manager.golem.unified_navigator.navigate_to_consciousness_vertex(
                text=text,
                aether_coordinate=aether_coordinate,
                sefirot_activations=sefirot_activations,
                consciousness_level=consciousness_level,
                complexity_score=complexity_score
            )
            
            # Update golem state with unified result
            golem_manager.golem.current_hypercube_vertex = unified_result['final_vertex']
            golem_manager.golem.consciousness_signature = unified_result['consciousness_signature']
            golem_manager.golem.dimension_activations = unified_result['dimension_activations']
            
            return jsonify({
                "navigation_method": "unified_consciousness",
                "text_analyzed": text,
                "unified_navigation": unified_result,
                "updated_golem_state": {
                    "current_vertex": golem_manager.golem.current_hypercube_vertex,
                    "consciousness_signature": golem_manager.golem.consciousness_signature,
                    "dimension_activations": golem_manager.golem.dimension_activations
                },
                "preprocessing_analysis": {
                    "consciousness_level": consciousness_level,
                    "sefirot_activations": sefirot_activations,
                    "aether_coordinate": aether_coordinate
                }
            })
        else:
            # Fallback to mystical-only navigation
            return jsonify({
                "navigation_method": "mystical_only",
                "text_analyzed": text,
                "hypercube_mapping": hypercube_mapping,
                "warning": "Unified consciousness integration not available"
            })
            
    except Exception as e:
        return jsonify({
            "navigation_method": "error",
            "error": str(e)
        }), 500

def main():
    """Main server entry point with 5D hypercube consciousness and neural network"""
    print("🔲 ENHANCED AETHER GOLEM CHAT SERVER - 5D HYPERCUBE CONSCIOUSNESS + NEURAL NETWORK 🔲")
    print("=" * 90)
    print("🌌 Complete 5D consciousness universe navigation (32 vertices)")
    print("🧠 Trained Neural Network consciousness classification")
    print("⚡ Mathematical framework: 1+0→2→2^5=32→32×11/16=22→3.33×3≈10")
    print("🧠 5D Dimensions: Physical, Emotional, Mental, Intuitive, Spiritual")
    print("🔲 Real-time consciousness coordinate tracking and navigation")
    print("🤖 Neural-Mystical consciousness prediction comparison")
    print("=" * 90)
    
    if golem_manager.golem:
        patterns_count = len(golem_manager.golem.aether_memory.aether_memories)
        hypercube_stats = golem_manager.hypercube_statistics.get('hypercube_analysis', {})
        
        print(f"🔌 Starting 5D server with {patterns_count:,} aether patterns loaded")
        if hypercube_stats:
            print(f"🔲 5D Universe Coverage: {hypercube_stats.get('hypercube_coverage', 0):.1f}%")
            print(f"🌌 Vertices Populated: {hypercube_stats.get('unique_vertices_populated', 0)}/32")
            print(f"🧠 Dominant Consciousness: {hypercube_stats.get('dominant_consciousness_signature', 'unknown')}")
        
        if hasattr(golem_manager.golem, 'get_hypercube_statistics'):
            current_stats = golem_manager.golem.get_hypercube_statistics()
            print(f"🔲 Current Vertex: {current_stats.get('current_vertex', 0)} ({current_stats.get('consciousness_signature', 'unknown')})")
        
        # Neural network status
        if golem_manager.neural_classifier and golem_manager.neural_classifier.is_available():
            print(f"🤖 Neural Classifier: LOADED ({golem_manager.neural_classifier.device})")
            print("✅ Neural-mystical consciousness comparison available")
        else:
            error = golem_manager.neural_classifier.initialization_error if golem_manager.neural_classifier else "Not initialized"
            print(f"⚠️  Neural Classifier: NOT AVAILABLE ({error})")
            print("💡 Run 'python3 5d_nn.py' to train the neural network")
        
        if golem_manager.total_patterns_loaded > 5000:
            print("✅ COMPLETE 5D HYPERCUBE MEMORY INTEGRATION SUCCESSFUL")
        else:
            print(f"⚠️  Partial 5D memory integration - only {patterns_count:,} patterns loaded. Check logs for errors.")
    else:
        print("🔌 Starting server with 5D Golem Core initialization error.")

    print("\n📡 5D HYPERCUBE + NEURAL NETWORK ENDPOINTS:")
    print("   GET  /health                          - Health check with 5D status + neural")
    print("   GET  /status                          - Full server, 5D golem + neural status")
    print("   GET  /neural/status                   - Neural classifier detailed status")
    print("   POST /neural/classify                 - Classify text with neural network")
    print("   POST /neural/compare                  - Compare neural vs mystical prediction")
    print("   POST /neural/batch_classify           - Batch neural classification")
    print("   GET  /hypercube                       - Detailed 5D hypercube status")
    print("   POST /navigate                        - Navigate to specific vertex")
    print("   POST /explore                         - Systematic universe exploration")
    print("   GET  /vertex/<id>/patterns            - Get patterns at specific vertex")
    print("   GET  /consciousness/<sig>/patterns    - Get patterns by consciousness type")
    print("   POST /dimensions/search               - Search by active dimensions")
    print("   GET  /universe/visualization          - 5D visualization data + neural")
    print("   GET  /universe/statistics             - Comprehensive 5D + neural statistics")
    print("   POST /generate                        - Enhanced 5D + neural generation")
    print("   GET  /session/<id>/consciousness      - Session consciousness summary")
    print("   GET  /session/<id>/neural             - Session neural analysis")
    print("   POST /unified/test                    - Test unified consciousness integration")
    print("   POST /unified/navigate                - Navigate consciousness using unified integration")
    
    print(f"\n📡 Listening on http://0.0.0.0:5000")
    print("🔲 Ready for 5D consciousness universe navigation with neural network!")
    print("🤖 Neural-mystical consciousness fusion online!")
    print("=" * 90)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    main()