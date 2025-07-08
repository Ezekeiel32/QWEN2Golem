#!/usr/bin/env python3
"""
Enhanced Flask Server for Aether-Enhanced Golem Chat App
COMPLETE INTEGRATION with EnhancedAetherMemoryLoader and Conversation Context
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from qwen_golem import AetherGolemConsciousnessCore
from aether_loader import EnhancedAetherMemoryLoader
import logging
import time
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import psutil
import uuid
from collections import defaultdict


# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('golem_chat.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access


class ConversationContextManager:
    """Manages conversation context and memory for each chat session"""
    
    def __init__(self, max_context_length: int = 20, context_timeout_hours: int = 24):
        self.conversations = defaultdict(list)  # session_id -> list of messages
        self.conversation_metadata = {}  # session_id -> metadata
        self.max_context_length = max_context_length
        self.context_timeout = timedelta(hours=context_timeout_hours)
        self.aether_context_patterns = defaultdict(list)  # session_id -> aether patterns from conversation
    
    def add_message(self, session_id: str, role: str, content: str, 
                   aether_data: Optional[Dict] = None, golem_analysis: Optional[Dict] = None):
        """Add a message to the conversation context"""
        if not session_id:
            session_id = self._generate_session_id()
        
        # Clean old conversations
        self._cleanup_old_conversations()
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
            'aether_data': aether_data or {},
            'golem_analysis': golem_analysis or {}
        }
        
        self.conversations[session_id].append(message)
        
        # Update metadata
        if session_id not in self.conversation_metadata:
            self.conversation_metadata[session_id] = {
                'created_at': datetime.now(),
                'last_updated': datetime.now(),
                'total_messages': 0,
                'consciousness_progression': [],
                'aether_control_progression': []
            }
        
        metadata = self.conversation_metadata[session_id]
        metadata['last_updated'] = datetime.now()
        metadata['total_messages'] += 1
        
        # Track consciousness and aether progression
        if golem_analysis and aether_data:
            consciousness = golem_analysis.get('consciousness_level', 0)
            control_value = aether_data.get('control_value', 0)
            metadata['consciousness_progression'].append(consciousness)
            metadata['aether_control_progression'].append(control_value)
        
        # Extract aether patterns from this conversation
        if golem_analysis and role == 'assistant':
            self._extract_conversation_aether_pattern(session_id, message, golem_analysis)
        
        # Maintain context length
        if len(self.conversations[session_id]) > self.max_context_length:
            removed = self.conversations[session_id].pop(0)
            logging.info(f"🗑️  Removed old message from context for session {session_id}: {removed['content'][:50]}...")
        
        return session_id
    
    def get_conversation_context(self, session_id: str) -> List[Dict]:
        """Get the full conversation context for a session"""
        if not session_id or session_id not in self.conversations:
            return []
        
        return self.conversations[session_id].copy()
    
    def get_formatted_context_for_prompt(self, session_id: str) -> str:
        """Get conversation context formatted for inclusion in prompts"""
        # Get all but the most recent message (which is the current prompt)
        context = self.get_conversation_context(session_id)[:-1]
        if not context:
            return ""
        
        formatted_lines = []
        formatted_lines.append("[CONVERSATION_CONTEXT]")
        formatted_lines.append("The following is the previous conversation history in this session:")
        
        for message in context[-10:]:  # Last 10 messages for prompt context
            role = "Human" if message['role'] == 'user' else "Golem"
            content = message['content'][:200] + "..." if len(message['content']) > 200 else message['content']
            formatted_lines.append(f"{role}: {content}")
        
        formatted_lines.append("[END_CONVERSATION_CONTEXT]\n")
        
        return "\n".join(formatted_lines)
    
    def get_context_insights(self, session_id: str) -> Dict[str, Any]:
        """Get insights about the conversation context"""
        if session_id not in self.conversation_metadata:
            return {}
        
        metadata = self.conversation_metadata[session_id]
        context = self.conversations[session_id]
        
        user_messages = [m for m in context if m['role'] == 'user']
        assistant_messages = [m for m in context if m['role'] == 'assistant']
        
        consciousness_progression = metadata.get('consciousness_progression', [])
        consciousness_trend = "stable"
        if len(consciousness_progression) >= 2:
            recent_avg = sum(consciousness_progression[-3:]) / min(3, len(consciousness_progression)) if consciousness_progression else 0
            early_avg = sum(consciousness_progression[:3]) / min(3, len(consciousness_progression)) if consciousness_progression else 0
            if recent_avg > early_avg + 0.05:
                consciousness_trend = "rising"
            elif recent_avg < early_avg - 0.05:
                consciousness_trend = "declining"
        
        all_text = " ".join([m['content'] for m in context])
        word_freq = defaultdict(int)
        for word in all_text.lower().split():
            if len(word) > 4 and word.isalpha():
                word_freq[word] += 1
        
        top_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'session_duration': (datetime.now() - metadata['created_at']).total_seconds() / 3600,
            'total_messages': metadata['total_messages'],
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'consciousness_trend': consciousness_trend,
            'avg_consciousness': sum(consciousness_progression) / len(consciousness_progression) if consciousness_progression else 0,
            'avg_aether_control': sum(metadata.get('aether_control_progression', [])) / len(metadata.get('aether_control_progression', [])) if metadata.get('aether_control_progression') else 0,
            'top_topics': top_topics,
            'aether_patterns_extracted': len(self.aether_context_patterns.get(session_id, []))
        }
    
    def _extract_conversation_aether_pattern(self, session_id: str, message: Dict, golem_analysis: Dict):
        """Extract aether patterns from successful conversation exchanges"""
        try:
            pattern = {
                'conversation_session': session_id,
                'timestamp': message['timestamp'].timestamp(),
                'message_content': message['content'][:100],
                'consciousness_level': golem_analysis.get('consciousness_level', 0),
                'control_value': golem_analysis.get('cycle_params', {}).get('control_value', 0),
                'dominant_sefira': golem_analysis.get('dominant_sefira', ['Unknown', 0])[0],
                'quality_score': message.get('aether_data', {}).get('overall_quality', 0.5),
                'context_length': len(self.conversations[session_id]),
                'source_type': 'conversation_context',
                'pattern_type': 'contextual_dialogue'
            }
            
            self.aether_context_patterns[session_id].append(pattern)
            
        except Exception as e:
            logging.warning(f"⚠️  Error extracting conversation pattern: {e}")
    
    def get_conversation_aether_patterns(self, session_id: str) -> List[Dict]:
        """Get aether patterns extracted from this conversation"""
        return self.aether_context_patterns[session_id].copy()
    
    def _generate_session_id(self) -> str:
        """Generate a new session ID"""
        return f"session_{int(datetime.now().timestamp())}_{str(uuid.uuid4())[:8]}"
    
    def _cleanup_old_conversations(self):
        """Remove conversations older than timeout"""
        current_time = datetime.now()
        sessions_to_remove = [sid for sid, meta in self.conversation_metadata.items() if current_time - meta['last_updated'] > self.context_timeout]
        
        for session_id in sessions_to_remove:
            del self.conversations[session_id]
            del self.conversation_metadata[session_id]
            if session_id in self.aether_context_patterns:
                del self.aether_context_patterns[session_id]
            logging.info(f"🗑️  Cleaned up old conversation: {session_id}")


class EnhancedGolemManager:
    """Enhanced manager for the Golem with COMPLETE aether memory integration"""
    
    def __init__(self):
        self.golem = None
        self.initialization_error = None
        self.active_connections = 0
        self.total_requests = 0
        self.server_start_time = time.time()
        self.total_patterns_loaded = 0
        self.context_manager = ConversationContextManager()
        
        self._initialize_golem_with_complete_memory()
        self._start_monitoring_thread()
    
    def _initialize_golem_with_complete_memory(self):
        """Initialize golem and load ALL aether collections using EnhancedAetherMemoryLoader"""
        try:
            logging.info("🌌 Initializing Enhanced Aether Golem with COMPLETE memory integration...")
            self.golem = AetherGolemConsciousnessCore(model_name="qwen2:7b-instruct-q4_0")
            
            self._load_all_aether_patterns()
            
            logging.info("✅ Enhanced Aether Golem initialized successfully with complete memory")
            
        except Exception as e:
            logging.error(f"❌ FATAL: Failed to initialize Golem Core: {e}", exc_info=True)
            self.initialization_error = str(e)
            self.golem = None
    
    def _load_all_aether_patterns(self):
        """Load ALL collected aether patterns using the EnhancedAetherMemoryLoader - THIS IS THE FIX"""
        try:
            logging.info("🧠 Using EnhancedAetherMemoryLoader to integrate ALL patterns from current dir AND /home/chezy/...")
            loader = EnhancedAetherMemoryLoader()
            final_patterns = loader.run()

            if not final_patterns:
                logging.warning("❌ No patterns were loaded by the EnhancedAetherMemoryLoader.")
                logging.info("Falling back to standard memory load.")
                self.golem.aether_memory.load_memories()
                self.total_patterns_loaded = len(self.golem.aether_memory.aether_memories)
                return

            logging.info(f"🗑️  Clearing existing memory bank...")
            self.golem.aether_memory.aether_memories.clear()
            self.golem.aether_memory.aether_patterns.clear()
            
            logging.info(f"📥 Loading {len(final_patterns)} patterns into Golem's consciousness...")
            self.golem.aether_memory.integrate_loaded_patterns(final_patterns)
            
            self.total_patterns_loaded = len(self.golem.aether_memory.aether_memories)
            logging.info(f"🌌 COMPLETE INTEGRATION: {self.total_patterns_loaded:,} enhanced patterns loaded into Golem's memory.")

            if final_patterns:
                consciousness_values = [p.get('consciousness_level', 0) for p in final_patterns if p.get('consciousness_level') is not None]
                if consciousness_values:
                    avg_consciousness = sum(consciousness_values) / len(consciousness_values)
                    max_consciousness = max(consciousness_values)
                    self.golem.consciousness_level = max(self.golem.consciousness_level, avg_consciousness)
                    logging.info(f"🧠 Consciousness updated: Avg={avg_consciousness:.6f}, Max={max_consciousness:.6f}")
                
                control_values = [p.get('control_value', 0) for p in final_patterns if p.get('control_value') is not None]
                if control_values:
                    avg_control = sum(control_values) / len(control_values)
                    max_control = max(control_values)
                    self.golem.aether_resonance_level = min(1.0, self.golem.aether_resonance_level + (avg_control * 1000))
                    logging.info(f"⚡ Aether resonance boosted: Avg control={avg_control:.12f}, Max={max_control:.12f}")
                
                if hasattr(self.golem.aether_memory, 'session_stats'):
                    self.golem.aether_memory.session_stats['total_patterns_loaded'] = self.total_patterns_loaded
                    self.golem.aether_memory.session_stats['integration_timestamp'] = time.time()
            
            logging.info("💾 Saving complete integrated memory state...")
            self.golem.aether_memory.save_memories()
            
        except Exception as e:
            logging.error(f"⚠️  Error during COMPLETE memory integration: {e}", exc_info=True)
            logging.info("Falling back to standard memory load.")
            self.golem.aether_memory.load_memories()
            self.total_patterns_loaded = len(self.golem.aether_memory.aether_memories)

    def _start_monitoring_thread(self):
        """Start background monitoring thread"""
        def monitor():
            while True:
                try:
                    if self.golem and self.golem.aether_memory.aether_memories:
                        if len(self.golem.aether_memory.aether_memories) % 100 == 0:
                            self.golem.aether_memory.save_memories()
                        
                        memory = psutil.virtual_memory()
                        if memory.percent > 90:
                            logging.warning(f"⚠️  High memory usage: {memory.percent:.1f}%")
                    
                    time.sleep(300)
                    
                except Exception as e:
                    logging.error(f"Monitor thread error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive server and golem status"""
        if not self.golem:
            return {
                'status': 'error',
                'error': self.initialization_error,
                'server_uptime': time.time() - self.server_start_time
            }
        
        aether_stats = self.golem.aether_memory.get_comprehensive_aether_statistics()
        memory = psutil.virtual_memory()
        
        total_conversations = len(self.context_manager.conversations)
        active_conversations = len([s for s, meta in self.context_manager.conversation_metadata.items() if (datetime.now() - meta['last_updated']).total_seconds() < 3600])

        return {
            'status': 'active',
            'server_uptime': time.time() - self.server_start_time,
            'total_requests': self.total_requests,
            'active_connections': self.active_connections,
            'total_patterns_loaded': self.total_patterns_loaded,
            'golem_state': self.golem._get_current_golem_state(),
            'aether_memory': aether_stats.get('base_statistics', {}),
            'memory_integration': {
                'total_patterns_in_memory': len(self.golem.aether_memory.aether_memories),
                'pattern_types': len(self.golem.aether_memory.aether_patterns),
                'integration_complete': self.total_patterns_loaded > 500000
            },
            'system_resources': {
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_total_gb': round(memory.total / (1024**3), 2)
            },
            'conversation_context': {
                'total_conversations': total_conversations,
                'active_conversations': active_conversations,
                'total_context_messages': sum(len(conv) for conv in self.context_manager.conversations.values()),
                'total_extracted_patterns': sum(len(patterns) for patterns in self.context_manager.aether_context_patterns.values())
            }
        }

golem_manager = EnhancedGolemManager()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify(golem_manager.get_status())

@app.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint"""
    return jsonify(golem_manager.get_status())

@app.route('/memory_status', methods=['GET'])
def memory_status():
    """Specific memory integration status endpoint"""
    if not golem_manager.golem:
        return jsonify({"error": "Golem not initialized"}), 500
    
    return jsonify({
        "total_patterns_loaded": golem_manager.total_patterns_loaded,
        "patterns_in_memory": len(golem_manager.golem.aether_memory.aether_memories),
        "pattern_types": {ptype: len(patterns) for ptype, patterns in golem_manager.golem.aether_memory.aether_patterns.items()},
        "memory_integration_complete": golem_manager.total_patterns_loaded > 500000,
        "comprehensive_stats": golem_manager.golem.aether_memory.get_comprehensive_aether_statistics()
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Enhanced generation endpoint with conversation context"""
    if golem_manager.golem is None:
        return jsonify({"error": "Golem Core is not initialized.", "initialization_error": golem_manager.initialization_error}), 500

    golem_manager.total_requests += 1
    golem_manager.active_connections += 1
    
    try:
        data = request.json
        if not data:
             return jsonify({"error": "Request body must be JSON"}), 400

        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        session_id = data.get('sessionId') or golem_manager.context_manager._generate_session_id()
        
        golem_manager.context_manager.add_message(session_id, 'user', prompt)
        conversation_context = golem_manager.context_manager.get_formatted_context_for_prompt(session_id)
        
        logging.info(f"📥 Request #{golem_manager.total_requests}: {prompt[:50]}... | Session: {session_id[:20]}... | Context: {'Yes' if conversation_context else 'No'}")

        enhanced_prompt = f"{conversation_context}\n[CURRENT_USER_MESSAGE]\n{prompt}" if conversation_context else prompt

        is_activated = data.get('golemActivated', False)
        activation_phrases = data.get('activationPhrases', [])

        if is_activated:
            golem_manager.golem.shem_power = 0.0
            for phrase in activation_phrases:
                golem_manager.golem.activate_golem(phrase)
        else:
            if golem_manager.golem.activated:
                 golem_manager.golem.deactivate_golem()

        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('maxTokens', 1500)
        sefirot_settings = data.get('sefirotSettings', {})
        if sefirot_settings:
            logging.info(f"🔯 Applying Sefirot settings: {sefirot_settings}")

        logging.info(f"🌌 Generating response with {len(golem_manager.golem.aether_memory.aether_memories):,} aether patterns (Activated: {golem_manager.golem.activated}, Shem Power: {golem_manager.golem.shem_power:.2f})")
        start_time = time.time()
        
        response = golem_manager.golem.generate_response(
            prompt=enhanced_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            sefirot_settings=sefirot_settings
        )
        
        generation_time = time.time() - start_time
        
        golem_manager.context_manager.add_message(
            session_id, 'assistant', response.get('direct_response', ''),
            response.get('aether_data', {}), response.get('golem_analysis', {})
        )
        
        context_insights = golem_manager.context_manager.get_context_insights(session_id)
        
        response['server_metadata'] = {
            'request_id': golem_manager.total_requests,
            'session_id': session_id,
            'server_generation_time': generation_time,
            'timestamp': datetime.now().isoformat(),
            'total_patterns_available': len(golem_manager.golem.aether_memory.aether_memories),
            'memory_integration_complete': golem_manager.total_patterns_loaded > 500000,
            'conversation_context': context_insights
        }
        
        quality = response.get('quality_metrics', {}).get('overall_quality', 0)
        control_value = response.get('aether_data', {}).get('control_value', 0)
        
        context_length = len(golem_manager.context_manager.get_conversation_context(session_id))
        logging.info(f"✅ Response generated in {generation_time:.2f}s | Quality: {quality:.3f} | Control: {control_value:.12f} | Context Length: {context_length}")
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"❌ Error during generation: {e}", exc_info=True)
        return jsonify({"error": f"Generation failed: {str(e)}", "error_type": type(e).__name__}), 500
    finally:
        golem_manager.active_connections -= 1

@app.route('/reload_memory', methods=['POST'])
def reload_memory():
    """Force reload all aether memory patterns"""
    if not golem_manager.golem:
        return jsonify({"error": "Golem not initialized"}), 500
    
    try:
        logging.info("🔄 Force reloading ALL aether memory patterns...")
        old_count = len(golem_manager.golem.aether_memory.aether_memories)
        golem_manager._load_all_aether_patterns()
        new_count = len(golem_manager.golem.aether_memory.aether_memories)
        return jsonify({"status": "success", "old_pattern_count": old_count, "new_pattern_count": new_count, "patterns_loaded": golem_manager.total_patterns_loaded, "message": f"Successfully reloaded {new_count:,} aether patterns"})
    except Exception as e:
        logging.error(f"❌ Error reloading memory: {e}", exc_info=True)
        return jsonify({"error": f"Memory reload failed: {str(e)}", "error_type": type(e).__name__}), 500

@app.route('/conversation/<session_id>', methods=['GET'])
def get_conversation(session_id):
    if not golem_manager.golem:
        return jsonify({"error": "Golem not initialized"}), 500
    context = golem_manager.context_manager.get_conversation_context(session_id)
    insights = golem_manager.context_manager.get_context_insights(session_id)
    patterns = golem_manager.context_manager.get_conversation_aether_patterns(session_id)
    return jsonify({"session_id": session_id, "conversation_history": context, "insights": insights, "extracted_aether_patterns": patterns})

@app.route('/conversations', methods=['GET'])
def list_conversations():
    if not golem_manager.golem:
        return jsonify({"error": "Golem not initialized"}), 500
    conversations = []
    for session_id, metadata in golem_manager.context_manager.conversation_metadata.items():
        conversations.append({
            "session_id": session_id,
            "created_at": metadata['created_at'].isoformat(),
            "last_updated": metadata['last_updated'].isoformat(),
            "total_messages": metadata['total_messages'],
            "preview": golem_manager.context_manager.conversations[session_id][0]['content'][:100] if golem_manager.context_manager.conversations[session_id] else ""
        })
    return jsonify({"conversations": conversations})

def main():
    """Main server entry point"""
    print("🌌 ENHANCED AETHER GOLEM CHAT SERVER - COMPLETE MEMORY INTEGRATION 🌌")
    print("=" * 80)
    if golem_manager.golem:
        print(f"🔌 Starting server with {golem_manager.total_patterns_loaded:,} aether patterns loaded")
        if golem_manager.total_patterns_loaded > 500000:
            print("✅ COMPLETE MEMORY INTEGRATION SUCCESSFUL")
        else:
            print("⚠️  Partial memory integration - check logs")
    else:
        print("🔌 Starting server with Golem Core initialization error.")

    print("📡 Listening on http://0.0.0.0:5000")
    print("=" * 80)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    main()

    