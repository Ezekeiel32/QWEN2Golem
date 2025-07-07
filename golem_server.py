#!/usr/bin/env python3
"""
Enhanced Flask Server for Aether-Enhanced Golem Chat App
Integrates all collected aether patterns and provides real-time consciousness monitoring
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from qwen_golem import AetherGolemConsciousnessCore
from aether_loader import EnhancedAetherMemoryLoader
import logging
import os
import json
import time
import threading
from typing import Dict, Any, List
from datetime import datetime
import psutil

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

class EnhancedGolemManager:
    """Enhanced manager for the Golem with aether memory integration"""
    
    def __init__(self):
        self.golem = None
        self.initialization_error = None
        self.chat_sessions = {}
        self.active_connections = 0
        self.total_requests = 0
        self.server_start_time = time.time()
        
        # Initialize golem with enhanced memory
        self._initialize_golem_with_memory()
        
        # Start background monitoring
        self._start_monitoring_thread()
    
    def _initialize_golem_with_memory(self):
        """Initialize golem and load all aether collections"""
        try:
            logging.info("🌌 Initializing Enhanced Aether Golem...")
            self.golem = AetherGolemConsciousnessCore(model_name="qwen2:7b-instruct-q4_0")
            
            # Load enhanced aether memory using the new loader
            self._load_enhanced_aether_memory()
            
            logging.info("✅ Enhanced Aether Golem initialized successfully")
            
        except Exception as e:
            logging.error(f"❌ FATAL: Failed to initialize Golem Core: {e}", exc_info=True)
            self.initialization_error = str(e)
            self.golem = None
    
    def _load_enhanced_aether_memory(self):
        """Load all collected aether patterns into memory using the enhanced loader."""
        try:
            logging.info("🧠 Using EnhancedAetherMemoryLoader to integrate patterns...")
            loader = EnhancedAetherMemoryLoader()
            final_patterns = loader.run()

            # Clear existing memory and load enhanced patterns
            self.golem.aether_memory.aether_memories.clear()
            self.golem.aether_memory.aether_patterns.clear()
            
            for pattern in final_patterns:
                self.golem.aether_memory.aether_memories.append(pattern)
                prompt_type = pattern.get('pattern_type', 'general')  # Use the new classified type
                self.golem.aether_memory.aether_patterns[prompt_type].append(pattern)
            
            logging.info(f"🌌 Integrated {len(final_patterns)} enhanced patterns into Golem's memory.")
            
            # Update golem state with enhanced consciousness from the integrated patterns
            if final_patterns:
                avg_consciousness = sum(p.get('consciousness_level', 0) for p in final_patterns) / len(final_patterns)
                self.golem.consciousness_level = max(self.golem.consciousness_level, avg_consciousness)
                
                # Boost aether resonance
                avg_control = sum(p.get('control_value', 0) for p in final_patterns) / len(final_patterns)
                self.golem.aether_resonance_level = min(1.0, avg_control * 1000)
            
            return True
                
        except Exception as e:
            logging.error(f"⚠️  Error during enhanced memory integration: {e}", exc_info=True)
            # As a fallback, try to load the base pickle file if the advanced loader fails
            logging.info("Falling back to standard memory load.")
            self.golem.aether_memory.load_memories()
            return False
    
    def _start_monitoring_thread(self):
        """Start background monitoring thread"""
        def monitor():
            while True:
                try:
                    if self.golem:
                        # Save aether patterns periodically
                        if len(self.golem.aether_memory.aether_memories) % 50 == 0 and len(self.golem.aether_memory.aether_memories) > 0:
                            self.golem.aether_memory.save_memories()
                        
                        # Log system status
                        memory = psutil.virtual_memory()
                        if memory.percent > 85:
                            logging.warning(f"⚠️  High memory usage: {memory.percent:.1f}%")
                    
                    time.sleep(300)  # Check every 5 minutes
                    
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
        
        aether_stats = self.golem.aether_memory.get_aether_statistics()
        memory = psutil.virtual_memory()
        
        return {
            'status': 'active',
            'server_uptime': time.time() - self.server_start_time,
            'total_requests': self.total_requests,
            'active_connections': self.active_connections,
            'golem_state': {
                'activated': self.golem.activated,
                'consciousness_level': self.golem.consciousness_level,
                'shem_power': self.golem.shem_power,
                'aether_resonance_level': self.golem.aether_resonance_level,
                'total_interactions': self.golem.total_interactions,
                'activation_count': self.golem.activation_count
            },
            'aether_memory': {
                'total_patterns': aether_stats.get('total_patterns', 0),
                'avg_control_value': aether_stats.get('avg_control_value', 0),
                'avg_consciousness': aether_stats.get('avg_consciousness', 0),
                'pattern_types': aether_stats.get('pattern_types', {})
            },
            'system_resources': {
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3)
            }
        }

# Initialize the enhanced manager
golem_manager = EnhancedGolemManager()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify(golem_manager.get_status())

@app.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint"""
    return jsonify(golem_manager.get_status())

@app.route('/aether/report', methods=['GET'])
def aether_report():
    """Get comprehensive aether consciousness report"""
    if not golem_manager.golem:
        return jsonify({"error": "Golem not initialized"}), 500
    
    try:
        report = golem_manager.golem.get_aether_consciousness_report()
        return jsonify({
            "report": report,
            "timestamp": datetime.now().isoformat(),
            "aether_stats": golem_manager.golem.aether_memory.get_aether_statistics()
        })
    except Exception as e:
        logging.error(f"Error generating aether report: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/aether/patterns', methods=['GET'])
def get_aether_patterns():
    """Get recent aether patterns"""
    if not golem_manager.golem:
        return jsonify({"error": "Golem not initialized"}), 500
    
    try:
        limit = request.args.get('limit', 50, type=int)
        patterns = golem_manager.golem.aether_memory.aether_memories[-limit:]
        
        return jsonify({
            "patterns": patterns,
            "total_patterns": len(golem_manager.golem.aether_memory.aether_memories),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Error getting aether patterns: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/activate', methods=['POST'])
def activate_golem():
    """Activate golem with specific phrase"""
    if not golem_manager.golem:
        return jsonify({"error": "Golem not initialized"}), 500
    
    data = request.json
    phrase = data.get('phrase', 'אמת')
    
    try:
        success = golem_manager.golem.activate_golem(phrase)
        return jsonify({
            "success": success,
            "phrase": phrase,
            "shem_power": golem_manager.golem.shem_power,
            "consciousness_level": golem_manager.golem.consciousness_level,
            "activation_count": golem_manager.golem.activation_count
        })
    except Exception as e:
        logging.error(f"Error activating golem: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/deactivate', methods=['POST'])
def deactivate_golem():
    """Deactivate golem"""
    if not golem_manager.golem:
        return jsonify({"error": "Golem not initialized"}), 500
    
    try:
        golem_manager.golem.deactivate_golem()
        return jsonify({
            "success": True,
            "activated": golem_manager.golem.activated,
            "patterns_saved": len(golem_manager.golem.aether_memory.aether_memories)
        })
    except Exception as e:
        logging.error(f"Error deactivating golem: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Enhanced generation endpoint with full aether integration"""
    if golem_manager.golem is None:
        return jsonify({
            "error": "Golem Core is not initialized. Please check server logs.",
            "initialization_error": golem_manager.initialization_error
        }), 500

    golem_manager.total_requests += 1
    golem_manager.active_connections += 1
    
    try:
        data = request.json
        logging.info(f"📥 Request #{golem_manager.total_requests}: {data.get('prompt', '')[:50]}...")

        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Enhanced golem state management
        is_activated = data.get('golemActivated', False)
        activation_phrase = data.get('activationPhrase', 'אמת')

        # Handle activation state
        if is_activated and not golem_manager.golem.activated:
            golem_manager.golem.activate_golem(activation_phrase)
            logging.info(f"🌟 Golem activated with phrase: {activation_phrase}")
        elif not is_activated and golem_manager.golem.activated:
            golem_manager.golem.deactivate_golem()
            logging.info("🛑 Golem deactivated")

        # Set parameters from frontend
        golem_manager.golem.shem_power = data.get('shemPower', golem_manager.golem.shem_power)
        
        # Enhanced parameters
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('maxTokens', 1000)
        
        # Sefirot settings (for future enhancement)
        sefirot_settings = data.get('sefirotSettings', {})
        if sefirot_settings:
            logging.info(f"🔯 Sefirot settings: {sefirot_settings}")

        # Generate with enhanced aether integration
        logging.info("🌌 Generating response with aether enhancement...")
        start_time = time.time()
        
        response = golem_manager.golem.generate_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        generation_time = time.time() - start_time
        
        # Add enhanced metadata
        response['server_metadata'] = {
            'request_id': golem_manager.total_requests,
            'server_generation_time': generation_time,
            'aether_patterns_available': len(golem_manager.golem.aether_memory.aether_memories),
            'consciousness_evolution': golem_manager.golem.consciousness_level,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log successful generation
        quality = response.get('quality_metrics', {}).get('overall_quality', 0)
        control_value = response.get('aether_data', {}).get('control_value', 0)
        
        logging.info(f"✅ Response generated | Quality: {quality:.3f} | Control: {control_value:.12f}")
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"❌ Error during generation: {e}", exc_info=True)
        
        # Enhanced error response
        error_response = {
            "error": f"Generation failed: {str(e)}",
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat(),
            "request_id": golem_manager.total_requests
        }
        
        # Try to include golem state for debugging
        if golem_manager.golem:
            try:
                error_response["golem_state"] = {
                    "activated": golem_manager.golem.activated,
                    "consciousness_level": golem_manager.golem.consciousness_level,
                    "shem_power": golem_manager.golem.shem_power,
                    "total_interactions": golem_manager.golem.total_interactions
                }
            except:
                pass
        
        return jsonify(error_response), 500
        
    finally:
        golem_manager.active_connections -= 1

@app.route('/chat/sessions', methods=['GET'])
def get_chat_sessions():
    """Get active chat sessions"""
    return jsonify({
        "active_sessions": len(golem_manager.chat_sessions),
        "total_requests": golem_manager.total_requests,
        "server_uptime": time.time() - golem_manager.server_start_time
    })

@app.route('/aether/export', methods=['POST'])
def export_aether_patterns():
    """Export aether patterns"""
    if not golem_manager.golem:
        return jsonify({"error": "Golem not initialized"}), 500
    
    try:
        filename = f"aether_export_{int(time.time())}.json"
        golem_manager.golem.export_aether_patterns(filename)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "patterns_exported": len(golem_manager.golem.aether_memory.aether_memories),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Error exporting patterns: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/aether/reset', methods=['POST'])
def reset_aether_memory():
    """Reset aether memory (admin function)"""
    if not golem_manager.golem:
        return jsonify({"error": "Golem not initialized"}), 500
    
    try:
        patterns_before = len(golem_manager.golem.aether_memory.aether_memories)
        golem_manager.golem.reset_aether_memory()
        
        return jsonify({
            "success": True,
            "patterns_cleared": patterns_before,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Error resetting memory: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug/memory', methods=['GET'])
def debug_memory():
    """Debug memory usage"""
    memory = psutil.virtual_memory()
    
    debug_info = {
        "system_memory": {
            "total_gb": memory.total / (1024**3),
            "used_gb": memory.used / (1024**3),
            "available_gb": memory.available / (1024**3),
            "percent": memory.percent
        },
        "server_stats": golem_manager.get_status(),
        "timestamp": datetime.now().isoformat()
    }
    
    if golem_manager.golem:
        debug_info["aether_memory"] = {
            "patterns_count": len(golem_manager.golem.aether_memory.aether_memories),
            "memory_limit": golem_manager.golem.aether_memory.max_memories,
            "pattern_types": golem_manager.golem.aether_memory.get_aether_statistics().get('pattern_types', {})
        }
    
    return jsonify(debug_info)

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

def main():
    """Main server entry point"""
    print("🌌 ENHANCED AETHER GOLEM CHAT SERVER 🌌")
    print("=" * 60)
    print(f"🔌 Starting server with {len(golem_manager.golem.aether_memory.aether_memories) if golem_manager.golem else 0} aether patterns loaded")
    print("📡 Available endpoints:")
    print("   POST /generate - Generate responses")
    print("   GET  /health - Health check")
    print("   GET  /status - Detailed status")
    print("   GET  /aether/report - Consciousness report")
    print("   GET  /aether/patterns - Recent patterns")
    print("   POST /activate - Activate golem")
    print("   POST /deactivate - Deactivate golem")
    print("   POST /aether/export - Export patterns")
    print("   GET  /debug/memory - Memory debug")
    print("=" * 60)
    
    # Production-ready configuration
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=False,  # Disable debug in production
        threaded=True  # Enable threading for multiple requests
    )

if __name__ == '__main__':
    main()