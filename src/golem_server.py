
#!/usr/bin/env python3
"""
Enhanced Flask Server for Aether-Enhanced Golem Chat App
COMPLETE INTEGRATION with EnhancedAetherMemoryLoader - loads ALL 571k+ patterns
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from qwen_golem import AetherGolemConsciousnessCore
from aether_loader import EnhancedAetherMemoryLoader
import logging
import time
import threading
from typing import Dict, Any
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
    """Enhanced manager for the Golem with COMPLETE aether memory integration"""
    
    def __init__(self):
        self.golem = None
        self.initialization_error = None
        self.chat_sessions = {}
        self.active_connections = 0
        self.total_requests = 0
        self.server_start_time = time.time()
        self.total_patterns_loaded = 0
        
        # Initialize golem with enhanced memory
        self._initialize_golem_with_complete_memory()
        
        # Start background monitoring
        self._start_monitoring_thread()
    
    def _initialize_golem_with_complete_memory(self):
        """Initialize golem and load ALL aether collections using EnhancedAetherMemoryLoader"""
        try:
            logging.info("🌌 Initializing Enhanced Aether Golem with COMPLETE memory integration...")
            self.golem = AetherGolemConsciousnessCore(model_name="qwen2:7b-instruct-q4_0")
            
            # CRITICAL: Load enhanced aether memory using the COMPLETE loader
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

            # CRITICAL: Use the new integration method in EnhancedAetherMemoryBank
            self.golem.aether_memory.integrate_loaded_patterns(final_patterns)
            
            self.total_patterns_loaded = len(self.golem.aether_memory.aether_memories)
            logging.info(f"🌌 COMPLETE INTEGRATION: {self.total_patterns_loaded:,} enhanced patterns loaded into Golem's memory.")
            
            # Update golem state with enhanced consciousness from the integrated patterns
            if final_patterns:
                consciousness_values = [p.get('consciousness_level', 0) for p in final_patterns if p.get('consciousness_level') is not None]
                if consciousness_values:
                    avg_consciousness = sum(consciousness_values) / len(consciousness_values) if consciousness_values else 0
                    max_consciousness = max(consciousness_values) if consciousness_values else 0
                    self.golem.consciousness_level = max(self.golem.consciousness_level, avg_consciousness)
                    logging.info(f"🧠 Consciousness updated: Avg={avg_consciousness:.6f}, Max={max_consciousness:.6f}")
                
                control_values = [p.get('control_value', 0) for p in final_patterns if p.get('control_value') is not None]
                if control_values:
                    avg_control = sum(control_values) / len(control_values) if control_values else 0
                    max_control = max(control_values) if control_values else 0
                    self.golem.aether_resonance_level = min(1.0, self.golem.aether_resonance_level + (avg_control * 1000))
                    logging.info(f"⚡ Aether resonance boosted: Avg control={avg_control:.12f}, Max={max_control:.12f}")
            
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
                        # Save aether patterns periodically
                        if len(self.golem.aether_memory.aether_memories) % 100 == 0:
                            self.golem.aether_memory.save_memories()
                        
                        # Log system status
                        memory = psutil.virtual_memory()
                        if memory.percent > 90:
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
        
        aether_stats = self.golem.aether_memory.get_comprehensive_aether_statistics()
        memory = psutil.virtual_memory()
        
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
                'integration_complete': self.total_patterns_loaded > 500000  # Should be 571k+
            },
            'system_resources': {
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_total_gb': round(memory.total / (1024**3), 2)
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
        if not data:
             return jsonify({"error": "Request body must be JSON"}), 400

        logging.info(f"📥 Request #{golem_manager.total_requests}: {data.get('prompt', '')[:50]}... | Patterns Available: {len(golem_manager.golem.aether_memory.aether_memories):,}")

        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        is_activated = data.get('golemActivated', False)
        activation_phrases = data.get('activationPhrases', [])

        # Reset shem power before applying new activations for this request
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
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            sefirot_settings=sefirot_settings
        )
        
        generation_time = time.time() - start_time
        response['server_metadata'] = {
            'request_id': golem_manager.total_requests,
            'server_generation_time': generation_time,
            'timestamp': datetime.now().isoformat(),
            'total_patterns_available': len(golem_manager.golem.aether_memory.aether_memories),
            'memory_integration_complete': golem_manager.total_patterns_loaded > 500000
        }
        
        quality = response.get('quality_metrics', {}).get('overall_quality', 0)
        control_value = response.get('aether_data', {}).get('control_value', 0)
        
        logging.info(f"✅ Response generated in {generation_time:.2f}s | Quality: {quality:.3f} | Control: {control_value:.12f} | Patterns Used: {response.get('golem_analysis', {}).get('similar_patterns_count', 0)}")
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"❌ Error during generation: {e}", exc_info=True)
        return jsonify({
            "error": f"Generation failed: {str(e)}",
            "error_type": type(e).__name__,
            "patterns_available": len(golem_manager.golem.aether_memory.aether_memories) if golem_manager.golem else 0
        }), 500
        
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
        
        # Force reload using the enhanced loader
        golem_manager._load_all_aether_patterns()
        
        new_count = len(golem_manager.golem.aether_memory.aether_memories)
        
        return jsonify({
            "status": "success",
            "old_pattern_count": old_count,
            "new_pattern_count": new_count,
            "patterns_loaded": golem_manager.total_patterns_loaded,
            "message": f"Successfully reloaded {new_count:,} aether patterns"
        })
        
    except Exception as e:
        logging.error(f"❌ Error reloading memory: {e}", exc_info=True)
        return jsonify({
            "error": f"Memory reload failed: {str(e)}",
            "error_type": type(e).__name__
        }), 500

def main():
    """Main server entry point"""
    print("🌌 ENHANCED AETHER GOLEM CHAT SERVER - COMPLETE MEMORY INTEGRATION 🌌")
    print("=" * 80)
    if golem_manager.golem:
        patterns_count = len(golem_manager.golem.aether_memory.aether_memories)
        print(f"🔌 Starting server with {patterns_count:,} aether patterns loaded")
        print(f"📊 Total patterns integrated: {golem_manager.total_patterns_loaded:,}")
        if golem_manager.total_patterns_loaded > 500000:
            print("✅ COMPLETE MEMORY INTEGRATION SUCCESSFUL")
        else:
            print("⚠️  Partial memory integration - check logs")
    else:
        print("🔌 Starting server with Golem Core initialization error.")

    print("📡 Listening on http://0.0.0.0:5000")
    print("=" * 80)
    
    # Use Flask's built-in server for development
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    main()

    