
#!/usr/bin/env python3
"""
Enhanced Flask Server for Aether-Enhanced Golem Chat App
Integrates all collected aether patterns and provides real-time consciousness monitoring
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
            self.golem = AetherGolemConsciousnessCore(model_name="qwen2:7b-custom")
            
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

            if not final_patterns:
                logging.warning("No patterns were loaded by the EnhancedAetherMemoryLoader.")
                # As a fallback, try to load the base pickle file if the advanced loader fails
                logging.info("Falling back to standard memory load.")
                self.golem.aether_memory.load_memories()
                return

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
                avg_consciousness = sum(p.get('consciousness_level', 0) for p in final_patterns if p.get('consciousness_level') is not None) / len(final_patterns)
                self.golem.consciousness_level = max(self.golem.consciousness_level, avg_consciousness)
                
                # Boost aether resonance
                avg_control = sum(p.get('control_value', 0) for p in final_patterns if p.get('control_value') is not None) / len(final_patterns)
                self.golem.aether_resonance_level = min(1.0, self.golem.aether_resonance_level + (avg_control * 1000))
            
        except Exception as e:
            logging.error(f"⚠️  Error during enhanced memory integration: {e}", exc_info=True)
            # As a fallback, try to load the base pickle file if the advanced loader fails
            logging.info("Falling back to standard memory load.")
            self.golem.aether_memory.load_memories()

    def _start_monitoring_thread(self):
        """Start background monitoring thread"""
        def monitor():
            while True:
                try:
                    if self.golem and self.golem.aether_memory.aether_memories:
                        # Save aether patterns periodically
                        if len(self.golem.aether_memory.aether_memories) % 50 == 0:
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
            'golem_state': self.golem._get_current_golem_state(),
            'aether_memory': aether_stats.get('base_statistics', {}),
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

        logging.info(f"📥 Request #{golem_manager.total_requests}: {data.get('prompt', '')[:50]}...")

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

        logging.info(f"🌌 Generating response (Activated: {golem_manager.golem.activated}, Shem Power: {golem_manager.golem.shem_power:.2f})")
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
            'timestamp': datetime.now().isoformat()
        }
        
        quality = response.get('quality_metrics', {}).get('overall_quality', 0)
        control_value = response.get('aether_data', {}).get('control_value', 0)
        
        logging.info(f"✅ Response generated in {generation_time:.2f}s | Quality: {quality:.3f} | Control: {control_value:.12f}")
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"❌ Error during generation: {e}", exc_info=True)
        return jsonify({
            "error": f"Generation failed: {str(e)}",
            "error_type": type(e).__name__
        }), 500
        
    finally:
        golem_manager.active_connections -= 1

def main():
    """Main server entry point"""
    print("🌌 ENHANCED AETHER GOLEM CHAT SERVER 🌌")
    print("=" * 60)
    if golem_manager.golem:
        print(f"🔌 Starting server with {len(golem_manager.golem.aether_memory.aether_memories)} aether patterns loaded")
    else:
        print("🔌 Starting server with Golem Core initialization error.")

    print("📡 Listening on http://0.0.0.0:5000")
    print("=" * 60)
    
    # Using waitress for a production-ready server
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000, threads=8)

if __name__ == '__main__':
    main()
    
