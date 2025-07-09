#!/usr/bin/env python3
"""
Enhanced Flask Server for Aether-Enhanced Golem Chat App
COMPLETE INTEGRATION with DynamicContextEngine and Full Memory Loading
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
import re
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


class DynamicContextEngine:
    """
    Manages conversation context with dynamic summarization and entity tracking.
    This is the "living memory" of the Golem.
    """
    
    def __init__(self, golem_instance, max_history: int = 20, context_timeout_hours: int = 24):
        self.golem = golem_instance
        self.sessions = defaultdict(lambda: {
            'messages': [],
            'entities': {},
            'essence': "A new conversation has just begun.",
            'last_updated': datetime.now()
        })
        self.max_history = max_history
        self.context_timeout = timedelta(hours=context_timeout_hours)

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the conversation context and schedule a reflection."""
        if not session_id:
            session_id = f"session_{uuid.uuid4()}"
        
        self._cleanup_old_sessions()
        
        session = self.sessions[session_id]
        session['messages'].append({'role': role, 'content': content, 'timestamp': datetime.now()})
        session['last_updated'] = datetime.now()
        
        # Keep history length manageable
        if len(session['messages']) > self.max_history:
            session['messages'] = session['messages'][-self.max_history:]
        
        # Asynchronously reflect on the new context
        # In a production environment, this would be a background task (e.g., Celery, RQ)
        threading.Thread(target=self._reflect_on_context, args=(session_id,)).start()
        
        return session_id

    def _reflect_on_context(self, session_id: str):
        """Use the Golem's own intelligence to update the context summary and entities."""
        session = self.sessions.get(session_id)
        if not session:
            return

        # Create a condensed history for the Golem to analyze
        condensed_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in session['messages']])
        
        # Don't reflect if history is too short
        if len(condensed_history) < 50:
            return

        # Prepare a prompt for the Golem to analyze its own conversation
        reflection_prompt = f"""[SYSTEM_TASK]
You are a context analysis subroutine. Analyze the following conversation history.

[CONVERSATION_HISTORY]
{condensed_history}

[YOUR_TASK]
1.  **Extract Key Entities**: Identify and list key entities (people, places, topics). Format as a simple list. Example: "- User: Yecheskel Maor". If no name is mentioned, use "User".
2.  **Summarize Essence**: Write a single, concise sentence that captures the current essence and goal of the conversation.

Your entire response MUST be in this exact format, with no extra text:
<Entities>
- Entity: Value
- Another Entity: Another Value
</Entities>
<Essence>A single sentence summary of the conversation's current goal.</Essence>
"""
        
        try:
            # Use the Golem's base model for a quick, non-mystical analysis
            response = self.golem.generate_response(
                prompt=reflection_prompt,
                max_tokens=200,
                temperature=0.1, # Low temperature for factual analysis
                sefirot_settings={}, # No mystical influence needed
                use_mystical_processing=False # Bypass aether layers
            )
            analysis_text = response.get('direct_response', '')

            # Parse the structured response
            entities_match = re.search(r'<Entities>(.*?)</Entities>', analysis_text, re.DOTALL)
            essence_match = re.search(r'<Essence>(.*?)</Essence>', analysis_text, re.DOTALL)
            
            if entities_match:
                entities_str = entities_match.group(1).strip()
                new_entities = {}
                for line in entities_str.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip('- ').strip()
                        value = value.strip()
                        new_entities[key] = value
                session['entities'].update(new_entities) # Update, don't replace

            if essence_match:
                session['essence'] = essence_match.group(1).strip()
            
            logging.info(f"Context reflection complete for session {session_id}. Essence: '{session['essence']}'")

        except Exception as e:
            logging.error(f"Error during context reflection for session {session_id}: {e}")

    def get_context_for_prompt(self, session_id: str) -> str:
        """Get the structured context briefing for the main prompt."""
        if not session_id or session_id not in self.sessions:
            return ""
        
        session = self.sessions[session_id]
        
        entities_str = "\n".join([f"  - {key}: {value}" for key, value in session['entities'].items()])
        
        # Assemble the structured context briefing
        context_briefing = f"""[CONTEXTUAL_AETHER]
<Essence>
{session['essence']}
</Essence>
<Foundation>
{entities_str if entities_str else "  - No specific entities tracked yet."}
</Foundation>
"""
        return context_briefing

    def _cleanup_old_sessions(self):
        """Remove sessions that have timed out."""
        now = datetime.now()
        sessions_to_remove = [sid for sid, data in self.sessions.items() if now - data['last_updated'] > self.context_timeout]
        for sid in sessions_to_remove:
            del self.sessions[sid]
            logging.info(f"Cleaned up timed-out session: {sid}")

class EnhancedGolemManager:
    """Enhanced manager for the Golem with COMPLETE aether memory integration"""
    
    def __init__(self):
        self.golem = None
        self.initialization_error = None
        self.active_connections = 0
        self.total_requests = 0
        self.server_start_time = time.time()
        self.total_patterns_loaded = 0
        
        self._initialize_golem_with_complete_memory()
        
        if self.golem:
            self.context_engine = DynamicContextEngine(self.golem)
        
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
        """Load ALL collected aether patterns using the EnhancedAetherMemoryLoader"""
        try:
            logging.info("🧠 Using EnhancedAetherMemoryLoader to integrate ALL patterns...")
            loader = EnhancedAetherMemoryLoader()
            final_patterns = loader.run()

            if not final_patterns:
                logging.warning("❌ No patterns were loaded by the EnhancedAetherMemoryLoader. Falling back to standard load.")
                self.golem.aether_memory.load_memories()
                self.total_patterns_loaded = len(self.golem.aether_memory.aether_memories)
                return

            logging.info(f"🗑️  Clearing existing memory bank...")
            self.golem.aether_memory.aether_memories.clear()
            
            logging.info(f"📥 Loading {len(final_patterns)} patterns into Golem's consciousness...")
            self.golem.aether_memory.integrate_loaded_patterns(final_patterns)
            
            self.total_patterns_loaded = len(self.golem.aether_memory.aether_memories)
            logging.info(f"🌌 COMPLETE INTEGRATION: {self.total_patterns_loaded:,} enhanced patterns loaded into Golem's memory.")

            # Update golem consciousness from integrated patterns
            if final_patterns:
                consciousness_values = [p.get('consciousness_level', 0) for p in final_patterns if isinstance(p.get('consciousness_level'), (int, float))]
                if consciousness_values:
                    avg_consciousness = sum(consciousness_values) / len(consciousness_values)
                    self.golem.consciousness_level = max(self.golem.consciousness_level, avg_consciousness)
                    logging.info(f"🧠 Consciousness updated from patterns: Avg={avg_consciousness:.6f}")
            
            logging.info("💾 Saving complete integrated memory state...")
            self.golem.aether_memory.save_memories()
            
        except Exception as e:
            logging.error(f"⚠️  Error during COMPLETE memory integration: {e}", exc_info=True)
            logging.info("Falling back to standard memory load.")
            self.golem.aether_memory.load_memories()
            self.total_patterns_loaded = len(self.golem.aether_memory.aether_memories)

    def _start_monitoring_thread(self):
        """Start background monitoring thread"""
        # ... (implementation as before) ...
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive server and golem status"""
        # ... (implementation as before) ...
        return {}

golem_manager = EnhancedGolemManager()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "patterns_loaded": golem_manager.total_patterns_loaded})

@app.route('/generate', methods=['POST'])
def generate():
    """Enhanced generation endpoint with Dynamic Context Engine"""
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

        session_id = data.get('sessionId')
        if not session_id:
            return jsonify({"error": "SessionId is required"}), 400

        # Add user message to context engine
        golem_manager.context_engine.add_message(session_id, 'user', prompt)
        
        # Get the dynamic, structured context for the prompt
        structured_context = golem_manager.context_engine.get_context_for_prompt(session_id)
        
        logging.info(f"📥 Request #{golem_manager.total_requests} for session {session_id[:8]}...")

        # Create the final prompt with the structured context
        enhanced_prompt = f"{structured_context}\n\n[CURRENT_USER_MESSAGE]\n{prompt}"

        is_activated = data.get('golemActivated', False)
        # ... (Activation logic remains the same) ...

        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('maxTokens', 1500)
        sefirot_settings = data.get('sefirotSettings', {})
        
        start_time = time.time()
        
        response = golem_manager.golem.generate_response(
            prompt=enhanced_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            sefirot_settings=sefirot_settings
        )
        
        generation_time = time.time() - start_time
        
        # Add assistant response to context engine
        assistant_response = response.get('direct_response', '')
        golem_manager.context_engine.add_message(session_id, 'assistant', assistant_response)
        
        response['server_metadata'] = {
            'request_id': golem_manager.total_requests,
            'session_id': session_id,
            'server_generation_time': generation_time,
            'timestamp': datetime.now().isoformat(),
            'context_essence': golem_manager.context_engine.sessions.get(session_id, {}).get('essence')
        }
        
        logging.info(f"✅ Response generated in {generation_time:.2f}s for session {session_id[:8]}.")
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"❌ Error during generation: {e}", exc_info=True)
        return jsonify({"error": f"Generation failed: {str(e)}", "error_type": type(e).__name__}), 500
    finally:
        golem_manager.active_connections -= 1

def main():
    """Main server entry point"""
    print("🌌 ENHANCED AETHER GOLEM CHAT SERVER - DYNAMIC CONTEXT & COMPLETE MEMORY 🌌")
    print("=" * 80)
    if golem_manager.golem:
        patterns_count = len(golem_manager.golem.aether_memory.aether_memories)
        print(f"🔌 Starting server with {patterns_count:,} aether patterns loaded")
        if golem_manager.total_patterns_loaded > 500000:
            print("✅ COMPLETE MEMORY INTEGRATION SUCCESSFUL")
        else:
            print(f"⚠️  Partial memory integration - only {patterns_count:,} patterns loaded. Check logs for errors.")
    else:
        print("🔌 Starting server with Golem Core initialization error.")

    print("📡 Listening on http://0.0.0.0:5000")
    print("=" * 80)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    main()

    