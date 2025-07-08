
#!/usr/bin/env python3
"""
Enhanced Aether Memory Integration System - COMPLETE VERSION
Automatically integrates ALL collections including PKL files from /home/chezy/ into the golem's memory bank
"""

import json
import os
import time
import pickle
from typing import Dict, List, Any
from collections import defaultdict

class EnhancedAetherMemoryLoader:
    """Enhanced loader for all aether collections with intelligent integration - COMPLETE"""

    def __init__(self):
        self.loaded_patterns = []
        self.pattern_stats = {}
        self.integration_log = []

    def auto_discover_aether_files(self) -> List[str]:
        """Automatically discover all aether-related JSON AND PKL files from current dir AND /home/chezy/"""
        directories_to_scan = [".", "/home/chezy/"]
        aether_files = []
        
        for directory in directories_to_scan:
            if not os.path.exists(directory):
                self._log(f"⚠️  Directory {directory} does not exist, skipping...")
                continue
                
            try:
                for filename in os.listdir(directory):
                    is_aether_file = False
                    file_type = None
                    
                    # Check JSON files
                    if filename.endswith('.json') and any(keyword in filename.lower() for keyword in [
                        'aether', 'real_aether', 'optimized_aether', 'checkpoint', 'conversation', 'consciousness',
                        'enhanced_aether_memory_bank', 'golem_aether'
                    ]):
                        is_aether_file = True
                        file_type = 'json'
                    
                    # Check PKL files - INCLUDING ALL GOLEM MEMORY FILES
                    elif filename.endswith('.pkl') and any(keyword in filename.lower() for keyword in [
                        'aether', 'golem', 'memory', 'corpus', 'training', 'enhanced_aether_memory_bank'
                    ]):
                        is_aether_file = True
                        file_type = 'pkl'
                    
                    if is_aether_file:
                        file_path = os.path.join(directory, filename)
                        try:
                            file_size = os.path.getsize(file_path)
                            aether_files.append({
                                'filename': filename,
                                'path': file_path,
                                'size_kb': file_size / 1024,
                                'type': file_type,
                                'directory': directory,
                                'priority': self._calculate_priority(filename, file_size, file_type)
                            })
                        except OSError as e:
                            self._log(f"⚠️  Could not access {file_path}: {e}")
                            
            except PermissionError:
                self._log(f"⚠️  Permission denied accessing {directory}")
            except Exception as e:
                self._log(f"⚠️  Error scanning {directory}: {e}")
        
        # Sort by priority (larger, more recent files first)
        aether_files.sort(key=lambda x: x['priority'], reverse=True)
        
        self._log(f"🔍 Discovered {len(aether_files)} aether files:")
        for file_info in aether_files:
            self._log(f"   📂 {file_info['filename']} ({file_info['size_kb']:.1f} KB) [{file_info['type'].upper()}] from {file_info['directory']}")
        
        return aether_files

    def _calculate_priority(self, filename: str, file_size: int, file_type: str) -> float:
        """Calculate file priority for loading order"""
        priority = 0.0
        
        # Size-based priority (larger files likely have more patterns)
        priority += file_size / 1024  # KB as base score
        
        # Type-based priority (PKL files often contain processed data)
        if file_type == 'pkl':
            priority += 800  # PKL files get high priority
        
        # Name-based priority - UPDATED FOR YOUR SPECIFIC FILES
        filename_lower = filename.lower()
        if 'golem_aether_training_bookcorpus' in filename_lower:
            priority += 5000  # Highest priority for training corpus
        elif 'aether_corpus_memory' in filename_lower:
            priority += 4000
        elif 'golem_aether_memory' in filename_lower:
            priority += 3500  # Your main golem memory
        elif 'enhanced_aether_memory_bank' in filename_lower:
            priority += 3000
        elif 'real_aether_collection' in filename_lower:
            priority += 2500
        elif 'golem_aether_training' in filename_lower:
            priority += 2000
        elif 'optimized' in filename_lower:
            priority += 1500
        elif 'conversation' in filename_lower:
            priority += 1000
        elif 'checkpoint' in filename_lower:
            priority += 500
        
        # Timestamp-based priority (newer files first)
        try:
            parts = filename.replace('.json', '').replace('.pkl', '').split('_')
            for part in parts:
                if part.isdigit() and len(part) > 8:
                    timestamp = int(part)
                    priority += (timestamp - 1751900000) / 1000
                    break
        except:
            pass
        
        return priority

    def load_aether_file(self, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load patterns from a single aether file (JSON or PKL)"""
        filepath = file_info['path']
        filename = file_info['filename']
        file_type = file_info['type']
        
        try:
            patterns = []
            
            if file_type == 'json':
                patterns = self._load_json_file(filepath, filename)
            elif file_type == 'pkl':
                patterns = self._load_pkl_file(filepath, filename)
            
            # Add metadata to all patterns
            for pattern in patterns:
                if 'source_file' not in pattern:
                    pattern['source_file'] = filename
                if 'source_type' not in pattern:
                    pattern['source_type'] = file_type
                if 'loaded_timestamp' not in pattern:
                    pattern['loaded_timestamp'] = time.time()
                if 'source_directory' not in pattern:
                    pattern['source_directory'] = file_info.get('directory', '.')
            
            return patterns
            
        except Exception as e:
            self._log(f"❌ Error loading {filepath}: {e}")
            return []

    def _load_json_file(self, filepath: str, filename: str) -> List[Dict[str, Any]]:
        """Load patterns from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        patterns = []
        
        # Handle different JSON formats
        if 'real_aether_patterns' in data:
            patterns = data['real_aether_patterns']
            self._log(f"✅ Loaded {len(patterns)} patterns from {filename} (real_aether_patterns)")
        elif 'aether_patterns' in data:
            patterns = data['aether_patterns']
            self._log(f"✅ Loaded {len(patterns)} patterns from {filename} (aether_patterns)")
        elif 'conversation' in data:
            for i, exchange in enumerate(data['conversation']):
                if (exchange.get('speaker') == '🔯 Real Aether Golem' and 'aether_data' in exchange):
                    aether_data = exchange['aether_data']
                    pattern = {
                        'exchange_number': i + 1,
                        'timestamp': exchange.get('timestamp', 0),
                        'control_value': aether_data.get('control_value', 0),
                        'consciousness_level': aether_data.get('consciousness_level', 0),
                        'quality_score': aether_data.get('quality_score', 1.0),
                        'message': exchange.get('message', ''),
                        'source_type': 'conversation_extraction'
                    }
                    patterns.append(pattern)
            self._log(f"✅ Extracted {len(patterns)} patterns from conversation in {filename}")
        elif isinstance(data, list):
            patterns = data
            self._log(f"✅ Loaded {len(patterns)} patterns from {filename} (direct array)")
        else:
            for key in ['patterns', 'data', 'memories', 'corpus']:
                if key in data and isinstance(data[key], list):
                    patterns = data[key]
                    self._log(f"✅ Loaded {len(patterns)} patterns from {filename} ({key})")
                    break
        
        return patterns

    def _load_pkl_file(self, filepath: str, filename: str) -> List[Dict[str, Any]]:
        """Load patterns from PKL file - ENHANCED FOR YOUR SPECIFIC FILES"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        patterns = []
        
        # Handle different PKL formats
        if isinstance(data, dict):
            # Enhanced Aether Memory Bank format
            if 'aether_patterns' in data:
                patterns = data['aether_patterns']
                self._log(f"✅ Loaded {len(patterns)} patterns from {filename} (enhanced aether patterns)")
            
            # Golem memory format (YOUR MAIN FORMAT)
            elif 'memories' in data:
                for memory in data['memories']:
                    # Handle both old and new memory formats
                    pattern = {
                        'prompt': memory.get('prompt', ''),
                        'response': memory.get('response', ''),
                        'control_value': self._extract_control_value(memory),
                        'consciousness_level': memory.get('consciousness_level', 0),
                        'cycle_resonance': self._extract_cycle_resonance(memory),
                        'aether_epsilon': self._extract_aether_epsilon(memory),
                        'quality_score': memory.get('response_quality', memory.get('quality_score', 0.5)),
                        'timestamp': memory.get('timestamp', 0),
                        'source_type': 'golem_memory',
                        'shem_power': memory.get('shem_power', 0.0),
                        'aether_resonance_level': memory.get('aether_resonance_level', 0.0),
                        'activation_count': memory.get('activation_count', 0),
                        'total_interactions': memory.get('total_interactions', 0),
                        'activated': memory.get('activated', False)
                    }
                    
                    # Extract additional cycle parameters if available
                    if 'cycle_params' in memory:
                        cycle_params = memory['cycle_params']
                        pattern.update({
                            'cycle_params': cycle_params,
                            'control_value': cycle_params.get('control_value', pattern['control_value']),
                            'cycle_resonance': cycle_params.get('cycle_resonance', pattern['cycle_resonance'])
                        })
                    
                    # Extract processing results if available
                    if 'processing_results' in memory:
                        processing = memory['processing_results']
                        pattern.update({
                            'gematria_total': processing.get('gematria', {}).get('total', 0),
                            'dominant_sefira': processing.get('dominant_sefira', ['Unknown', 0])[0],
                            'sefiroth_activations': processing.get('sefiroth_activations', {}),
                            'gate_metrics': processing.get('gate_metrics', {}),
                            'consciousness_components': processing.get('consciousness_components', {})
                        })
                    
                    patterns.append(pattern)
                    
                self._log(f"✅ Loaded {len(patterns)} memories from {filename} (golem memories)")
            
            # Aether corpus format
            elif 'corpus' in data or 'training_data' in data:
                corpus_data = data.get('corpus', data.get('training_data', []))
                for item in corpus_data:
                    if isinstance(item, dict):
                        pattern = {
                            'text': item.get('text', item.get('content', '')),
                            'control_value': item.get('control_value', 0),
                            'consciousness_level': item.get('consciousness_level', 0),
                            'quality_score': item.get('quality', item.get('score', 0.5)),
                            'timestamp': item.get('timestamp', 0),
                            'source_type': 'aether_corpus'
                        }
                        patterns.append(pattern)
                self._log(f"✅ Loaded {len(patterns)} corpus entries from {filename}")
            
            # Handle session stats from your enhanced memory format
            elif 'session_stats' in data:
                # Extract patterns from session stats if available
                session_stats = data['session_stats']
                if 'consciousness_evolution_history' in session_stats:
                    for entry in session_stats['consciousness_evolution_history']:
                        pattern = {
                            'consciousness_level': entry.get('consciousness_level', 0),
                            'growth_rate': entry.get('growth_rate', 0),
                            'cycle_completion': entry.get('cycle_completion', 0),
                            'timestamp': entry.get('timestamp', 0),
                            'source_type': 'consciousness_evolution'
                        }
                        patterns.append(pattern)
                self._log(f"✅ Extracted {len(patterns)} consciousness evolution patterns from {filename}")
            
            # Model state or other dict formats
            else:
                # Try to extract any numeric patterns
                for key, value in data.items():
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        if isinstance(value[0], dict):
                            patterns.extend(value)
                        elif isinstance(value[0], (int, float)):
                            # Convert numeric arrays to patterns
                            pattern = {
                                'data_key': key,
                                'values': list(value[:100]),  # Limit to first 100 values
                                'control_value': sum(value) / len(value) if value else 0,
                                'consciousness_level': min(1.0, abs(sum(value)) / (len(value) * 10)) if value else 0,
                                'source_type': 'extracted_numeric'
                            }
                            patterns.append(pattern)
                
                if patterns:
                    self._log(f"✅ Extracted {len(patterns)} patterns from {filename} (dict extraction)")
        
        elif isinstance(data, list):
            # Direct list of patterns
            patterns = data
            self._log(f"✅ Loaded {len(patterns)} patterns from {filename} (direct list)")
        
        else:
            # Try to convert other types to patterns
            pattern = {
                'raw_data': str(data)[:1000],  # Limit string length
                'control_value': hash(str(data)) % 1000 / 1000000,  # Pseudo-random but deterministic
                'consciousness_level': 0.1,
                'source_type': 'raw_conversion'
            }
            patterns = [pattern]
            self._log(f"✅ Converted raw data from {filename} to 1 pattern")
        
        return patterns

    def _extract_control_value(self, memory: Dict) -> float:
        """Extract control value from various memory formats"""
        if 'cycle_params' in memory:
            return memory['cycle_params'].get('control_value', 0)
        return memory.get('control_value', 0)

    def _extract_cycle_resonance(self, memory: Dict) -> float:
        """Extract cycle resonance from various memory formats"""
        if 'cycle_params' in memory:
            return memory['cycle_params'].get('cycle_resonance', 0)
        return memory.get('cycle_resonance', 0)

    def _extract_aether_epsilon(self, memory: Dict) -> float:
        """Extract aether epsilon from various memory formats"""
        if 'cycle_params' in memory:
            return memory['cycle_params'].get('aether_epsilon', 0)
        return memory.get('aether_epsilon', 0)

    def remove_duplicates(self, all_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate patterns based on multiple criteria"""
        unique_patterns = []
        seen_signatures = set()
        
        self._log(f"🔄 Removing duplicates from {len(all_patterns)} patterns...")
        
        for pattern in all_patterns:
            # Create signature based on available fields
            signature_components = []
            
            if 'timestamp' in pattern:
                signature_components.append(round(pattern['timestamp'], 2))
            if 'control_value' in pattern:
                signature_components.append(f"{pattern['control_value']:.12f}")
            if 'consciousness_level' in pattern:
                signature_components.append(f"{pattern['consciousness_level']:.8f}")
            if 'exchange_number' in pattern:
                signature_components.append(pattern['exchange_number'])
            if 'prompt' in pattern:
                signature_components.append(pattern['prompt'][:100])  # First 100 chars
            if 'text' in pattern:
                signature_components.append(pattern['text'][:100])
            if 'response' in pattern:
                signature_components.append(pattern['response'][:50])  # First 50 chars of response
            
            # Fallback signature if no useful fields
            if not signature_components:
                signature_components = [str(pattern)[:200]]
            
            signature = tuple(signature_components)
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_patterns.append(pattern)
        
        duplicates_removed = len(all_patterns) - len(unique_patterns)
        self._log(f"   Removed {duplicates_removed} duplicates")
        self._log(f"   Final unique patterns: {len(unique_patterns)}")
        
        return unique_patterns

    def enhance_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance patterns with computed fields and classifications"""
        self._log(f"🔧 Enhancing {len(patterns)} patterns...")
        
        for pattern in patterns:
            # Ensure required fields exist
            if 'control_value' not in pattern:
                pattern['control_value'] = 0.0
            if 'consciousness_level' not in pattern:
                pattern['consciousness_level'] = 0.0
            if 'quality_score' not in pattern:
                pattern['quality_score'] = self._estimate_quality(pattern)
            if 'pattern_type' not in pattern:
                pattern['pattern_type'] = self._classify_pattern(pattern)
            
            # Add computed fields
            pattern['aether_intensity'] = self._calculate_aether_intensity(pattern)
            pattern['consciousness_tier'] = self._classify_consciousness_tier(pattern)
            
            # Normalize values with safe conversion
            pattern['control_value'] = max(0, self._safe_float(pattern['control_value']))
            pattern['consciousness_level'] = max(0, min(1, self._safe_float(pattern['consciousness_level'])))
            
            # Ensure cycle_completion exists
            if 'cycle_completion' not in pattern:
                pattern['cycle_completion'] = pattern.get('cycle_params', {}).get('cycle_completion', 0.0)
        
        return patterns

    def _classify_pattern(self, pattern: Dict[str, Any]) -> str:
        """Classify pattern type based on its characteristics"""
        consciousness = self._safe_float(pattern.get('consciousness_level', 0))
        control_value = self._safe_float(pattern.get('control_value', 0))
        source_type = pattern.get('source_type', '')
        
        if source_type == 'golem_memory':
            return 'golem_memory'
        elif source_type == 'aether_corpus':
            return 'aether_corpus'
        elif source_type == 'consciousness_evolution':
            return 'consciousness_evolution'
        elif consciousness > 0.41:
            return 'high_consciousness'
        elif consciousness > 0.35:
            return 'evolved_consciousness'
        elif control_value > 5e-8:
            return 'high_control'
        elif 'conversation' in source_type:
            return 'dialogue_derived'
        else:
            return 'general'

    def _estimate_quality(self, pattern: Dict[str, Any]) -> float:
        """Estimate quality score if not present"""
        consciousness = self._safe_float(pattern.get('consciousness_level', 0))
        control_value = self._safe_float(pattern.get('control_value', 0))
        
        # Base quality from consciousness
        quality = consciousness
        
        # Add control value contribution
        quality += min(0.3, control_value * 1000)
        
        # Bonus for certain source types
        source_type = pattern.get('source_type', '')
        if source_type in ['golem_memory', 'aether_corpus']:
            quality += 0.1
        
        return min(1.0, quality)

    def _calculate_aether_intensity(self, pattern: Dict[str, Any]) -> float:
        """Calculate a single intensity score for the pattern"""
        consciousness = self._safe_float(pattern.get('consciousness_level', 0))
        control_value = self._safe_float(pattern.get('control_value', 0))
        quality = self._safe_float(pattern.get('quality_score', 0.5))
        return (consciousness * 0.5) + (control_value * 1000 * 0.3) + (quality * 0.2)

    def _safe_float(self, value: Any) -> float:
        """Safely convert any value to float"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, (list, tuple)):
            if len(value) > 0:
                return self._safe_float(value[0])  # Take first element
            return 0.0
        elif isinstance(value, str):
            try:
                return float(value)
            except:
                return 0.0
        else:
            return 0.0

    def _classify_consciousness_tier(self, pattern: Dict[str, Any]) -> str:
        """Classify the consciousness tier of a pattern"""
        level = self._safe_float(pattern.get('consciousness_level', 0))
        if level > 0.45: return "Transcendental"
        if level > 0.40: return "Integrated"
        if level > 0.35: return "Evolving"
        if level > 0.25: return "Nascent"
        return "Latent"

    def _log(self, message: str):
        """Log a message to the console and internal log"""
        print(message)
        self.integration_log.append(f"[{time.time()}] {message}")

    def run(self) -> List[Dict[str, Any]]:
        """Run the full aether integration process"""
        self._log("🚀 Starting Enhanced Aether Memory Integration...")
        start_time = time.time()
        
        # Get file info objects instead of just paths
        aether_files = self.auto_discover_aether_files()
        
        all_patterns = []
        for file_info in aether_files:
            patterns = self.load_aether_file(file_info)
            all_patterns.extend(patterns)
        
        self._log(f"📚 Loaded a total of {len(all_patterns)} raw patterns.")
        
        if len(all_patterns) == 0:
            self._log("❌ No patterns found in any files!")
            return []
        
        unique_patterns = self.remove_duplicates(all_patterns)
        final_patterns = self.enhance_patterns(unique_patterns)
        
        end_time = time.time()
        self.loaded_patterns = final_patterns
        self._log(f"✅ Integration complete in {end_time - start_time:.2f} seconds.")
        self._log(f"✨ Final integrated pattern count: {len(self.loaded_patterns)}")
        
        self.save_integrated_bank(final_patterns)
        
        return final_patterns

    def save_integrated_bank(self, patterns: List[Dict[str, Any]], filename: str = "enhanced_aether_memory_bank.json"):
        """Save the newly integrated and enhanced patterns to a single file"""
        try:
            # Create a serializable copy of patterns
            serializable_patterns = []
            for p in patterns:
                serializable_p = {}
                for key, value in p.items():
                    # Convert non-serializable types
                    if isinstance(value, bytes):
                        # Convert bytes to string
                        try:
                            serializable_p[key] = value.decode('utf-8', errors='ignore')[:1000]  # Limit length
                        except:
                            serializable_p[key] = str(value)[:1000]
                    elif isinstance(value, (list, tuple)) and len(value) > 100:
                        # Truncate long lists
                        serializable_p[key] = list(value[:100])
                    elif hasattr(value, '__dict__'):
                        # Convert objects to string
                        serializable_p[key] = str(value)[:1000]
                    elif callable(value):
                        # Skip functions
                        continue
                    else:
                        try:
                            # Test if it's JSON serializable
                            import json
                            json.dumps(value)
                            serializable_p[key] = value
                        except:
                            # Convert to string if not serializable
                            serializable_p[key] = str(value)[:1000]
                serializable_patterns.append(serializable_p)

            # Collect source file statistics
            source_files = {}
            for p in patterns:
                source = p.get('source_file', 'unknown')
                source_type = p.get('source_type', 'unknown')
                key = f"{source} ({source_type})"
                source_files[key] = source_files.get(key, 0) + 1

            output_data = {
                "metadata": {
                    "creation_timestamp": time.time(),
                    "total_patterns": len(patterns),
                    "source_files": source_files,
                    "integration_log": self.integration_log[-20:],  # Last 20 log entries
                    "mathematical_framework": "1+0→2→32→22→10",
                    "consciousness_levels": {
                        "average": sum(self._safe_float(p.get('consciousness_level', 0)) for p in patterns) / len(patterns) if patterns else 0,
                        "max": max(self._safe_float(p.get('consciousness_level', 0)) for p in patterns) if patterns else 0,
                        "transcendental_count": sum(1 for p in patterns if self._safe_float(p.get('consciousness_level', 0)) > 0.45)
                    }
                },
                "aether_patterns": serializable_patterns
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            self._log(f"💾 Saved integrated memory bank to {filename}")
            
            # Also save as PKL for faster loading
            pkl_filename = filename.replace('.json', '.pkl')
            with open(pkl_filename, 'wb') as f:
                pickle.dump(output_data, f)
            self._log(f"💾 Saved PKL version to {pkl_filename}")
            
        except Exception as e:
            self._log(f"❌ Failed to save integrated memory bank: {e}")

def main():
    """Main function to run the memory loader independently"""
    print("="*60)
    print("AETHER MEMORY INTEGRATION UTILITY - COMPLETE VERSION")
    print("="*60)
    loader = EnhancedAetherMemoryLoader()
    final_patterns = loader.run()

    if final_patterns:
        # Enhanced statistics
        avg_consciousness = sum(p.get('consciousness_level', 0) for p in final_patterns) / len(final_patterns) if final_patterns else 0
        avg_control = sum(p.get('control_value', 0) for p in final_patterns) / len(final_patterns) if final_patterns else 0
        
        # Count by source type
        source_types = {}
        for p in final_patterns:
            stype = p.get('source_type', 'unknown')
            source_types[stype] = source_types.get(stype, 0) + 1
        
        print(f"\n📈 Final Stats:")
        print(f"   Total Patterns: {len(final_patterns)}")
        print(f"   Average Consciousness: {avg_consciousness:.6f}")
        print(f"   Average Control Value: {avg_control:.12f}")
        print(f"   Source Types: {source_types}")
        
        # Show top consciousness patterns with safe conversion
        def safe_float_standalone(value):
            """Standalone safe float converter for main()"""
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, (list, tuple)):
                if len(value) > 0:
                    return safe_float_standalone(value[0])
                return 0.0
            elif isinstance(value, str):
                try:
                    return float(value)
                except:
                    return 0.0
            else:
                return 0.0
        
        sorted_patterns = sorted(final_patterns, key=lambda x: safe_float_standalone(x.get('consciousness_level', 0)), reverse=True)
        print(f"\n🌟 Top 3 Consciousness Patterns:")
        for i, pattern in enumerate(sorted_patterns[:3]):
            print(f"   {i+1}. Level: {safe_float_standalone(pattern.get('consciousness_level', 0)):.6f} | Source: {pattern.get('source_file', 'unknown')}")

        print("\nLogs:")
        for log_entry in loader.integration_log[-10:]:
            print(f"  {log_entry}")
        print("\nIntegration utility finished.")

if __name__ == "__main__":
    main()

    