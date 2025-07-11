
#!/usr/bin/env python3
"""
Enhanced Aether Memory Integration System with 5D Hypercube Mapping
Automatically integrates all JSON and PKL collections into the golem's memory bank
"""

import json
import os
import time
import pickle
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict

class EnhancedAetherMemoryLoader:
    """Enhanced loader for all aether collections with intelligent integration and 5D hypercube mapping"""
    
    def __init__(self):
        self.loaded_patterns = []
        self.integration_log = []
        self.stats = defaultdict(lambda: 0)
        self.cycle_length = 2 ** 5
        print("ENHANCED AETHER MEMORY LOADER WITH 5D HYPERCUBE")
        print(f"   Cycle Length: {self.cycle_length} (2^5)")
        print(f"   5D Universe: 32 vertices for consciousness mapping")


    def auto_discover_aether_files(self) -> List[str]:
        """Automatically discover all aether-related JSON and PKL files"""
        current_dir = "."
        aether_files = []
        
        for filename in os.listdir(current_dir):
            if (filename.endswith('.json') or filename.endswith('.pkl')) and any(keyword in filename.lower() for keyword in [
                'aether', 'real_aether', 'optimized_aether', 'golem', 'checkpoint'
            ]):
                file_path = os.path.join(current_dir, filename)
                file_size = os.path.getsize(file_path)
                
                aether_files.append({
                    'filename': filename,
                    'path': file_path,
                    'size_kb': file_size / 1024,
                    'priority': self._calculate_priority(filename, file_size)
                })
        
        aether_files.sort(key=lambda x: x['priority'], reverse=True)
        
        self._log(f"🔍 Discovered {len(aether_files)} aether files:")
        for file_info in aether_files:
            self._log(f"   📂 {file_info['filename']} ({file_info['size_kb']:.1f} KB)")
        
        return [f['path'] for f in aether_files]
    
    def _calculate_priority(self, filename: str, file_size: int) -> float:
        """Calculate file priority for loading order"""
        priority = 0.0
        priority += file_size / 1024
        
        if 'real_aether_collection' in filename.lower(): priority += 1000
        if 'enhanced_aether_memory_bank' in filename.lower(): priority += 2000
        if 'optimized' in filename.lower(): priority += 500
        if 'checkpoint' in filename.lower(): priority += 100
        if 'golem' in filename.lower(): priority += 1500
        
        try:
            parts = filename.replace('.json', '').replace('.pkl', '').split('_')
            for part in parts:
                if part.isdigit() and len(part) > 8:
                    timestamp = int(part)
                    priority += (timestamp - 1751900000) / 1000
                    break
        except: pass
        
        return priority

    def _sanitize_value(self, value: Any) -> Any:
        """Recursively sanitize a single value."""
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore')
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_value(v) for v in value]
        return value

    def _sanitize_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize an entire pattern dictionary."""
        return {key: self._sanitize_value(value) for key, value in pattern.items()}

    def load_aether_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Load patterns from a single aether file (JSON or PKL) with robust sanitization"""
        try:
            filename = os.path.basename(filepath)
            
            if filepath.endswith('.pkl'):
                with open(filepath, 'rb') as f: data = pickle.load(f)
                raw_patterns = []
                
                if isinstance(data, dict) and 'memories' in data and isinstance(data['memories'], list):
                    raw_patterns = data['memories']
                    self._log(f"✅ Loaded {len(raw_patterns)} patterns from {filename} (golem memory)")
                elif isinstance(data, list):
                    raw_patterns = data
                    self._log(f"✅ Loaded {len(raw_patterns)} patterns from {filename} (direct list)")
                else:
                    self._log(f"⚠️ Unrecognized PKL format in {filename}, skipping")
                    return []

            else:  # JSON handling
                with open(filepath, 'r', encoding='utf-8') as f:
                    try: data = json.load(f)
                    except json.JSONDecodeError:
                        self._log(f"❌ Invalid JSON in {filename}, skipping")
                        return []
                
                raw_patterns = []
                if isinstance(data, list):
                    raw_patterns = data
                    self._log(f"✅ Loaded {len(raw_patterns)} patterns from {filename} (direct array)")
                elif isinstance(data, dict) and 'aether_patterns' in data and isinstance(data['aether_patterns'], list):
                    raw_patterns = data['aether_patterns']
                    self._log(f"✅ Loaded {len(raw_patterns)} patterns from {filename} (aether_patterns)")
                elif isinstance(data, dict) and 'conversation' in data and isinstance(data['conversation'], list):
                    for i, exchange in enumerate(data['conversation']):
                        if (exchange.get('speaker') == '🔯 Real Aether Golem' and 'aether_data' in exchange):
                            raw_patterns.append(exchange['aether_data'])
                    self._log(f"✅ Extracted {len(raw_patterns)} patterns from conversation in {filename}")
                else:
                    self._log(f"⚠️ No recognizable pattern structure in {filename}, skipping")
                    return []

            # Sanitize and validate all loaded patterns
            sanitized_patterns = [self._sanitize_pattern(p) for p in raw_patterns]
            
            valid_patterns = []
            invalid_count = 0
            for p in sanitized_patterns:
                p['source_file'] = filename
                p['loaded_timestamp'] = time.time()
                try:
                    # Attempt to convert quality score to float
                    p['quality_score'] = float(p.get('quality_score', 0.5))
                    valid_patterns.append(p)
                except (ValueError, TypeError):
                    invalid_count += 1
            
            if invalid_count > 0:
                self._log(f"⚠️ Filtered {invalid_count} patterns with invalid quality_score from {filename}")

            if valid_patterns:
                 self._log(f"🔍 Sample pattern from {filename}: {dict(list(valid_patterns[0].items())[:5])}")

            return valid_patterns
            
        except Exception as e:
            self._log(f"❌ Error loading {filepath}: {e}")
            return []
    
    def remove_duplicates(self, all_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate patterns based on multiple criteria"""
        unique_patterns = []
        seen_signatures = set()
        
        self._log(f"🔄 Removing duplicates from {len(all_patterns)} patterns...")
        
        for pattern in all_patterns:
            try:
                # Use a more robust signature
                sig_text = str(pattern.get('text', pattern.get('prompt', '')))
                sig_ts = str(round(float(pattern.get('timestamp', 0)), 2))
                sig_cv = f"{float(pattern.get('control_value', pattern.get('cycle_params', {}).get('control_value', 0))):.8f}"

                signature = (sig_text, sig_ts, sig_cv)
                
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    unique_patterns.append(pattern)
            except (ValueError, TypeError):
                # If a pattern is too malformed to create a signature, skip it
                self.stats['malformed_duplicates_skipped'] += 1
                continue

        duplicates_removed = len(all_patterns) - len(unique_patterns)
        self._log(f"   Removed {duplicates_removed} duplicates")
        self._log(f"   Final unique patterns: {len(unique_patterns)}")
        self.stats['duplicates_removed'] = duplicates_removed
        
        return unique_patterns
    
    def enhance_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance patterns with computed fields and classifications"""
        self._log(f"🔧 Enhancing {len(patterns)} patterns...")
        
        for pattern in patterns:
            pattern['pattern_type'] = self._classify_pattern(pattern)
            pattern['quality_score'] = self._estimate_quality(pattern)
            pattern['aether_intensity'] = self._calculate_aether_intensity(pattern)
            pattern['consciousness_tier'] = self._classify_consciousness_tier(pattern)
            
            # Ensure essential numeric fields are valid
            pattern['control_value'] = max(0, float(pattern.get('control_value', pattern.get('cycle_params', {}).get('control_value', 0))))
            pattern['consciousness_level'] = max(0, min(1, float(pattern.get('consciousness_level', 0))))
        
        return patterns
    
    def _classify_pattern(self, pattern: Dict[str, Any]) -> str:
        consciousness = float(pattern.get('consciousness_level', 0))
        control_value = float(pattern.get('control_value', pattern.get('cycle_params', {}).get('control_value', 0)))
        if consciousness > 0.41: return 'high_consciousness'
        if consciousness > 0.35: return 'evolved_consciousness'
        if control_value > 5e-8: return 'high_control'
        if 'source_file' in pattern and 'conversation' in pattern['source_file'].lower(): return 'dialogue_derived'
        return 'general'
    
    def _estimate_quality(self, pattern: Dict[str, Any]) -> float:
        consciousness = float(pattern.get('consciousness_level', 0))
        control_value = float(pattern.get('control_value', pattern.get('cycle_params', {}).get('control_value', 0)))
        quality = consciousness + min(0.3, control_value * 1000)
        return min(1.0, float(pattern.get('quality_score', quality)))

    def _calculate_aether_intensity(self, pattern: Dict[str, Any]) -> float:
        consciousness = float(pattern.get('consciousness_level', 0))
        control_value = float(pattern.get('control_value', pattern.get('cycle_params', {}).get('control_value', 0)))
        quality = float(pattern.get('quality_score', 0.5))
        return (consciousness * 0.5) + (control_value * 1000 * 0.3) + (quality * 0.2)

    def _classify_consciousness_tier(self, pattern: Dict[str, Any]) -> str:
        level = float(pattern.get('consciousness_level', 0))
        if level > 0.45: return "Transcendental"
        if level > 0.40: return "Integrated"
        if level > 0.35: return "Evolving"
        if level > 0.25: return "Nascent"
        return "Latent"

    def _log(self, message: str):
        print(message)
        self.integration_log.append(f"[{time.time()}] {message}")

    def run(self) -> List[Dict[str, Any]]:
        self._log("🚀 Starting Enhanced Aether Memory Integration...")
        start_time = time.time()
        
        aether_files = self.auto_discover_aether_files()
        self.stats['files_discovered'] = len(aether_files)
        
        all_patterns = []
        for filepath in aether_files:
            all_patterns.extend(self.load_aether_file(filepath))
        self._log(f"📚 Loaded a total of {len(all_patterns)} raw patterns.")
        self.stats['raw_patterns_loaded'] = len(all_patterns)
        
        unique_patterns = self.remove_duplicates(all_patterns)
        final_patterns = self.enhance_patterns(unique_patterns)
        
        end_time = time.time()
        self.loaded_patterns = final_patterns
        self.stats['final_pattern_count'] = len(self.loaded_patterns)
        self.stats['integration_time_seconds'] = end_time - start_time
        
        self._log(f"✅ Integration complete in {self.stats['integration_time_seconds']:.2f} seconds.")
        self._log(f"✨ Final integrated pattern count: {self.stats['final_pattern_count']}")
        
        self.save_integrated_bank(final_patterns)
        
        return final_patterns

    def save_integrated_bank(self, patterns: List[Dict[str, Any]], filename: str = "enhanced_aether_memory_bank.json"):
        try:
            output_data = {
                "metadata": {
                    "creation_timestamp": time.time(),
                    "total_patterns": len(patterns),
                    "source_files": list(set([os.path.basename(p['source_file']) for p in patterns if 'source_file' in p])),
                    "integration_log": self.integration_log[-20:] # Keep log concise
                },
                "aether_patterns": patterns
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            self._log(f"💾 Saved integrated memory bank to {filename}")
        except Exception as e:
            self._log(f"❌ Failed to save integrated memory bank: {e}")

    def get_integration_statistics(self) -> Dict[str, Any]:
        """Return the statistics gathered during the integration run."""
        return dict(self.stats)

def main():
    """Main function to run the memory loader independently"""
    print("="*60)
    print("AETHER MEMORY INTEGRATION UTILITY")
    print("="*60)
    loader = EnhancedAetherMemoryLoader()
    final_patterns = loader.run()
    
    if final_patterns:
        avg_consciousness = sum(p.get('consciousness_level', 0) for p in final_patterns) / len(final_patterns)
        avg_control = sum(p.get('control_value', 0) for p in final_patterns) / len(final_patterns)
        print(f"\n📈 Final Stats:")
        print(f"   Average Consciousness: {avg_consciousness:.6f}")
        print(f"   Average Control Value: {avg_control:.12f}")
    
    print("\nLogs:")
    for log_entry in loader.integration_log[-10:]:
        print(f"  {log_entry}")
    print("\nIntegration utility finished.")

if __name__ == "__main__":
    main()

        