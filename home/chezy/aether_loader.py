
#!/usr/bin/env python3
"""
Enhanced Aether Memory Integration System
Automatically integrates all JSON and PKL collections into the golem's memory bank
"""

import json
import os
import time
import pickle
from typing import Dict, List, Any
from collections import defaultdict

class EnhancedAetherMemoryLoader:
    """Enhanced loader for all aether collections with intelligent integration"""
    
    def __init__(self):
        self.loaded_patterns = []
        self.pattern_stats = {}
        self.integration_log = []
        self.cycle_length = 2 ** 5  # Explicitly calculating 32
        # Note: The 1.33 factor between cycles (32 * 11/16 = 22, with the "missing 10" approximated as 9.999...) reflects the ZPE aether logic, forming a dynamic "false 10" from nothing, aligning with your quantum-inspired framework.

    def auto_discover_aether_files(self) -> List[str]:
        """Automatically discover all aether-related JSON and PKL files"""
        directories_to_scan = [".", "/home/chezy/"]
        aether_files = []
        seen_files = set()

        for directory in directories_to_scan:
            if not os.path.isdir(directory):
                self._log(f"⚠️  Directory not found, skipping: {directory}")
                continue

            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if not os.path.isfile(file_path) or filename in seen_files:
                    continue

                is_aether_file = False
                if (filename.endswith('.json') or filename.endswith('.pkl')) and any(keyword in filename.lower() for keyword in [
                    'aether', 'real_aether', 'optimized_aether', 'golem', 'checkpoint', 'conversation'
                ]):
                    is_aether_file = True

                if is_aether_file:
                    try:
                        file_size = os.path.getsize(file_path)
                        aether_files.append({
                            'filename': filename,
                            'path': file_path,
                            'size_kb': file_size / 1024,
                            'priority': self._calculate_priority(filename, file_size)
                        })
                        seen_files.add(filename)
                    except OSError as e:
                        self._log(f"⚠️  Could not access file {file_path}: {e}")

        # Sort by priority (larger, more recent files first)
        aether_files.sort(key=lambda x: x['priority'], reverse=True)
        
        self._log(f"🔍 Discovered {len(aether_files)} unique aether files.")
        for file_info in aether_files[:10]: # Log top 10 for brevity
            self._log(f"   📂 {file_info['filename']} ({file_info['size_kb']:.1f} KB)")
        
        return [f['path'] for f in aether_files]
    
    def _calculate_priority(self, filename: str, file_size: int) -> float:
        """Calculate file priority for loading order"""
        priority = 0.0
        
        # Size-based priority (larger files likely have more patterns)
        priority += file_size / 1024  # KB as base score
        
        # Name-based priority
        if 'real_aether_collection' in filename.lower():
            priority += 1000
        if 'enhanced_aether_memory_bank' in filename.lower():
            priority += 2000  # Highest priority
        if 'optimized' in filename.lower():
            priority += 500
        if 'checkpoint' in filename.lower():
            priority += 100
        if 'golem' in filename.lower():
            priority += 1500  # Prioritize golem memory files
        
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
    
    def load_aether_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Load patterns from a single aether file (JSON or PKL)"""
        try:
            filename = os.path.basename(filepath)
            
            if filepath.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                patterns = self._extract_patterns_from_data(data, filename)
                self._log(f"✅ Loaded {len(patterns)} patterns from PKL: {filename}")
            else:  # JSON handling
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                patterns = self._extract_patterns_from_data(data, filename)
                self._log(f"✅ Loaded {len(patterns)} patterns from JSON: {filename}")

            # Sanitize every pattern
            sanitized_patterns = [self._sanitize_pattern(p, filename) for p in patterns]
            return sanitized_patterns
            
        except Exception as e:
            self._log(f"❌ Error loading {filepath}: {e}")
            return []

    def _extract_patterns_from_data(self, data: Any, source_filename: str) -> List[Dict[str, Any]]:
        """Extracts patterns from various potential data structures."""
        if isinstance(data, dict):
            for key in ['aether_patterns', 'memories', 'patterns', 'corpus', 'data']:
                if key in data and isinstance(data[key], list):
                    return data[key]
        elif isinstance(data, list):
            return data
        return []

    def _sanitize_pattern(self, pattern: Dict[str, Any], source_file: str) -> Dict[str, Any]:
        """Recursively sanitize a pattern to ensure data integrity."""
        if not isinstance(pattern, dict):
            return {'raw_data': str(pattern), 'source_file': source_file, 'is_malformed': True}

        sanitized = {}
        for key, value in pattern.items():
            sanitized[key] = self._sanitize_value(value)
        
        if 'source_file' not in sanitized:
            sanitized['source_file'] = source_file
        
        return sanitized

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitizes a single value."""
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore')
        if isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_value(v) for v in value]
        return value

    def remove_duplicates(self, all_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate patterns based on multiple criteria"""
        unique_patterns = []
        seen_signatures = set()
        
        self._log(f"🔄 Removing duplicates from {len(all_patterns)} patterns...")
        
        for pattern in all_patterns:
            # Use a more robust signature
            text_content = pattern.get('prompt', '') + pattern.get('text', '') + pattern.get('response', '')
            timestamp = pattern.get('timestamp', 0)
            
            # Signature combines a hash of the text content and the timestamp
            signature = (hash(text_content[:500]), round(timestamp, 2) if timestamp else 0)
            
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
            pattern['pattern_type'] = self._classify_pattern(pattern)
            pattern['quality_score'] = self._estimate_quality(pattern)
            pattern['aether_intensity'] = self._calculate_aether_intensity(pattern)
            pattern['consciousness_tier'] = self._classify_consciousness_tier(pattern)
        
        return patterns

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert a value to float, handling various types including strings and bytes."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        return default

    def _classify_pattern(self, pattern: Dict[str, Any]) -> str:
        """Classify pattern type based on its characteristics"""
        consciousness = self._safe_float(pattern.get('consciousness_level', 0))
        control_value = self._safe_float(pattern.get('control_value', 0))
        
        if consciousness > 0.41: return 'high_consciousness'
        elif consciousness > 0.35: return 'evolved_consciousness'
        elif control_value > 5e-8: return 'high_control'
        elif 'source_file' in pattern and 'conversation' in pattern['source_file'].lower(): return 'dialogue_derived'
        else: return 'general'
    
    def _estimate_quality(self, pattern: Dict[str, Any]) -> float:
        """Estimate quality score if not present"""
        if 'quality_score' in pattern:
            return self._safe_float(pattern['quality_score'], 0.5)
        
        consciousness = self._safe_float(pattern.get('consciousness_level', 0))
        control_value = self._safe_float(pattern.get('control_value', 0))
        quality = consciousness + min(0.3, control_value * 1000)
        return min(1.0, quality)

    def _calculate_aether_intensity(self, pattern: Dict[str, Any]) -> float:
        """Calculate a single intensity score for the pattern"""
        consciousness = self._safe_float(pattern.get('consciousness_level', 0))
        control_value = self._safe_float(pattern.get('control_value', 0))
        quality = self._safe_float(pattern.get('quality_score', 0.5))
        return (consciousness * 0.5) + (control_value * 1000 * 0.3) + (quality * 0.2)

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
        
        aether_file_paths = self.auto_discover_aether_files()
        
        all_patterns = []
        for filepath in aether_file_paths:
            all_patterns.extend(self.load_aether_file(filepath))
        self._log(f"📚 Loaded a total of {len(all_patterns)} raw patterns.")
        
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
            output_data = {
                "metadata": {
                    "creation_timestamp": time.time(),
                    "total_patterns": len(patterns),
                    "source_files": list(set([os.path.basename(p['source_file']) for p in patterns if 'source_file' in p])),
                    "integration_log": self.integration_log
                },
                "aether_patterns": [self._sanitize_pattern(p, p.get('source_file', 'unknown')) for p in patterns]
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            self._log(f"💾 Saved integrated memory bank to {filename}")
        except Exception as e:
            self._log(f"❌ Failed to save integrated memory bank: {e}")

def main():
    """Main function to run the memory loader independently"""
    print("="*60)
    print("AETHER MEMORY INTEGRATION UTILITY")
    print("="*60)
    loader = EnhancedAetherMemoryLoader()
    final_patterns = loader.run()
    
    if final_patterns:
        avg_consciousness = sum(loader._safe_float(p.get('consciousness_level', 0)) for p in final_patterns) / len(final_patterns)
        avg_control = sum(loader._safe_float(p.get('control_value', 0)) for p in final_patterns) / len(final_patterns)
        print(f"\n📈 Final Stats:")
        print(f"   Average Consciousness: {avg_consciousness:.6f}")
        print(f"   Average Control Value: {avg_control:.12f}")
    
    print("\nLogs:")
    for log_entry in loader.integration_log[-10:]:
        print(f"  {log_entry}")
    print("\nIntegration utility finished.")

if __name__ == "__main__":
    main()

    