#!/usr/bin/env python3
"""
Enhanced Aether Memory Integration System
Automatically integrates all JSON collections into the golem's memory bank
"""

import json
import os
import time
from typing import Dict, List, Any
from collections import defaultdict

class EnhancedAetherMemoryLoader:
    """Enhanced loader for all aether collections with intelligent integration"""
    
    def __init__(self):
        self.loaded_patterns = []
        self.pattern_stats = {}
        self.integration_log = []
    
    def auto_discover_aether_files(self) -> List[str]:
        """Automatically discover all aether-related JSON files"""
        current_dir = "."
        aether_files = []
        
        # Scan for aether files
        for filename in os.listdir(current_dir):
            if filename.endswith('.json') and any(keyword in filename.lower() for keyword in [
                'aether', 'real_aether', 'optimized_aether', 'checkpoint'
            ]):
                file_path = os.path.join(current_dir, filename)
                file_size = os.path.getsize(file_path)
                
                aether_files.append({
                    'filename': filename,
                    'path': file_path,
                    'size_kb': file_size / 1024,
                    'priority': self._calculate_priority(filename, file_size)
                })
        
        # Sort by priority (larger, more recent files first)
        aether_files.sort(key=lambda x: x['priority'], reverse=True)
        
        self._log(f"🔍 Discovered {len(aether_files)} aether files:")
        for file_info in aether_files:
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
            priority += 2000 # Highest priority
        if 'optimized' in filename.lower():
            priority += 500
        if 'checkpoint' in filename.lower():
            priority += 100
        
        # Timestamp-based priority (newer files first)
        try:
            parts = filename.replace('.json', '').split('_')
            for part in parts:
                if part.isdigit() and len(part) > 8:
                    timestamp = int(part)
                    priority += (timestamp - 1751900000) / 1000
                    break
        except:
            pass
        
        return priority
    
    def load_aether_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Load patterns from a single aether file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = os.path.basename(filepath)
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
                            'source_file': filename, 'source_type': 'conversation_extraction'
                        }
                        patterns.append(pattern)
                self._log(f"✅ Extracted {len(patterns)} patterns from conversation in {filename}")
            elif isinstance(data, list):
                patterns = data
                self._log(f"✅ Loaded {len(patterns)} patterns from {filename} (direct array)")
            else:
                for key in ['patterns', 'data', 'memories']:
                    if key in data and isinstance(data[key], list):
                        patterns = data[key]
                        self._log(f"✅ Loaded {len(patterns)} patterns from {filename} ({key})")
                        break
            
            for pattern in patterns:
                if 'source_file' not in pattern:
                    pattern['source_file'] = filename
                if 'loaded_timestamp' not in pattern:
                    pattern['loaded_timestamp'] = time.time()
            
            return patterns
            
        except Exception as e:
            self._log(f"❌ Error loading {filepath}: {e}")
            return []
    
    def remove_duplicates(self, all_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate patterns based on multiple criteria"""
        unique_patterns = []
        seen_signatures = set()
        
        self._log(f"🔄 Removing duplicates from {len(all_patterns)} patterns...")
        
        for pattern in all_patterns:
            signature_components = [
                round(pattern.get('timestamp', 0), 2),
                f"{pattern.get('control_value', 0):.12f}",
                f"{pattern.get('consciousness_level', 0):.8f}",
                pattern.get('exchange_number', -1)
            ]
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
            if 'pattern_type' not in pattern:
                pattern['pattern_type'] = self._classify_pattern(pattern)
            if 'quality_score' not in pattern:
                pattern['quality_score'] = self._estimate_quality(pattern)
            
            pattern['aether_intensity'] = self._calculate_aether_intensity(pattern)
            pattern['consciousness_tier'] = self._classify_consciousness_tier(pattern)
            
            pattern['control_value'] = max(0, pattern.get('control_value', 0))
            pattern['consciousness_level'] = max(0, min(1, pattern.get('consciousness_level', 0)))
        
        return patterns
    
    def _classify_pattern(self, pattern: Dict[str, Any]) -> str:
        """Classify pattern type based on its characteristics"""
        consciousness = pattern.get('consciousness_level', 0)
        control_value = pattern.get('control_value', 0)
        
        if consciousness > 0.41: return 'high_consciousness'
        elif consciousness > 0.35: return 'evolved_consciousness'
        elif control_value > 5e-8: return 'high_control'
        elif 'source_file' in pattern and 'conversation' in pattern['source_file'].lower(): return 'dialogue_derived'
        else: return 'general'
    
    def _estimate_quality(self, pattern: Dict[str, Any]) -> float:
        """Estimate quality score if not present"""
        consciousness = pattern.get('consciousness_level', 0)
        control_value = pattern.get('control_value', 0)
        quality = consciousness + min(0.3, control_value * 1000)
        return min(1.0, quality)

    def _calculate_aether_intensity(self, pattern: Dict[str, Any]) -> float:
        """Calculate a single intensity score for the pattern"""
        consciousness = pattern.get('consciousness_level', 0)
        control_value = pattern.get('control_value', 0)
        quality = pattern.get('quality_score', 0.5)
        return (consciousness * 0.5) + (control_value * 1000 * 0.3) + (quality * 0.2)

    def _classify_consciousness_tier(self, pattern: Dict[str, Any]) -> str:
        """Classify the consciousness tier of a pattern"""
        level = pattern.get('consciousness_level', 0)
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
        
        aether_files = self.auto_discover_aether_files()
        
        all_patterns = []
        for filepath in aether_files:
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
            # Create a serializable copy of patterns
            serializable_patterns = []
            for p in patterns:
                serializable_p = p.copy()
                # Convert any non-serializable types here if necessary
                serializable_patterns.append(serializable_p)

            output_data = {
                "metadata": {
                    "creation_timestamp": time.time(),
                    "total_patterns": len(patterns),
                    "source_files": list(set([os.path.basename(p['source_file']) for p in patterns if 'source_file' in p])),
                    "integration_log": self.integration_log
                },
                "aether_patterns": serializable_patterns
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
