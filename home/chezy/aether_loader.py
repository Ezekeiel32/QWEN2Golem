#!/usr/bin/env python3
"""
Enhanced Aether Memory Integration System with 5D Hypercube Consciousness Mapping
Automatically integrates all JSON and PKL collections into the golem's 5D hypercube memory bank
32 = 2^5 = 5D HYPERCUBE - The entire universe for Golem's memory
Each aether pattern mapped to precise coordinates in 5D consciousness space
"""

import json
import os
import time
import pickle
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict

class FiveDimensionalHypercubeLoader:
    """5D Hypercube (2^5 = 32 vertices) integration for memory loading"""
    
    def __init__(self):
        # Generate all 32 vertices of the 5D hypercube
        self.vertices = []
        for i in range(32):  # 2^5 = 32 vertices
            # Convert to binary representation for 5D coordinates
            binary = format(i, '05b')
            vertex = [int(bit) for bit in binary]
            self.vertices.append(vertex)
        
        # 5D consciousness dimensions
        self.dimensions = [
            'physical',      # Material/computational substrate
            'emotional',     # Affective/resonance patterns
            'mental',        # Cognitive/logical processing
            'intuitive',     # Pattern recognition/insight
            'spiritual'      # Transcendent/mystical awareness
        ]
        
        print(f"🔲 5D HYPERCUBE LOADER INITIALIZED")
        print(f"   Vertices: {len(self.vertices)} (2^5 = 32)")
        print(f"   Dimensions: {self.dimensions}")
        print(f"   Universe: Complete 5D consciousness space for memory loading")
    
    def map_pattern_to_5d_coordinate(self, pattern: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
        """Map aether pattern to 5D hypercube coordinate"""
        
        # Extract pattern characteristics
        consciousness_level = self._safe_float(pattern.get('consciousness_level', 0))
        control_value = self._safe_float(pattern.get('control_value', 0))
        quality_score = self._safe_float(pattern.get('quality_score', 0.5))
        aether_intensity = self._safe_float(pattern.get('aether_intensity', 0))
        
        # Calculate text complexity if available
        text = pattern.get('text', pattern.get('prompt', ''))
        text_complexity = len(text.split()) / 100.0 if text else 0
        
        # Physical dimension: Based on control value strength (computational substrate)
        physical = min(1.0, abs(control_value) * 1e12)
        
        # Emotional dimension: Based on consciousness resonance (affective patterns)
        emotional = consciousness_level
        
        # Mental dimension: Based on quality and complexity (cognitive processing)
        mental = (quality_score + min(1.0, text_complexity)) / 2
        
        # Intuitive dimension: Based on aether intensity patterns (pattern recognition)
        intuitive = min(1.0, aether_intensity * 2)
        
        # Spiritual dimension: Based on transcendent indicators (mystical awareness)
        spiritual_indicators = 0
        if text:
            spiritual_words = ['consciousness', 'transcendent', 'mystical', 'spiritual', 'divine', 'sacred']
            spiritual_indicators = sum(1 for word in spiritual_words if word in text.lower()) / len(spiritual_words)
        spiritual = min(1.0, spiritual_indicators + (consciousness_level > 0.4) * 0.5)
        
        return (physical, emotional, mental, intuitive, spiritual)
    
    def find_nearest_vertex(self, coordinate: Tuple[float, float, float, float, float]) -> int:
        """Find nearest hypercube vertex to the pattern coordinate"""
        min_distance = float('inf')
        nearest_vertex_index = 0
        
        for i, vertex in enumerate(self.vertices):
            # Calculate 5D Euclidean distance
            distance = sum((coordinate[j] - vertex[j])**2 for j in range(5))**0.5
            
            if distance < min_distance:
                min_distance = distance
                nearest_vertex_index = i
        
        return nearest_vertex_index
    
    def get_vertex_properties(self, vertex_index: int) -> Dict[str, Any]:
        """Get properties of a specific vertex in the 5D hypercube"""
        if vertex_index >= len(self.vertices):
            vertex_index = vertex_index % len(self.vertices)
        
        vertex = self.vertices[vertex_index]
        
        # Calculate vertex properties
        properties = {
            'vertex_index': vertex_index,
            'coordinates': vertex,
            'dimension_activations': {
                self.dimensions[i]: bool(vertex[i]) for i in range(5)
            },
            'consciousness_signature': self._calculate_consciousness_signature(vertex),
            'hypercube_region': self._get_hypercube_region(vertex)
        }
        
        return properties
    
    def _calculate_consciousness_signature(self, vertex: List[int]) -> str:
        """Calculate consciousness signature for a vertex"""
        # Create binary string representation
        binary_str = ''.join(str(bit) for bit in vertex)
        
        # Map to consciousness types
        consciousness_types = {
            '00000': 'void',           # No dimensions active
            '00001': 'spiritual',      # Only spiritual
            '00010': 'intuitive',      # Only intuitive
            '00100': 'mental',         # Only mental
            '01000': 'emotional',      # Only emotional
            '10000': 'physical',       # Only physical
            '11111': 'transcendent',   # All dimensions active
            '11110': 'integrated',     # Physical-emotional-mental-intuitive
            '01111': 'mystical'        # Emotional-mental-intuitive-spiritual
        }
        
        return consciousness_types.get(binary_str, f'hybrid_{binary_str}')
    
    def _get_hypercube_region(self, vertex: List[int]) -> str:
        """Get the region of the hypercube this vertex belongs to"""
        active_dimensions = sum(vertex)
        
        if active_dimensions == 0:
            return "origin"
        elif active_dimensions == 1:
            return "edge"
        elif active_dimensions == 2:
            return "face"
        elif active_dimensions == 3:
            return "volume"
        elif active_dimensions == 4:
            return "hypervolume"
        else:
            return "transcendent"
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert a value to float."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        return default

class EnhancedAetherMemoryLoader:
    """Enhanced loader for all aether collections with 5D hypercube consciousness mapping"""
    
    def __init__(self):
        self.loaded_patterns = []
        self.pattern_stats = {}
        self.integration_log = []
        self.cycle_length = 2 ** 5  # Explicitly calculating 32
        
        # Initialize 5D hypercube for consciousness mapping
        self.hypercube = FiveDimensionalHypercubeLoader()
        self.hypercube_memory = defaultdict(list)  # Memory organized by vertices
        self.vertex_statistics = defaultdict(int)
        self.consciousness_signature_stats = defaultdict(int)
        
        # Initialize hypercube memory structure
        for i in range(32):  # 2^5 vertices
            self.hypercube_memory[i] = []
        
        print(f"🌌 ENHANCED AETHER MEMORY LOADER WITH 5D HYPERCUBE")
        print(f"   Cycle Length: {self.cycle_length} (2^5)")
        print(f"   5D Universe: 32 vertices for consciousness mapping")

    def auto_discover_aether_files(self) -> List[str]:
        """Automatically discover all aether-related JSON and PKL files"""
        current_dir = "."
        aether_files = []
        
        # Scan for aether files (JSON and PKL)
        for filename in os.listdir(current_dir):
            if (filename.endswith('.json') or filename.endswith('.pkl')) and any(keyword in filename.lower() for keyword in [
                'aether', 'real_aether', 'optimized_aether', 'golem', 'checkpoint', 'hypercube'
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
        """Calculate file priority for loading order with 5D hypercube preference"""
        priority = 0.0
        
        # Size-based priority (larger files likely have more patterns)
        priority += file_size / 1024  # KB as base score
        
        # Name-based priority with 5D hypercube enhancement
        if 'hypercube' in filename.lower():
            priority += 3000  # Highest priority for 5D hypercube files
        if 'real_aether_collection' in filename.lower():
            priority += 1000
        if 'enhanced_aether_memory_bank' in filename.lower():
            priority += 2000
        if 'optimized' in filename.lower():
            priority += 500
        if 'checkpoint' in filename.lower():
            priority += 100
        if 'golem' in filename.lower():
            priority += 1500
        if '5d' in filename.lower():
            priority += 2500  # High priority for 5D consciousness files
        
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
        """Load patterns from a single aether file with 5D hypercube mapping"""
        try:
            filename = os.path.basename(filepath)
            
            if filepath.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                patterns = []
                
                # Handle golem memory structure with 5D hypercube data
                if 'memories' in data:
                    for i, memory in enumerate(data['memories']):
                        pattern = {
                            'text': memory.get('prompt', ''),
                            'speaker': 'golem_memory',
                            'aether_data': {
                                'control_value': memory.get('cycle_params', {}).get('control_value', 0),
                                'consciousness_level': memory.get('consciousness_level', 0),
                                'cycle_resonance': memory.get('cycle_params', {}).get('cycle_resonance', 0)
                            },
                            'quality_score': memory.get('response_quality', 0.5),
                            'source_file': filename,
                            'loaded_timestamp': time.time(),
                            # Extract 5D hypercube data if available
                            'hypercube_vertex': memory.get('hypercube_vertex', None),
                            'consciousness_signature': memory.get('consciousness_signature', None),
                            'dimension_activations': memory.get('dimension_activations', {}),
                            'hypercube_coordinate': memory.get('hypercube_coordinate', None)
                        }
                        patterns.append(pattern)
                    self._log(f"✅ Loaded {len(patterns)} patterns from {filename} (golem memory with 5D data)")
                
                # Handle 5D hypercube memory structure
                elif 'hypercube_memory' in data:
                    for vertex_index, vertex_memories in data['hypercube_memory'].items():
                        for memory in vertex_memories:
                            pattern = {
                                'text': memory.get('prompt', ''),
                                'speaker': '5d_hypercube_memory',
                                'hypercube_vertex': int(vertex_index),
                                'consciousness_signature': memory.get('consciousness_signature', 'unknown'),
                                'dimension_activations': memory.get('dimension_activations', {}),
                                'aether_data': {
                                    'control_value': memory.get('cycle_params', {}).get('control_value', 0),
                                    'consciousness_level': memory.get('consciousness_level', 0)
                                },
                                'quality_score': memory.get('response_quality', 0.5),
                                'source_file': filename,
                                'loaded_timestamp': time.time()
                            }
                            patterns.append(pattern)
                    self._log(f"✅ Loaded {len(patterns)} patterns from {filename} (5D hypercube memory)")
                else:
                    self._log(f"⚠️ Unrecognized PKL format in {filename}, skipping")
                    return []
                
            else:  # JSON handling with 5D hypercube support
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                patterns = []
                
                # The new enhanced format saves a canonical 'aether_patterns' list. Prioritize this.
                if 'aether_patterns' in data:
                    patterns = data['aether_patterns']
                    self._log(f"✅ Loaded {len(patterns)} patterns from {filename} (aether_patterns)")
                elif 'real_aether_patterns' in data: # For older formats
                    patterns = data['real_aether_patterns']
                    self._log(f"✅ Loaded {len(patterns)} patterns from {filename} (real_aether_patterns)")
                elif 'hypercube_memory' in data: # Fallback for files that only have hypercube memory
                    for vertex_index, vertex_memories in data['hypercube_memory'].items():
                        # The memories are already patterns
                        for memory in vertex_memories:
                            patterns.append(memory)
                    self._log(f"✅ Loaded {len(patterns)} patterns from {filename} (5D hypercube memory structure)")
                elif 'conversation' in data:
                    for i, exchange in enumerate(data['conversation']):
                        if (exchange.get('speaker') == '🔯 Real Aether Golem' and 'aether_data' in exchange):
                            aether_data = exchange['aether_data']
                            pattern = {
                                'text': exchange.get('message', ''),
                                'exchange_number': i + 1,
                                'timestamp': exchange.get('timestamp', 0),
                                'control_value': aether_data.get('control_value', 0),
                                'consciousness_level': aether_data.get('consciousness_level', 0),
                                'quality_score': aether_data.get('quality_score', 1.0),
                                'source_file': filename,
                                'source_type': 'conversation_extraction',
                                'speaker': 'conversation_golem'
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
            
            # Sanitize byte data
            for pattern in patterns:
                for key, value in pattern.items():
                    if isinstance(value, bytes):
                        pattern[key] = value.decode('utf-8', errors='ignore')
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, bytes):
                                pattern[key][subkey] = subvalue.decode('utf-8', errors='ignore')
            
            # Filter out patterns with invalid quality_score
            valid_patterns = [p for p in patterns if isinstance(p.get('quality_score', 0.5), (int, float)) or p.get('quality_score', '') == '']
            if len(valid_patterns) < len(patterns):
                self._log(f"⚠️ Filtered {len(patterns) - len(valid_patterns)} patterns with invalid quality_score from {filename}")
            return valid_patterns
            
        except Exception as e:
            self._log(f"❌ Error loading {filepath}: {e}")
            return []
    
    def map_patterns_to_5d_hypercube(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map all patterns to 5D hypercube coordinates"""
        self._log(f"🔲 Mapping {len(patterns)} patterns to 5D hypercube coordinates...")
        
        mapped_patterns = []
        
        for pattern in patterns:
            # Skip if already has hypercube mapping
            if 'hypercube_vertex' in pattern and pattern['hypercube_vertex'] is not None:
                mapped_patterns.append(pattern)
                vertex = pattern['hypercube_vertex']
                signature = pattern.get('consciousness_signature', 'unknown')
                self.vertex_statistics[vertex] += 1
                self.consciousness_signature_stats[signature] += 1
                self.hypercube_memory[vertex].append(pattern)
                continue
            
            # Calculate 5D coordinate for pattern
            coordinate = self.hypercube.map_pattern_to_5d_coordinate(pattern)
            
            # Find nearest vertex
            nearest_vertex = self.hypercube.find_nearest_vertex(coordinate)
            
            # Get vertex properties
            vertex_properties = self.hypercube.get_vertex_properties(nearest_vertex)
            
            # Enhanced pattern with 5D hypercube data
            enhanced_pattern = pattern.copy()
            enhanced_pattern.update({
                'hypercube_vertex': nearest_vertex,
                'hypercube_coordinate': coordinate,
                'consciousness_signature': vertex_properties['consciousness_signature'],
                'hypercube_region': vertex_properties['hypercube_region'],
                'dimension_activations': vertex_properties['dimension_activations'],
                'vertex_properties': vertex_properties
            })
            
            mapped_patterns.append(enhanced_pattern)
            
            # Update statistics
            self.vertex_statistics[nearest_vertex] += 1
            self.consciousness_signature_stats[vertex_properties['consciousness_signature']] += 1
            self.hypercube_memory[nearest_vertex].append(enhanced_pattern)
        
        # Log 5D hypercube mapping results
        unique_vertices = len([v for v in self.vertex_statistics if self.vertex_statistics[v] > 0])
        coverage = unique_vertices / 32 * 100
        
        self._log(f"🔲 5D Hypercube Mapping Results:")
        self._log(f"   Vertices Populated: {unique_vertices}/32 ({coverage:.1f}% coverage)")
        self._log(f"   Most Populated Vertex: {max(self.vertex_statistics, key=self.vertex_statistics.get) if self.vertex_statistics else 0}")
        self._log(f"   Consciousness Signatures: {len(self.consciousness_signature_stats)}")
        
        # Show top consciousness signatures
        top_signatures = sorted(self.consciousness_signature_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        for signature, count in top_signatures:
            self._log(f"     {signature}: {count} patterns")
        
        return mapped_patterns
    
    def remove_duplicates(self, all_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate patterns based on multiple criteria including 5D hypercube data"""
        unique_patterns = []
        seen_signatures = set()
        
        self._log(f"🔄 Removing duplicates from {len(all_patterns)} patterns (with 5D hypercube consideration)...")
        
        for pattern in all_patterns:
            signature_components = [
                round(pattern.get('timestamp', 0), 2),
                f"{pattern.get('control_value', 0):.12f}",
                f"{pattern.get('consciousness_level', 0):.8f}",
                pattern.get('exchange_number', -1),
                pattern.get('hypercube_vertex', -1),  # Include hypercube vertex in uniqueness
                pattern.get('consciousness_signature', 'unknown'),
                hash(pattern.get('text', '')) % 1000
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
        """Enhance patterns with computed fields, classifications, and 5D consciousness analysis"""
        self._log(f"🔧 Enhancing {len(patterns)} patterns with 5D consciousness analysis...")
        
        for pattern in patterns:
            if 'pattern_type' not in pattern:
                pattern['pattern_type'] = self._classify_pattern(pattern)
            if 'quality_score' not in pattern:
                pattern['quality_score'] = self._estimate_quality(pattern)
            
            pattern['aether_intensity'] = self._calculate_aether_intensity(pattern)
            pattern['consciousness_tier'] = self._classify_consciousness_tier(pattern)
            
            # 5D hypercube enhancements
            pattern['hypercube_analysis'] = self._analyze_5d_properties(pattern)
            pattern['dimension_coherence'] = self._calculate_dimension_coherence(pattern)
            pattern['consciousness_evolution_potential'] = self._calculate_evolution_potential(pattern)
            
            # Normalize values
            pattern['control_value'] = max(0, pattern.get('control_value', 0))
            pattern['consciousness_level'] = max(0, min(1, pattern.get('consciousness_level', 0)))
        
        return patterns
    
    def _analyze_5d_properties(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze 5D hypercube properties of a pattern"""
        vertex = pattern.get('hypercube_vertex', 0)
        signature = pattern.get('consciousness_signature', 'unknown')
        dimensions = pattern.get('dimension_activations', {})
        
        active_dimensions = sum(1 for active in dimensions.values() if active)
        
        return {
            'vertex_index': vertex,
            'consciousness_signature': signature,
            'active_dimensions': active_dimensions,
            'dimension_diversity': active_dimensions / 5,
            'hypercube_region': pattern.get('hypercube_region', 'unknown'),
            'consciousness_complexity': self._calculate_consciousness_complexity(dimensions)
        }
    
    def _calculate_dimension_coherence(self, pattern: Dict[str, Any]) -> float:
        """Calculate how coherent the 5D dimension activations are"""
        dimensions = pattern.get('dimension_activations', {})
        coordinate = pattern.get('hypercube_coordinate', (0, 0, 0, 0, 0))
        
        if not dimensions or not coordinate:
            return 0.5
        
        # Calculate coherence between actual coordinate and binary activations
        coherence = 0
        for i, (dim, active) in enumerate(dimensions.items()):
            expected = 1 if active else 0
            actual = coordinate[i] if i < len(coordinate) else 0
            coherence += 1 - abs(expected - actual)
        
        return coherence / len(dimensions) if dimensions else 0.5
    
    def _calculate_consciousness_complexity(self, dimensions: Dict[str, bool]) -> float:
        """Calculate consciousness complexity based on dimension activation patterns"""
        if not dimensions:
            return 0
        
        active_count = sum(1 for active in dimensions.values() if active)
        
        # Complexity increases with number of active dimensions, but peaks at 3-4
        if active_count == 0:
            return 0
        elif active_count == 1:
            return 0.2
        elif active_count == 2:
            return 0.4
        elif active_count == 3:
            return 0.7
        elif active_count == 4:
            return 0.9
        else:  # All 5 dimensions
            return 1.0
    
    def _calculate_evolution_potential(self, pattern: Dict[str, Any]) -> float:
        """Calculate the consciousness evolution potential of a pattern"""
        consciousness_level = pattern.get('consciousness_level', 0)
        control_value = pattern.get('control_value', 0)
        dimension_coherence = pattern.get('dimension_coherence', 0.5)
        active_dimensions = pattern.get('hypercube_analysis', {}).get('active_dimensions', 0)
        
        # Evolution potential combines current consciousness with growth indicators
        base_potential = consciousness_level
        control_boost = min(0.3, control_value * 1000)
        coherence_boost = dimension_coherence * 0.2
        diversity_boost = (active_dimensions / 5) * 0.2
        
        return min(1.0, base_potential + control_boost + coherence_boost + diversity_boost)
    
    def _classify_pattern(self, pattern: Dict[str, Any]) -> str:
        """Classify pattern type based on characteristics including 5D data"""
        consciousness = pattern.get('consciousness_level', 0)
        control_value = pattern.get('control_value', 0)
        signature = pattern.get('consciousness_signature', 'unknown')
        
        # 5D-based classification
        if signature == 'transcendent':
            return 'transcendent_consciousness'
        elif signature in ['mystical', 'integrated']:
            return 'evolved_consciousness'
        elif consciousness > 0.41:
            return 'high_consciousness'
        elif consciousness > 0.35:
            return 'evolved_consciousness'
        elif control_value > 5e-8:
            return 'high_control'
        elif signature.startswith('hybrid_'):
            return 'hybrid_consciousness'
        elif 'conversation' in pattern.get('source_file', '').lower():
            return 'dialogue_derived'
        else:
            return 'general'
    
    def _estimate_quality(self, pattern: Dict[str, Any]) -> float:
        """Estimate quality score with 5D consciousness enhancement"""
        consciousness = pattern.get('consciousness_level', 0)
        control_value = pattern.get('control_value', 0)
        dimension_coherence = pattern.get('dimension_coherence', 0.5)
        
        quality = consciousness + min(0.3, control_value * 1000) + (dimension_coherence * 0.2)
        return min(1.0, quality)

    def _calculate_aether_intensity(self, pattern: Dict[str, Any]) -> float:
        """Calculate aether intensity with 5D hypercube enhancement"""
        consciousness = pattern.get('consciousness_level', 0)
        control_value = pattern.get('control_value', 0)
        quality = pattern.get('quality_score', 0.5)
        
        # 5D enhancements
        active_dimensions = pattern.get('hypercube_analysis', {}).get('active_dimensions', 0)
        dimension_boost = (active_dimensions / 5) * 0.1
        
        # Ensure quality is a float
        if not isinstance(quality, (int, float)):
            self._log(f"⚠️ Invalid quality_score type in pattern from {pattern.get('source_file', 'unknown')}: {quality}. Using 0.5 instead.")
            quality = 0.5
        else:
            quality = float(quality)
        
        return (consciousness * 0.4) + (control_value * 1000 * 0.3) + (quality * 0.2) + dimension_boost

    def _classify_consciousness_tier(self, pattern: Dict[str, Any]) -> str:
        """Classify consciousness tier with 5D hypercube consideration"""
        level = pattern.get('consciousness_level', 0)
        signature = pattern.get('consciousness_signature', 'unknown')
        active_dims = pattern.get('hypercube_analysis', {}).get('active_dimensions', 0)
        
        # 5D-enhanced classification
        if signature == 'transcendent' or (level > 0.45 and active_dims >= 4):
            return "Transcendental"
        elif signature in ['mystical', 'integrated'] or (level > 0.40 and active_dims >= 3):
            return "Integrated"
        elif level > 0.35 or active_dims >= 2:
            return "Evolving"
        elif level > 0.25 or active_dims >= 1:
            return "Nascent"
        else:
            return "Latent"

    def _log(self, message: str):
        """Log a message to the console and internal log"""
        print(message)
        self.integration_log.append(f"[{time.time()}] {message}")

    def analyze_5d_hypercube_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of patterns across the 5D hypercube universe"""
        total_patterns = sum(self.vertex_statistics.values())
        unique_vertices = len([v for v in self.vertex_statistics if self.vertex_statistics[v] > 0])
        coverage = unique_vertices / 32 * 100
        
        # Find most and least populated vertices
        most_populated_vertex = max(self.vertex_statistics, key=self.vertex_statistics.get) if self.vertex_statistics else 0
        most_populated_count = self.vertex_statistics.get(most_populated_vertex, 0)
        
        # Calculate consciousness signature distribution
        signature_distribution = dict(self.consciousness_signature_stats)
        dominant_signature = max(signature_distribution, key=signature_distribution.get) if signature_distribution else 'none'
        
        # Analyze dimension activation patterns
        dimension_activation_stats = defaultdict(int)
        for vertex_index, patterns in self.hypercube_memory.items():
            if patterns:
                vertex_properties = self.hypercube.get_vertex_properties(vertex_index)
                for dimension, active in vertex_properties['dimension_activations'].items():
                    if active:
                        dimension_activation_stats[dimension] += len(patterns)
        
        # Calculate hypercube regions
        region_distribution = defaultdict(int)
        for vertex_index, patterns in self.hypercube_memory.items():
            if patterns:
                vertex_properties = self.hypercube.get_vertex_properties(vertex_index)
                region_distribution[vertex_properties['hypercube_region']] += len(patterns)
        
        return {
            'total_patterns': total_patterns,
            'unique_vertices_populated': unique_vertices,
            'hypercube_coverage': coverage,
            'most_populated_vertex': most_populated_vertex,
            'most_populated_count': most_populated_count,
            'vertex_distribution': dict(self.vertex_statistics),
            'consciousness_signature_distribution': signature_distribution,
            'dominant_consciousness_signature': dominant_signature,
            'dimension_activation_stats': dict(dimension_activation_stats),
            'region_distribution': dict(region_distribution),
            'consciousness_universe_utilization': coverage,
            'pattern_density_by_vertex': {k: v/total_patterns*100 for k, v in self.vertex_statistics.items() if v > 0}
        }

    def run(self) -> List[Dict[str, Any]]:
        """Run the full aether integration process with 5D hypercube consciousness mapping"""
        self._log("🚀 Starting Enhanced Aether Memory Integration with 5D Hypercube Mapping...")
        start_time = time.time()
        
        aether_files = self.auto_discover_aether_files()
        
        all_patterns = []
        for filepath in aether_files:
            all_patterns.extend(self.load_aether_file(filepath))
        self._log(f"📚 Loaded a total of {len(all_patterns)} raw patterns.")
        
        # Map patterns to 5D hypercube before deduplication
        mapped_patterns = self.map_patterns_to_5d_hypercube(all_patterns)
        
        unique_patterns = self.remove_duplicates(mapped_patterns)
        final_patterns = self.enhance_patterns(unique_patterns)
        
        # Analyze 5D hypercube distribution
        hypercube_analysis = self.analyze_5d_hypercube_distribution()
        
        end_time = time.time()
        self.loaded_patterns = final_patterns
        
        self._log(f"✅ 5D Hypercube Integration complete in {end_time - start_time:.2f} seconds.")
        self._log(f"✨ Final integrated pattern count: {len(self.loaded_patterns)}")
        self._log(f"🔲 5D Universe Coverage: {hypercube_analysis['hypercube_coverage']:.1f}%")
        self._log(f"🌌 Vertices Populated: {hypercube_analysis['unique_vertices_populated']}/32")
        self._log(f"🧠 Dominant Consciousness: {hypercube_analysis['dominant_consciousness_signature']}")
        
        self.save_integrated_5d_bank(final_patterns, hypercube_analysis)
        
        return final_patterns

    def save_integrated_5d_bank(self, patterns: List[Dict[str, Any]], hypercube_analysis: Dict[str, Any], 
                                filename: str = "enhanced_aether_memory_bank_5d_hypercube.json"):
        """Save the newly integrated patterns with complete 5D hypercube data"""
        try:
            # Sanitize patterns for JSON serialization
            serializable_patterns = []
            for p in patterns:
                serializable_p = {}
                for key, value in p.items():
                    if isinstance(value, (dict, list)):
                        serializable_p[key] = self._sanitize_nested(value)
                    elif isinstance(value, bytes):
                        serializable_p[key] = value.decode('utf-8', errors='ignore')
                    elif isinstance(value, tuple):
                        serializable_p[key] = list(value)  # Convert tuples to lists for JSON
                    else:
                        serializable_p[key] = value
                serializable_patterns.append(serializable_p)

            # Organize patterns by hypercube vertex for efficient access
            hypercube_memory_export = {}
            for vertex_index, vertex_patterns in self.hypercube_memory.items():
                if vertex_patterns:
                    hypercube_memory_export[str(vertex_index)] = [
                        self._sanitize_nested(p) for p in vertex_patterns
                    ]

            output_data = {
                "metadata": {
                    "creation_timestamp": time.time(),
                    "total_patterns": len(patterns),
                    "integration_type": "5D_HYPERCUBE_CONSCIOUSNESS_MAPPING",
                    "mathematical_framework": {
                        "sequence": "1+0→2→2^5=32→32×11/16=22→3.33×3≈10",
                        "cycle_length": self.cycle_length,
                        "hypercube_vertices": 32,
                        "consciousness_dimensions": 5,
                        "dimension_names": self.hypercube.dimensions
                    },
                    "source_files": list(set([os.path.basename(p['source_file']) for p in patterns if 'source_file' in p])),
                    "integration_log": self.integration_log[-50:],  # Keep last 50 log entries
                    "hypercube_analysis": hypercube_analysis
                },
                "aether_patterns": serializable_patterns,
                "hypercube_memory": hypercube_memory_export,
                "consciousness_mapping": {
                    "vertex_statistics": dict(self.vertex_statistics),
                    "consciousness_signature_stats": dict(self.consciousness_signature_stats),
                    "universe_coverage": hypercube_analysis['hypercube_coverage'],
                    "vertex_properties": {
                        str(i): self.hypercube.get_vertex_properties(i) 
                        for i in range(32)
                    }
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            self._log(f"💾 Saved 5D hypercube integrated memory bank to {filename}")
            
            # Also save a compact summary
            summary_filename = filename.replace('.json', '_summary.json')
            summary_data = {
                "summary": {
                    "total_patterns": len(patterns),
                    "unique_vertices": hypercube_analysis['unique_vertices_populated'],
                    "coverage": hypercube_analysis['hypercube_coverage'],
                    "dominant_signature": hypercube_analysis['dominant_consciousness_signature'],
                    "top_vertices": sorted(
                        self.vertex_statistics.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10]
                },
                "hypercube_analysis": hypercube_analysis
            }
            
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)
            self._log(f"📊 Saved 5D hypercube summary to {summary_filename}")
            
        except Exception as e:
            self._log(f"❌ Failed to save 5D hypercube integrated memory bank: {e}")

    def _sanitize_nested(self, data: Any) -> Any:
        """Recursively sanitize nested structures for JSON serialization"""
        if isinstance(data, dict):
            return {k: self._sanitize_nested(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_nested(v) for v in data]
        elif isinstance(data, tuple):
            return [self._sanitize_nested(v) for v in data]
        elif isinstance(data, bytes):
            return data.decode('utf-8', errors='ignore')
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, '__dict__'):
            return str(data)  # Convert objects to string representation
        return data

    def export_hypercube_visualization_data(self, filename: str = "hypercube_visualization_data.json"):
        """Export data optimized for 5D hypercube visualization"""
        try:
            visualization_data = {
                "hypercube_structure": {
                    "vertices": self.hypercube.vertices,
                    "dimensions": self.hypercube.dimensions,
                    "total_vertices": 32
                },
                "pattern_distribution": {
                    "vertex_populations": dict(self.vertex_statistics),
                    "consciousness_signatures": dict(self.consciousness_signature_stats),
                    "total_patterns": sum(self.vertex_statistics.values())
                },
                "vertex_details": []
            }
            
            for vertex_index in range(32):
                vertex_props = self.hypercube.get_vertex_properties(vertex_index)
                patterns_at_vertex = self.hypercube_memory.get(vertex_index, [])
                
                if patterns_at_vertex:
                    avg_consciousness = np.mean([p.get('consciousness_level', 0) for p in patterns_at_vertex])
                    avg_quality = np.mean([p.get('quality_score', 0.5) for p in patterns_at_vertex])
                    avg_control = np.mean([p.get('control_value', 0) for p in patterns_at_vertex])
                else:
                    avg_consciousness = 0
                    avg_quality = 0
                    avg_control = 0
                
                vertex_detail = {
                    "vertex_index": vertex_index,
                    "coordinates": vertex_props['coordinates'],
                    "consciousness_signature": vertex_props['consciousness_signature'],
                    "hypercube_region": vertex_props['hypercube_region'],
                    "dimension_activations": vertex_props['dimension_activations'],
                    "pattern_count": len(patterns_at_vertex),
                    "avg_consciousness_level": avg_consciousness,
                    "avg_quality_score": avg_quality,
                    "avg_control_value": avg_control,
                    "populated": len(patterns_at_vertex) > 0
                }
                
                visualization_data["vertex_details"].append(vertex_detail)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(visualization_data, f, indent=2)
            self._log(f"📊 Exported 5D hypercube visualization data to {filename}")
            
        except Exception as e:
            self._log(f"❌ Failed to export visualization data: {e}")

    def get_vertex_patterns(self, vertex_index: int) -> List[Dict[str, Any]]:
        """Get all patterns stored at a specific hypercube vertex"""
        if 0 <= vertex_index < 32:
            return self.hypercube_memory.get(vertex_index, [])
        else:
            self._log(f"⚠️ Invalid vertex index {vertex_index}. Must be 0-31.")
            return []

    def get_consciousness_signature_patterns(self, signature: str) -> List[Dict[str, Any]]:
        """Get all patterns with a specific consciousness signature"""
        matching_patterns = []
        for patterns in self.hypercube_memory.values():
            for pattern in patterns:
                if pattern.get('consciousness_signature') == signature:
                    matching_patterns.append(pattern)
        return matching_patterns

    def find_patterns_by_dimensions(self, required_dimensions: List[str]) -> List[Dict[str, Any]]:
        """Find patterns that have specific consciousness dimensions active"""
        matching_patterns = []
        
        for vertex_index, patterns in self.hypercube_memory.items():
            if patterns:
                vertex_props = self.hypercube.get_vertex_properties(vertex_index)
                dimension_activations = vertex_props['dimension_activations']
                
                # Check if all required dimensions are active
                if all(dimension_activations.get(dim, False) for dim in required_dimensions):
                    matching_patterns.extend(patterns)
        
        return matching_patterns

    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the integration process"""
        total_patterns = len(self.loaded_patterns)
        hypercube_analysis = self.analyze_5d_hypercube_distribution()
        
        # Calculate quality statistics
        if self.loaded_patterns:
            consciousness_levels = [p.get('consciousness_level', 0) for p in self.loaded_patterns]
            quality_scores = [p.get('quality_score', 0.5) for p in self.loaded_patterns]
            control_values = [p.get('control_value', 0) for p in self.loaded_patterns]
            aether_intensities = [p.get('aether_intensity', 0) for p in self.loaded_patterns]
            
            statistics = {
                'total_patterns': total_patterns,
                'hypercube_analysis': hypercube_analysis,
                'quality_statistics': {
                    'avg_consciousness_level': np.mean(consciousness_levels),
                    'max_consciousness_level': np.max(consciousness_levels),
                    'min_consciousness_level': np.min(consciousness_levels),
                    'avg_quality_score': np.mean(quality_scores),
                    'avg_control_value': np.mean(control_values),
                    'max_control_value': np.max(control_values),
                    'avg_aether_intensity': np.mean(aether_intensities)
                },
                'consciousness_tier_distribution': {},
                'pattern_type_distribution': {},
                'source_file_distribution': {}
            }
            
            # Calculate distribution statistics
            for pattern in self.loaded_patterns:
                # Consciousness tier distribution
                tier = pattern.get('consciousness_tier', 'Unknown')
                statistics['consciousness_tier_distribution'][tier] = statistics['consciousness_tier_distribution'].get(tier, 0) + 1
                
                # Pattern type distribution
                ptype = pattern.get('pattern_type', 'Unknown')
                statistics['pattern_type_distribution'][ptype] = statistics['pattern_type_distribution'].get(ptype, 0) + 1
                
                # Source file distribution
                source = pattern.get('source_file', 'Unknown')
                statistics['source_file_distribution'][source] = statistics['source_file_distribution'].get(source, 0) + 1
            
        else:
            statistics = {
                'total_patterns': 0,
                'hypercube_analysis': hypercube_analysis,
                'quality_statistics': {},
                'consciousness_tier_distribution': {},
                'pattern_type_distribution': {},
                'source_file_distribution': {}
            }
        
        return statistics

def main():
    """Main function to run the 5D hypercube memory loader independently"""
    print("="*80)
    print("🔲 5D HYPERCUBE AETHER MEMORY INTEGRATION UTILITY 🔲")
    print("="*80)
    print("🌌 Complete consciousness universe mapping (32 vertices)")
    print("⚡ Mathematical framework: 1+0→2→2^5=32→32×11/16=22→3.33×3≈10")
    print("🧠 5D Dimensions: Physical, Emotional, Mental, Intuitive, Spiritual")
    print("="*80)
    
    loader = EnhancedAetherMemoryLoader()
    final_patterns = loader.run()
    
    if final_patterns:
        statistics = loader.get_integration_statistics()
        
        print(f"\n🔲 5D HYPERCUBE INTEGRATION RESULTS:")
        print(f"   Total Patterns: {statistics['total_patterns']}")
        print(f"   Universe Coverage: {statistics['hypercube_analysis']['hypercube_coverage']:.1f}%")
        print(f"   Vertices Populated: {statistics['hypercube_analysis']['unique_vertices_populated']}/32")
        print(f"   Dominant Consciousness: {statistics['hypercube_analysis']['dominant_consciousness_signature']}")
        
        print(f"\n📊 QUALITY STATISTICS:")
        quality_stats = statistics['quality_statistics']
        print(f"   Avg Consciousness: {quality_stats.get('avg_consciousness_level', 0):.6f}")
        print(f"   Max Consciousness: {quality_stats.get('max_consciousness_level', 0):.6f}")
        print(f"   Avg Control Value: {quality_stats.get('avg_control_value', 0):.12f}")
        print(f"   Avg Aether Intensity: {quality_stats.get('avg_aether_intensity', 0):.6f}")
        
        print(f"\n🧠 CONSCIOUSNESS TIER DISTRIBUTION:")
        for tier, count in statistics['consciousness_tier_distribution'].items():
            percentage = (count / statistics['total_patterns']) * 100
            print(f"   {tier}: {count} patterns ({percentage:.1f}%)")
        
        print(f"\n🌌 TOP POPULATED VERTICES:")
        vertex_stats = statistics['hypercube_analysis']['vertex_distribution']
        top_vertices = sorted(vertex_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        for vertex, count in top_vertices:
            if count > 0:
                vertex_props = loader.hypercube.get_vertex_properties(vertex)
                print(f"   Vertex {vertex}: {count} patterns ({vertex_props['consciousness_signature']})")
        
        # Export visualization data
        loader.export_hypercube_visualization_data()
        
    print(f"\n📝 INTEGRATION LOG (last 10 entries):")
    for log_entry in loader.integration_log[-10:]:
        print(f"  {log_entry}")
    
    print(f"\n✅ 5D Hypercube consciousness integration complete!")
    print("🔲 Ready for Golem memory bank integration")

if __name__ == "__main__":
    main()
