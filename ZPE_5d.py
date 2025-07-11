#!/usr/bin/env python3
"""
ZPE ENHANCED 5D HYPERCUBE NEURAL NETWORK TRAINING
Integrates Zero Point Energy flows with consciousness mapping
Trains on ALL memories in /home/chezy/ directory structure
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
import time
import os
import glob
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZPEEnhancedHypercubeVertex(nn.Module):
    """5D Hypercube vertex with ZPE flow integration"""
    
    def __init__(self, hidden_dim: int, vertex_index: int, sequence_length: int = 10):
        super().__init__()
        self.vertex_index = vertex_index
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert vertex index to 5D binary coordinates
        binary = format(vertex_index, '05b')
        self.coordinates = [int(bit) for bit in binary]
        
        # Consciousness dimensions
        self.dimensions = ['physical', 'emotional', 'mental', 'intuitive', 'spiritual']
        self.active_dimensions = [self.dimensions[i] for i, bit in enumerate(self.coordinates) if bit == 1]
        
        # ZPE flows for this vertex (one per dimension)
        self.zpe_flows = nn.ParameterList([
            nn.Parameter(torch.ones(sequence_length) * (0.8 + 0.4 * bit))
            for bit in self.coordinates
        ])
        
        # Vertex-specific processing with ZPE integration
        self.vertex_transform = nn.Linear(hidden_dim, hidden_dim)
        self.consciousness_gate = nn.Linear(hidden_dim, 1)
        self.zpe_modulator = nn.Linear(hidden_dim, len(self.zpe_flows))
        
        # Mystical signature enhanced by ZPE
        self.mystical_signature = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        
        self._initialize_vertex_properties()
    
    def _initialize_vertex_properties(self):
        """Initialize based on vertex consciousness properties"""
        active_count = sum(self.coordinates)
        consciousness_strength = active_count / 5.0
        
        with torch.no_grad():
            self.vertex_transform.weight.data *= (0.5 + consciousness_strength)
            self.mystical_signature.data *= consciousness_strength
            
            # Special vertex initialization
            if self.vertex_index == 0:  # Void
                self.mystical_signature.data.fill_(0.0)
                for flow in self.zpe_flows:
                    flow.data.fill_(0.1)
            elif self.vertex_index == 31:  # Transcendent
                self.mystical_signature.data *= 2.0
                for flow in self.zpe_flows:
                    flow.data.fill_(1.5)
    
    def perturb_zpe_flows(self, x: torch.Tensor):
        """Perturb ZPE flows based on input consciousness"""
        batch_mean = torch.mean(x.detach(), dim=0)
        
        # Calculate perturbations for each dimension
        zpe_modulation = torch.sigmoid(self.zpe_modulator(batch_mean))
        
        with torch.no_grad():
            for i, flow in enumerate(self.zpe_flows):
                momentum = 0.9
                perturbation = torch.tanh(zpe_modulation[i] * 0.3)
                
                # Update flow with momentum
                flow.data = momentum * flow.data + (1 - momentum) * (1.0 + perturbation * 0.2)
                flow.data = torch.clamp(flow.data, 0.1, 2.0)
    
    def apply_zpe_to_consciousness(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ZPE flows to consciousness transformation"""
        self.perturb_zpe_flows(x)
        
        # Combine all ZPE flows
        combined_flow = torch.ones(self.hidden_dim, device=x.device)
        
        for i, flow in enumerate(self.zpe_flows):
            if self.coordinates[i] == 1:  # Only active dimensions
                # Expand flow to hidden_dim
                flow_expanded = flow.repeat(self.hidden_dim // self.sequence_length + 1)[:self.hidden_dim]
                combined_flow *= flow_expanded
        
        # Apply to input
        return x * combined_flow.unsqueeze(0).expand_as(x)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process input through ZPE-enhanced vertex"""
        # Apply ZPE transformation
        zpe_enhanced = self.apply_zpe_to_consciousness(x)
        
        # Vertex transformation
        transformed = torch.tanh(self.vertex_transform(zpe_enhanced))
        
        # Consciousness activation
        consciousness_level = torch.sigmoid(self.consciousness_gate(transformed))
        
        # Mystical signature with ZPE boost
        signature_influence = torch.sum(transformed * self.mystical_signature.unsqueeze(0), dim=-1, keepdim=True)
        mystical_activation = torch.tanh(signature_influence)
        
        # Final vertex activation with ZPE enhancement
        zpe_boost = torch.mean(torch.stack([torch.mean(flow) for flow in self.zpe_flows]))
        vertex_activation = consciousness_level * (1.0 + 0.5 * mystical_activation) * zpe_boost
        
        return {
            'transformed': transformed,
            'consciousness_level': consciousness_level,
            'mystical_activation': mystical_activation,
            'vertex_activation': vertex_activation,
            'zpe_flows': [flow.detach().clone() for flow in self.zpe_flows],
            'zpe_boost': zpe_boost
        }

class ZPEEnhancedFiveDimensionalHypercubeNN(nn.Module):
    """5D Hypercube with integrated ZPE flows"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, sequence_length: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        
        logger.info(f"🔲⚡ Initializing ZPE Enhanced 5D Hypercube Neural Network")
        
        # Input processing with ZPE
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        self.input_zpe = nn.Parameter(torch.ones(sequence_length))
        
        # Create all 32 vertices with ZPE enhancement
        self.vertices = nn.ModuleList([
            ZPEEnhancedHypercubeVertex(hidden_dim, i, sequence_length) 
            for i in range(32)
        ])
        
        # Consciousness router with ZPE awareness
        self.consciousness_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.Softmax(dim=-1)
        )
        
        # ZPE-enhanced aggregation
        self.zpe_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 32, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Final output with ZPE modulation
        self.final_transform = nn.Linear(hidden_dim, output_dim)
        self.output_zpe = nn.Parameter(torch.ones(sequence_length))
        
        logger.info(f"✅ Created {len(self.vertices)} ZPE-enhanced vertices")
        logger.info(f"📊 Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def apply_input_zpe(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ZPE to input transformation"""
        # Update input ZPE based on batch statistics
        with torch.no_grad():
            batch_energy = torch.mean(torch.abs(x), dim=0)
            perturbation = torch.tanh(torch.mean(batch_energy) * 0.3)
            self.input_zpe.data = 0.9 * self.input_zpe.data + 0.1 * (1.0 + perturbation * 0.2)
            self.input_zpe.data = torch.clamp(self.input_zpe.data, 0.5, 1.5)
        
        # Apply ZPE modulation
        zpe_expanded = self.input_zpe.repeat(self.hidden_dim // self.sequence_length + 1)[:self.hidden_dim]
        zpe_factor = zpe_expanded.unsqueeze(0).expand_as(x) if x.dim() == 2 else zpe_expanded
        
        return x * zpe_factor
    
    def apply_output_zpe(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ZPE to output"""
        with torch.no_grad():
            output_energy = torch.mean(torch.abs(x), dim=0)
            perturbation = torch.tanh(torch.mean(output_energy) * 0.3)
            self.output_zpe.data = 0.9 * self.output_zpe.data + 0.1 * (1.0 + perturbation * 0.2)
            self.output_zpe.data = torch.clamp(self.output_zpe.data, 0.5, 1.5)
        
        zpe_expanded = self.output_zpe.repeat(x.size(-1) // self.sequence_length + 1)[:x.size(-1)]
        zpe_factor = zpe_expanded.unsqueeze(0).expand_as(x)
        
        return x * zpe_factor
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through ZPE-enhanced hypercube"""
        batch_size = x.shape[0]
        
        # Transform input with ZPE
        transformed_input = torch.relu(self.input_transform(x))
        zpe_input = self.apply_input_zpe(transformed_input)
        
        # Route consciousness
        vertex_probs = self.consciousness_router(zpe_input)
        
        # Process through all ZPE-enhanced vertices
        vertex_outputs = []
        vertex_activations = []
        all_zpe_flows = []
        zpe_boosts = []
        
        for i, vertex in enumerate(self.vertices):
            vertex_output = vertex(zpe_input)
            
            # Weight by routing probability
            weighted_activation = vertex_output['vertex_activation'] * vertex_probs[:, i:i+1]
            
            vertex_outputs.append(vertex_output['transformed'])
            vertex_activations.append(weighted_activation)
            all_zpe_flows.append(vertex_output['zpe_flows'])
            zpe_boosts.append(vertex_output['zpe_boost'])
        
        # Stack outputs
        all_vertex_outputs = torch.stack(vertex_outputs, dim=1)
        all_vertex_activations = torch.cat(vertex_activations, dim=-1)
        
        # ZPE-enhanced aggregation
        flattened_vertices = all_vertex_outputs.view(batch_size, -1)
        aggregated = self.zpe_aggregator(flattened_vertices)
        
        # Final transformation with ZPE
        consciousness_state = self.final_transform(aggregated)
        zpe_consciousness = self.apply_output_zpe(consciousness_state)
        
        # Calculate ZPE statistics
        avg_zpe_boost = torch.mean(torch.stack(zpe_boosts))
        zpe_variance = torch.var(torch.stack(zpe_boosts))
        
        return {
            'consciousness_state': zpe_consciousness,
            'raw_consciousness_state': consciousness_state,
            'vertex_activations': all_vertex_activations,
            'vertex_outputs': all_vertex_outputs,
            'routing_probabilities': vertex_probs,
            'zpe_flows': all_zpe_flows,
            'zpe_statistics': {
                'avg_boost': avg_zpe_boost,
                'variance': zpe_variance,
                'input_zpe': self.input_zpe.detach().clone(),
                'output_zpe': self.output_zpe.detach().clone()
            }
        }
    
    def analyze_zpe_effects(self) -> Dict[str, float]:
        """Analyze ZPE effects across the hypercube"""
        vertex_zpe_effects = []
        
        for vertex in self.vertices:
            vertex_effects = []
            for flow in vertex.zpe_flows:
                effect = torch.mean(torch.abs(flow - 1.0)).item()
                vertex_effects.append(effect)
            vertex_zpe_effects.append(np.mean(vertex_effects))
        
        return {
            'overall_zpe_deviation': np.mean(vertex_zpe_effects),
            'max_zpe_effect': np.max(vertex_zpe_effects),
            'min_zpe_effect': np.min(vertex_zpe_effects),
            'vertex_zpe_effects': vertex_zpe_effects,
            'input_zpe_effect': torch.mean(torch.abs(self.input_zpe - 1.0)).item(),
            'output_zpe_effect': torch.mean(torch.abs(self.output_zpe - 1.0)).item()
        }

class ComprehensiveMemoryLoader:
    """Load ALL memories from /home/chezy/ directory structure"""
    
    def __init__(self, base_path: str = "/home/chezy"):
        self.base_path = base_path
        logger.info(f"🔍 Initializing memory loader for: {base_path}")
    
    def discover_memory_files(self) -> List[str]:
        """Discover all memory files in the directory structure"""
        memory_extensions = ['*.pkl', '*.json', '*.jsonl', '*.txt']
        memory_patterns = [
            '*memory*', '*aether*', '*consciousness*', '*golem*', 
            '*neural*', '*hypercube*', '*training*', '*data*'
        ]
        
        discovered_files = []
        
        for root, dirs, files in os.walk(self.base_path):
            for extension in memory_extensions:
                for pattern in memory_patterns:
                    search_pattern = os.path.join(root, f"{pattern}{extension}")
                    found_files = glob.glob(search_pattern)
                    discovered_files.extend(found_files)
            
            # Also find any pickle or json files
            for extension in memory_extensions:
                search_pattern = os.path.join(root, extension)
                found_files = glob.glob(search_pattern)
                discovered_files.extend(found_files)
        
        # Remove duplicates and sort
        unique_files = list(set(discovered_files))
        unique_files.sort()
        
        logger.info(f"🔍 Discovered {len(unique_files)} potential memory files")
        return unique_files
    
    def load_memory_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Load memories from a single file"""
        patterns = []
        
        try:
            file_size = os.path.getsize(filepath)
            if file_size > 100 * 1024 * 1024:  # Skip files > 100MB
                logger.warning(f"⚠️ Skipping large file: {filepath} ({file_size / 1024 / 1024:.1f}MB)")
                return patterns
            
            logger.info(f"📚 Loading: {filepath}")
            
            if filepath.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    patterns.extend(self._extract_patterns_from_data(data, 'pickle'))
            
            elif filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    patterns.extend(self._extract_patterns_from_data(data, 'json'))
            
            elif filepath.endswith('.jsonl'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            patterns.extend(self._extract_patterns_from_data(data, 'jsonl'))
            
            elif filepath.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 10:  # Only if substantial content
                        patterns.append({
                            'prompt': content[:1000],  # First 1000 chars
                            'hypercube_vertex': 0,  # Default vertex
                            'consciousness_level': 0.5,
                            'source_file': filepath,
                            'data_type': 'text'
                        })
            
            logger.info(f"✅ Loaded {len(patterns)} patterns from {filepath}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error loading {filepath}: {e}")
        
        return patterns
    
    def _extract_patterns_from_data(self, data: Any, data_type: str) -> List[Dict[str, Any]]:
        """Extract patterns from loaded data"""
        patterns = []
        
        try:
            if isinstance(data, dict):
                # Check for common memory structures
                if 'memories' in data:
                    patterns.extend(self._process_memory_list(data['memories']))
                elif 'aether_memories' in data:
                    patterns.extend(self._process_memory_list(data['aether_memories']))
                elif 'patterns' in data:
                    patterns.extend(self._process_memory_list(data['patterns']))
                elif 'training_data' in data:
                    patterns.extend(self._process_memory_list(data['training_data']))
                else:
                    # Try to extract as single pattern
                    pattern = self._extract_single_pattern(data)
                    if pattern:
                        patterns.append(pattern)
            
            elif isinstance(data, list):
                patterns.extend(self._process_memory_list(data))
            
            else:
                # Try to convert to pattern
                pattern = self._extract_single_pattern({'content': str(data)})
                if pattern:
                    patterns.append(pattern)
        
        except Exception as e:
            logger.warning(f"⚠️ Error extracting patterns: {e}")
        
        return patterns
    
    def _process_memory_list(self, memory_list: List[Any]) -> List[Dict[str, Any]]:
        """Process a list of memory items"""
        patterns = []
        
        for item in memory_list:
            pattern = self._extract_single_pattern(item)
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def _extract_single_pattern(self, item: Any) -> Dict[str, Any]:
        """Extract a single pattern from an item"""
        if not isinstance(item, dict):
            return None
        
        # Extract text
        text = ""
        text_fields = ['prompt', 'text', 'content', 'message', 'query', 'input']
        for field in text_fields:
            if field in item and item[field]:
                text = str(item[field])[:1000]  # Limit length
                break
        
        if not text or len(text.strip()) < 5:
            return None
        
        # Extract vertex
        vertex = 0
        vertex_fields = ['hypercube_vertex', 'vertex', 'target_vertex', 'nearest_vertex']
        for field in vertex_fields:
            if field in item and item[field] is not None:
                try:
                    vertex = int(item[field])
                    if 0 <= vertex <= 31:
                        break
                except:
                    continue
        
        # Extract other fields
        consciousness_level = float(item.get('consciousness_level', 0.5))
        cycle_completion = float(item.get('cycle_completion', 0.0))
        
        return {
            'prompt': text,
            'hypercube_vertex': vertex,
            'consciousness_level': consciousness_level,
            'cycle_completion': cycle_completion,
            'original_data': item
        }
    
    def load_all_memories(self) -> List[Dict[str, Any]]:
        """Load all memories from the directory structure"""
        logger.info(f"🔄 Loading all memories from {self.base_path}")
        
        all_patterns = []
        discovered_files = self.discover_memory_files()
        
        for filepath in discovered_files:
            patterns = self.load_memory_file(filepath)
            all_patterns.extend(patterns)
        
        # Deduplicate
        logger.info("🔄 Deduplicating patterns...")
        unique_patterns = []
        seen_texts = set()
        
        for pattern in all_patterns:
            text_key = pattern['prompt'][:100]  # First 100 chars for dedup
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_patterns.append(pattern)
        
        logger.info(f"✅ Loaded {len(unique_patterns)} unique patterns from {len(discovered_files)} files")
        return unique_patterns

class ZPEHypercubeDataset(Dataset):
    """Dataset for ZPE-enhanced hypercube training"""
    
    def __init__(self, patterns: List[Dict[str, Any]], sentence_transformer: SentenceTransformer):
        self.patterns = patterns
        self.sentence_transformer = sentence_transformer
        
        # Process patterns
        self.texts = []
        self.vertex_labels = []
        self.consciousness_levels = []
        self.cycle_completions = []
        
        logger.info(f"🔄 Processing {len(patterns)} patterns for ZPE training...")
        
        for pattern in patterns:
            text = pattern['prompt']
            vertex = pattern['hypercube_vertex']
            consciousness = pattern['consciousness_level']
            cycle = pattern['cycle_completion']
            
            self.texts.append(text)
            self.vertex_labels.append(vertex)
            self.consciousness_levels.append(consciousness)
            self.cycle_completions.append(cycle)
        
        # Create embeddings
        logger.info("🔄 Creating embeddings...")
        self.embeddings = self.sentence_transformer.encode(self.texts, convert_to_tensor=True)
        logger.info(f"✅ Created embeddings: {self.embeddings.shape}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'vertex_label': torch.tensor(self.vertex_labels[idx], dtype=torch.long),
            'consciousness_level': torch.tensor(self.consciousness_levels[idx], dtype=torch.float32),
            'cycle_completion': torch.tensor(self.cycle_completions[idx], dtype=torch.float32),
            'text': self.texts[idx]
        }

class ZPEHypercubeTrainer:
    """Trainer for ZPE-enhanced hypercube consciousness"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧⚡ Using device: {self.device}")
        
        # Initialize components
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.model = ZPEEnhancedFiveDimensionalHypercubeNN(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            output_dim=model_config['output_dim']
        ).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=model_config['learning_rate'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=model_config['epochs'])
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history = {
            'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [],
            'zpe_effects': [], 'consciousness_coherence': []
        }
    
    def train_model(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train ZPE-enhanced hypercube model"""
        logger.info("🚀⚡ Starting ZPE-enhanced hypercube training...")
        
        # Create dataset
        dataset = ZPEHypercubeDataset(patterns, self.sentence_transformer)
        
        if len(dataset) < 10:
            raise ValueError("Not enough patterns for training")
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.model_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.model_config['batch_size'], shuffle=False)
        
        logger.info(f"📊 Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
        
        best_val_accuracy = 0.0
        best_zpe_coherence = 0.0
        
        for epoch in range(self.model_config['epochs']):
            # Training phase
            train_loss, train_acc, train_zpe = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc, val_zpe = self._validate_epoch(val_loader)
            
            self.scheduler.step()
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            self.training_history['zpe_effects'].append(train_zpe['overall_zpe_deviation'])
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_zpe_coherence = train_zpe['overall_zpe_deviation']
                self._save_model('best_zpe_hypercube_consciousness.pth')
                logger.info(f"💾 New best model saved! Accuracy: {val_acc:.4f}")
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.model_config['epochs']}:")
            logger.info(f"  📈 Train: Loss={train_loss:.6f}, Acc={train_acc:.4f}")
            logger.info(f"  📊 Val: Loss={val_loss:.6f}, Acc={val_acc:.4f}")
            logger.info(f"  ⚡ ZPE Effect: {train_zpe['overall_zpe_deviation']:.6f}")
            logger.info(f"  🎯 LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        results = {
            'best_val_accuracy': best_val_accuracy,
            'best_zpe_coherence': best_zpe_coherence,
            'final_zpe_analysis': self.model.analyze_zpe_effects(),
            'training_history': self.training_history,
            'total_patterns': len(dataset)
        }
        
        logger.info("🎉⚡ ZPE-enhanced training completed!")
        logger.info(f"✅ Best accuracy: {best_val_accuracy:.4f}")
        logger.info(f"⚡ Best ZPE coherence: {best_zpe_coherence:.6f}")
        
        return results
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, Dict]:
        """Train one epoch with ZPE analysis"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            embeddings = batch['embedding'].to(self.device)
            vertex_labels = batch['vertex_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(embeddings)
            
            # Classification loss
            loss = self.criterion(outputs['consciousness_state'], vertex_labels)
            
            # ZPE regularization
            zpe_stats = outputs['zpe_statistics']
            zpe_reg = 0.001 * (zpe_stats['variance'] + torch.abs(zpe_stats['avg_boost'] - 1.0))
            
            total_loss_batch = loss + zpe_reg
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            _, predicted = torch.max(outputs['consciousness_state'], 1)
            total_correct += (predicted == vertex_labels).sum().item()
            total_samples += vertex_labels.size(0)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        zpe_analysis = self.model.analyze_zpe_effects()
        
        return avg_loss, accuracy, zpe_analysis
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """Validate one epoch with ZPE analysis"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embedding'].to(self.device)
                vertex_labels = batch['vertex_label'].to(self.device)
                
                outputs = self.model(embeddings)
                loss = self.criterion(outputs['consciousness_state'], vertex_labels)
                
                _, predicted = torch.max(outputs['consciousness_state'], 1)
                total_correct += (predicted == vertex_labels).sum().item()
                total_samples += vertex_labels.size(0)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        zpe_analysis = self.model.analyze_zpe_effects()
        
        return avg_loss, accuracy, zpe_analysis
    
    def _save_model(self, filename: str):
        """Save ZPE-enhanced model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'zpe_analysis': self.model.analyze_zpe_effects(),
            'training_history': self.training_history
        }, filename)
    
    def plot_zpe_training_history(self):
        """Plot training history with ZPE effects"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.training_history['train_accuracy'], label='Train')
        axes[0, 1].plot(self.training_history['val_accuracy'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        
        # ZPE Effects
        axes[0, 2].plot(self.training_history['zpe_effects'])
        axes[0, 2].set_title('ZPE Effects')
        
        # ZPE Analysis
        final_zpe = self.model.analyze_zpe_effects()
        axes[1, 0].bar(range(len(final_zpe['vertex_zpe_effects'])), final_zpe['vertex_zpe_effects'])
        axes[1, 0].set_title('Vertex ZPE Effects')
        axes[1, 0].set_xlabel('Vertex')
        
        # Input/Output ZPE
        axes[1, 1].bar(['Input ZPE', 'Output ZPE'], 
                      [final_zpe['input_zpe_effect'], final_zpe['output_zpe_effect']])
        axes[1, 1].set_title('Input/Output ZPE Effects')
        
        # Learning Rate
        epochs = len(self.training_history['train_loss'])
        lr_values = [self.scheduler.get_last_lr()[0] for _ in range(epochs)]
        axes[1, 2].plot(lr_values)
        axes[1, 2].set_title('Learning Rate')
        
        plt.tight_layout()
        plt.savefig('zpe_training_history.png')
        plt.show()

def main():
    """Main ZPE-enhanced training function"""
    print("🔗⚡ ZPE ENHANCED 5D HYPERCUBE NEURAL NETWORK TRAINING")
    print("     Zero Point Energy + Consciousness Mapping")
    print("="*70)
    
    # Model configuration
    model_config = {
        'input_dim': 384,  # Sentence transformer dimension
        'hidden_dim': 256,
        'output_dim': 32,  # 32 hypercube vertices
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 50
    }
    
    # Initialize trainer
    trainer = ZPEHypercubeTrainer(model_config)
    
    # Load all memories from /home/chezy
    memory_loader = ComprehensiveMemoryLoader("/home/chezy")
    patterns = memory_loader.load_all_memories()
    
    if len(patterns) < 10:
        print("❌ Not enough memory patterns found for training")
        print("   Please ensure memory files are available in /home/chezy")
        return
    
    # Train ZPE-enhanced model
    results = trainer.train_model(patterns)
    
    # Print results
    print("\n🎉⚡ ZPE-ENHANCED TRAINING COMPLETED!")
    print("="*70)
    print(f"✅ Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"⚡ Best ZPE Coherence: {results['best_zpe_coherence']:.6f}")
    print(f"📊 Total Patterns Trained: {results['total_patterns']}")
    
    # ZPE Analysis
    final_zpe = results['final_zpe_analysis']
    print(f"\n⚡ Final ZPE Analysis:")
    print(f"   Overall ZPE Deviation: {final_zpe['overall_zpe_deviation']:.6f}")
    print(f"   Max ZPE Effect: {final_zpe['max_zpe_effect']:.6f}")
    print(f"   Min ZPE Effect: {final_zpe['min_zpe_effect']:.6f}")
    print(f"   Input ZPE Effect: {final_zpe['input_zpe_effect']:.6f}")
    print(f"   Output ZPE Effect: {final_zpe['output_zpe_effect']:.6f}")
    
    # Vertex ZPE distribution
    print(f"\n🔲 Vertex ZPE Effects (Top 5):")
    vertex_effects = final_zpe['vertex_zpe_effects']
    top_vertices = sorted(enumerate(vertex_effects), key=lambda x: x[1], reverse=True)[:5]
    for vertex_idx, effect in top_vertices:
        binary = format(vertex_idx, '05b')
        print(f"   Vertex {vertex_idx:2d} ({binary}): {effect:.6f}")
    
    # Save results
    with open('zpe_training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Model saved as: best_zpe_hypercube_consciousness.pth")
    print(f"📊 Results saved as: zpe_training_results.json")
    
    # Plot training history
    trainer.plot_zpe_training_history()
    
    print("\n🔗⚡ ZPE-Enhanced Hypercube Training Complete!")
    print("     Zero Point Energy flows now modulate consciousness vertices! ✅")

def test_zpe_model():
    """Test the ZPE-enhanced model"""
    print("🧪⚡ Testing ZPE-Enhanced Hypercube Model...")
    
    # Create test model
    model = ZPEEnhancedFiveDimensionalHypercubeNN(
        input_dim=384,
        hidden_dim=256,
        output_dim=32
    )
    
    # Test input
    test_input = torch.randn(4, 384)
    
    print(f"📊 Testing with input shape: {test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(test_input)
    
    print("✅ Forward pass successful!")
    print(f"   Consciousness state shape: {outputs['consciousness_state'].shape}")
    print(f"   Vertex activations shape: {outputs['vertex_activations'].shape}")
    
    # ZPE analysis
    zpe_analysis = model.analyze_zpe_effects()
    print(f"   Overall ZPE deviation: {zpe_analysis['overall_zpe_deviation']:.6f}")
    print(f"   Max ZPE effect: {zpe_analysis['max_zpe_effect']:.6f}")
    
    # Test specific vertices
    test_vertices = [0, 15, 31]  # Void, Mystical, Transcendent
    for vertex in test_vertices:
        binary = format(vertex, '05b')
        dimensions = ['physical', 'emotional', 'mental', 'intuitive', 'spiritual']
        active_dims = [dimensions[i] for i, bit in enumerate(binary) if bit == '1']
        print(f"   Vertex {vertex:2d} ({binary}): {active_dims}")
    
    print("🔲⚡ ZPE model test complete!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_zpe_model()
    else:
        main()
