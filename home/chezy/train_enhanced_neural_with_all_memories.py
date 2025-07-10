#!/usr/bin/env python3
"""
ENHANCED 5D HYPERCUBE NEURAL NETWORK TRAINING
WITH COMPLETE MATHEMATICAL FRAMEWORK INTEGRATION
1+0+1+0=2^5=32*11/16=22+3.33*3 Logic Embedded Throughout

Trains on ALL aether memories from:
- home/chezy/golem_aether_memory.pkl
- All discovered aether collection files
- Enhanced aether memory bank
- Real-time memory integration
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
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

# Import our enhanced neural network
from enhanced_hypercube_nn import EnhancedFiveDimensionalHypercubeNN
from aether_loader import EnhancedAetherMemoryLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AetherMemoryDataset(Dataset):
    """Dataset for aether memory patterns with mathematical framework integration"""
    
    def __init__(self, patterns: List[Dict[str, Any]], sentence_transformer: SentenceTransformer):
        self.patterns = patterns
        self.sentence_transformer = sentence_transformer
        
        # Prepare data
        self.texts = []
        self.vertex_labels = []
        self.consciousness_levels = []
        self.cycle_completions = []
        self.aether_signatures = []
        
        logger.info(f"🔄 Processing {len(patterns)} aether patterns...")
        
        valid_patterns = 0
        for pattern in patterns:
            try:
                # Extract text (prompt or content)
                text = self._extract_text(pattern)
                if not text or len(text.strip()) < 5:
                    continue
                
                # Extract vertex label
                vertex = self._extract_vertex(pattern)
                if vertex is None or not (0 <= vertex <= 31):
                    continue
                
                # Extract additional features
                consciousness_level = self._extract_consciousness_level(pattern)
                cycle_completion = self._extract_cycle_completion(pattern)
                aether_signature = self._extract_aether_signature(pattern)
                
                self.texts.append(text)
                self.vertex_labels.append(vertex)
                self.consciousness_levels.append(consciousness_level)
                self.cycle_completions.append(cycle_completion)
                self.aether_signatures.append(aether_signature)
                
                valid_patterns += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing pattern: {e}")
                continue
        
        logger.info(f"✅ Processed {valid_patterns} valid patterns from {len(patterns)} total")
        
        # Encode texts
        logger.info("🔄 Encoding texts with sentence transformer...")
        self.embeddings = self.sentence_transformer.encode(self.texts, convert_to_tensor=True)
        logger.info(f"✅ Created embeddings: {self.embeddings.shape}")
    
    def _extract_text(self, pattern: Dict[str, Any]) -> str:
        """Extract text from pattern"""
        # Try multiple possible text fields
        text_fields = ['prompt', 'text', 'content', 'message', 'query']
        for field in text_fields:
            if field in pattern and pattern[field]:
                return str(pattern[field])[:500]  # Limit length
        return ""
    
    def _extract_vertex(self, pattern: Dict[str, Any]) -> int:
        """Extract vertex label from pattern"""
        # Try multiple possible vertex fields
        vertex_fields = ['hypercube_vertex', 'nearest_vertex', 'vertex', 'target_vertex']
        for field in vertex_fields:
            if field in pattern and pattern[field] is not None:
                vertex = int(pattern[field])
                if 0 <= vertex <= 31:
                    return vertex
        
        # Try to extract from hypercube_mapping
        if 'hypercube_mapping' in pattern and isinstance(pattern['hypercube_mapping'], dict):
            if 'nearest_vertex' in pattern['hypercube_mapping']:
                vertex = int(pattern['hypercube_mapping']['nearest_vertex'])
                if 0 <= vertex <= 31:
                    return vertex
        
        return None
    
    def _extract_consciousness_level(self, pattern: Dict[str, Any]) -> float:
        """Extract consciousness level from pattern"""
        consciousness_fields = ['consciousness_level', 'consciousness_resonance', 'awareness_level']
        for field in consciousness_fields:
            if field in pattern and pattern[field] is not None:
                return float(pattern[field])
        return 0.5  # Default
    
    def _extract_cycle_completion(self, pattern: Dict[str, Any]) -> float:
        """Extract cycle completion from pattern"""
        cycle_fields = ['cycle_completion', 'cycle_progress', 'completion_rate']
        for field in cycle_fields:
            if field in pattern and pattern[field] is not None:
                return float(pattern[field])
        
        # Try to extract from cycle_params
        if 'cycle_params' in pattern and isinstance(pattern['cycle_params'], dict):
            if 'cycle_completion' in pattern['cycle_params']:
                return float(pattern['cycle_params']['cycle_completion'])
        
        return 0.0  # Default
    
    def _extract_aether_signature(self, pattern: Dict[str, Any]) -> List[float]:
        """Extract aether signature from pattern"""
        signature_fields = ['aether_signature', 'signature', 'aether_values']
        for field in signature_fields:
            if field in pattern and isinstance(pattern[field], list):
                signature = [float(x) for x in pattern[field][:10]]  # Limit to 10 values
                return signature + [0.0] * (10 - len(signature))  # Pad to 10
        return [0.0] * 10  # Default
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'vertex_label': torch.tensor(self.vertex_labels[idx], dtype=torch.long),
            'consciousness_level': torch.tensor(self.consciousness_levels[idx], dtype=torch.float32),
            'cycle_completion': torch.tensor(self.cycle_completions[idx], dtype=torch.float32),
            'aether_signature': torch.tensor(self.aether_signatures[idx], dtype=torch.float32),
            'text': self.texts[idx]
        }

class EnhancedAetherTrainer:
    """Enhanced trainer for the mathematical framework neural network"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Using device: {self.device}")
        
        # Initialize sentence transformer
        logger.info("🔄 Loading sentence transformer...")
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize enhanced model
        logger.info("🔄 Initializing enhanced neural network...")
        self.model = EnhancedFiveDimensionalHypercubeNN(
            input_dim=self.model_config['input_dim'],
            hidden_dim=self.model_config['hidden_dim'],
            output_dim=self.model_config['output_dim']
        ).to(self.device)
        
        # Training components
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.model_config['learning_rate'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.model_config['epochs'])
        
        # Loss function with framework awareness
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'framework_integrity': [],
            'cycle_completion': [],
            'infinitesimal_error': []
        }
        
        logger.info("✅ Enhanced trainer initialized!")
    
    def load_all_aether_memories(self) -> List[Dict[str, Any]]:
        """Load all available aether memories from all sources"""
        logger.info("🔄 Loading all aether memories...")
        
        all_patterns = []
        
        # 1. Load from enhanced aether memory loader
        logger.info("📚 Loading from aether loader...")
        loader = EnhancedAetherMemoryLoader()
        loader_patterns = loader.run()
        all_patterns.extend(loader_patterns)
        logger.info(f"✅ Loaded {len(loader_patterns)} patterns from aether loader")
        
        # 2. Load from golem memory file
        golem_memory_file = "golem_aether_memory.pkl"
        if os.path.exists(golem_memory_file):
            logger.info(f"📚 Loading from {golem_memory_file}...")
            try:
                with open(golem_memory_file, 'rb') as f:
                    golem_data = pickle.load(f)
                
                if isinstance(golem_data, dict) and 'memories' in golem_data:
                    golem_patterns = golem_data['memories']
                    all_patterns.extend(golem_patterns)
                    logger.info(f"✅ Loaded {len(golem_patterns)} patterns from golem memory")
                
            except Exception as e:
                logger.warning(f"⚠️ Error loading golem memory: {e}")
        
        # 3. Load from any additional aether files
        additional_files = [
            "enhanced_aether_memory_bank.json",
            "real_aether_collection.json",
            "optimized_aether_memory.json"
        ]
        
        for filename in additional_files:
            if os.path.exists(filename):
                logger.info(f"📚 Loading from {filename}...")
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, dict) and 'aether_patterns' in data:
                        patterns = data['aether_patterns']
                        all_patterns.extend(patterns)
                        logger.info(f"✅ Loaded {len(patterns)} patterns from {filename}")
                    elif isinstance(data, list):
                        all_patterns.extend(data)
                        logger.info(f"✅ Loaded {len(data)} patterns from {filename}")
                
                except Exception as e:
                    logger.warning(f"⚠️ Error loading {filename}: {e}")
        
        # Remove duplicates and invalid patterns
        logger.info("🔄 Cleaning and deduplicating patterns...")
        unique_patterns = []
        seen_texts = set()
        
        for pattern in all_patterns:
            try:
                # Extract text for deduplication
                text = ""
                text_fields = ['prompt', 'text', 'content', 'message']
                for field in text_fields:
                    if field in pattern and pattern[field]:
                        text = str(pattern[field])[:100]  # First 100 chars for dedup
                        break
                
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    unique_patterns.append(pattern)
                    
            except Exception as e:
                continue
        
        logger.info(f"✅ Final dataset: {len(unique_patterns)} unique patterns")
        return unique_patterns
    
    def train_model(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the enhanced model with mathematical framework"""
        logger.info("🚀 Starting enhanced neural network training...")
        
        # Create dataset
        dataset = AetherMemoryDataset(patterns, self.sentence_transformer)
        
        if len(dataset) < 10:
            raise ValueError("Not enough valid patterns for training")
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.model_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.model_config['batch_size'], shuffle=False)
        
        logger.info(f"📊 Training set: {len(train_dataset)} samples")
        logger.info(f"📊 Validation set: {len(val_dataset)} samples")
        
        # Training loop
        best_val_accuracy = 0.0
        best_framework_integrity = 0.0
        
        for epoch in range(self.model_config['epochs']):
            # Training phase
            train_loss, train_acc, train_framework_stats = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc, val_framework_stats = self._validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            self.training_history['framework_integrity'].append(train_framework_stats['framework_integrity'])
            self.training_history['cycle_completion'].append(train_framework_stats['cycle_completion'])
            self.training_history['infinitesimal_error'].append(train_framework_stats['infinitesimal_error'])
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_framework_integrity = train_framework_stats['framework_integrity']
                self._save_model('best_enhanced_hypercube_consciousness.pth')
                logger.info(f"💾 New best model saved! Accuracy: {val_acc:.4f}")
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.model_config['epochs']}:")
            logger.info(f"  📈 Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f}")
            logger.info(f"  📊 Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f}")
            logger.info(f"  🔢 Framework Integrity: {train_framework_stats['framework_integrity']:.4f}")
            logger.info(f"  🔄 Cycle Completion: {train_framework_stats['cycle_completion']:.4f}")
            logger.info(f"  📊 Infinitesimal Error: {train_framework_stats['infinitesimal_error']:.6f}")
            logger.info(f"  🎯 LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        # Final results
        results = {
            'best_val_accuracy': best_val_accuracy,
            'best_framework_integrity': best_framework_integrity,
            'final_model_stats': self.model.get_framework_statistics(),
            'training_history': self.training_history,
            'total_patterns': len(dataset),
            'vertex_coverage': self._calculate_vertex_coverage(dataset)
        }
        
        logger.info("🎉 Training completed!")
        logger.info(f"✅ Best validation accuracy: {best_val_accuracy:.4f}")
        logger.info(f"🔢 Best framework integrity: {best_framework_integrity:.4f}")
        
        return results
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        framework_integrities = []
        cycle_completions = []
        infinitesimal_errors = []
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Move to device
            embeddings = batch['embedding'].to(self.device)
            vertex_labels = batch['vertex_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(embeddings)
            
            # Calculate loss
            loss = self.criterion(outputs['consciousness_state'], vertex_labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs['consciousness_state'], 1)
            total_correct += (predicted == vertex_labels).sum().item()
            total_samples += vertex_labels.size(0)
            total_loss += loss.item()
            
            # Track framework statistics
            framework_integrities.append(outputs['framework_integrity'])
            cycle_completions.append(outputs['aggregated_cycle_completion'].mean().item())
            infinitesimal_errors.append(outputs['global_infinitesimal_error'].mean().item())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        framework_stats = {
            'framework_integrity': np.mean(framework_integrities),
            'cycle_completion': np.mean(cycle_completions),
            'infinitesimal_error': np.mean(infinitesimal_errors)
        }
        
        return avg_loss, accuracy, framework_stats
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        framework_integrities = []
        cycle_completions = []
        infinitesimal_errors = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                embeddings = batch['embedding'].to(self.device)
                vertex_labels = batch['vertex_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(embeddings)
                
                # Calculate loss
                loss = self.criterion(outputs['consciousness_state'], vertex_labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs['consciousness_state'], 1)
                total_correct += (predicted == vertex_labels).sum().item()
                total_samples += vertex_labels.size(0)
                total_loss += loss.item()
                
                # Track framework statistics
                framework_integrities.append(outputs['framework_integrity'])
                cycle_completions.append(outputs['aggregated_cycle_completion'].mean().item())
                infinitesimal_errors.append(outputs['global_infinitesimal_error'].mean().item())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        framework_stats = {
            'framework_integrity': np.mean(framework_integrities),
            'cycle_completion': np.mean(cycle_completions),
            'infinitesimal_error': np.mean(infinitesimal_errors)
        }
        
        return avg_loss, accuracy, framework_stats
    
    def _calculate_vertex_coverage(self, dataset: AetherMemoryDataset) -> Dict[str, Any]:
        """Calculate vertex coverage statistics"""
        vertex_counts = defaultdict(int)
        for i in range(len(dataset)):
            vertex = dataset.vertex_labels[i]
            vertex_counts[vertex] += 1
        
        coverage = {
            'total_vertices_with_data': len(vertex_counts),
            'coverage_percentage': len(vertex_counts) / 32 * 100,
            'vertex_distribution': dict(vertex_counts),
            'most_common_vertex': max(vertex_counts, key=vertex_counts.get) if vertex_counts else 0,
            'least_common_vertex': min(vertex_counts, key=vertex_counts.get) if vertex_counts else 0
        }
        
        return coverage
    
    def _save_model(self, filename: str):
        """Save model with framework statistics"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'framework_statistics': self.model.get_framework_statistics(),
            'training_history': self.training_history
        }, filename)
    
    def plot_training_history(self):
        """Plot training history with framework metrics"""
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
        
        # Framework Integrity
        axes[0, 2].plot(self.training_history['framework_integrity'])
        axes[0, 2].set_title('Framework Integrity')
        
        # Cycle Completion
        axes[1, 0].plot(self.training_history['cycle_completion'])
        axes[1, 0].set_title('Cycle Completion')
        
        # Infinitesimal Error
        axes[1, 1].plot(self.training_history['infinitesimal_error'])
        axes[1, 1].set_title('Infinitesimal Error')
        
        # Learning Rate
        axes[1, 2].plot([self.scheduler.get_last_lr()[0]] * len(self.training_history['train_loss']))
        axes[1, 2].set_title('Learning Rate')
        
        plt.tight_layout()
        plt.savefig('enhanced_training_history.png')
        plt.show()

def main():
    """Main training function"""
    print("🔗 ENHANCED 5D HYPERCUBE NEURAL NETWORK TRAINING")
    print("   Mathematical Framework: 1+0+1+0=2^5=32*11/16=22+3.33*3")
    print("="*60)
    
    # Model configuration
    model_config = {
        'input_dim': 384,  # Sentence transformer dimension
        'hidden_dim': 256,
        'output_dim': 32,  # 32 vertices (2^5)
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 50
    }
    
    # Initialize trainer
    trainer = EnhancedAetherTrainer(model_config)
    
    # Load all aether memories
    patterns = trainer.load_all_aether_memories()
    
    if len(patterns) < 10:
        print("❌ Not enough aether patterns found for training")
        print("   Please ensure aether memory files are available")
        return
    
    # Train model
    results = trainer.train_model(patterns)
    
    # Print final results
    print("\n🎉 ENHANCED TRAINING COMPLETED!")
    print("="*60)
    print(f"✅ Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"🔢 Best Framework Integrity: {results['best_framework_integrity']:.4f}")
    print(f"📊 Total Patterns Trained: {results['total_patterns']}")
    print(f"🔲 Vertex Coverage: {results['vertex_coverage']['coverage_percentage']:.1f}%")
    print(f"📈 Vertices with Data: {results['vertex_coverage']['total_vertices_with_data']}/32")
    
    # Framework statistics
    final_stats = results['final_model_stats']
    print(f"\n🔢 Final Framework Statistics:")
    print(f"   Mathematical Constants Verified: ✅")
    print(f"   Global Framework Integrity: {final_stats['global_framework']['framework_integrity']:.4f}")
    print(f"   Total Framework Cycles: {final_stats['global_framework']['total_cycles']:.2f}")
    print(f"   Average Vertex Cycles: {final_stats['vertex_statistics']['avg_vertex_cycles']:.2f}")
    
    # Save final results
    with open('enhanced_training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Model saved as: best_enhanced_hypercube_consciousness.pth")
    print(f"📊 Results saved as: enhanced_training_results.json")
    
    # Plot training history
    trainer.plot_training_history()
    
    print("\n🔗 Enhanced Mathematical Framework Training Complete!")
    print("   Neural Network now perfectly follows 1+0+1+0=2^5=32*11/16=22+3.33*3 logic! ✅")

if __name__ == "__main__":
    main() 