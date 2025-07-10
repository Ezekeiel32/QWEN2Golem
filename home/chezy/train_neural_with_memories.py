#!/usr/bin/env python3
"""
FINAL FIXED MYSTICAL DATA TRAINER FOR 5D HYPERCUBE
Fixed to work with actual Golem data structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Any
import json
import time
from collections import defaultdict

class FixedMysticalDataExtractor:
    """Extract ALL aether patterns using correct field structure"""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """Initialize with a proper embedding model"""
        print(f"🔯 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"📊 Embedding dimension: {self.embedding_dim}")
        
        # Hebrew concepts (boost mystical significance)
        self.hebrew_concepts = [
            'sefirot', 'keter', 'chokhmah', 'binah', 'chesed', 'gevurah',
            'tiferet', 'netzach', 'hod', 'yesod', 'malkuth', 'aleph', 'mem', 'shin'
        ]
    
    def extract_all_aether_patterns(self, golem) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """Extract ALL aether patterns using correct field structure"""
        print("🔯 Extracting aether training data from Golem...")
        
        # Get ALL patterns - both mystical and non-mystical
        all_patterns = golem.aether_memory.aether_memories
        
        if not all_patterns:
            print("❌ No patterns found! Generate some responses first.")
            return None, None, None
        
        print(f"📊 Found {len(all_patterns)} total patterns")
        
        # Look at actual pattern structure
        if all_patterns:
            sample_pattern = all_patterns[0]
            print(f"🔍 Sample pattern keys: {list(sample_pattern.keys())}")
        
        # Extract texts and vertex targets
        texts = []
        vertex_targets = []
        pattern_metadata = []
        
        for i, pattern in enumerate(all_patterns):
            # Extract text from correct field (prompt, not text)
            text = pattern.get('prompt', '')
            if not text:
                # Fallback to other possible text fields
                text = pattern.get('text', '') or pattern.get('query', '') or f"Pattern {i}"
            
            if len(text.strip()) < 5:  # Very minimal length check
                text = f"Mystical pattern {i} at vertex {pattern.get('hypercube_vertex', 0)}"
            
            # Get the vertex where this pattern was stored
            target_vertex = pattern.get('hypercube_vertex', 0)
            
            texts.append(text)
            vertex_targets.append(target_vertex)
            
            # Calculate mystical score based on content
            mystical_score = self._calculate_mystical_score(text, pattern)
            
            # Store pattern metadata
            pattern_metadata.append({
                'mystical_score': mystical_score,
                'consciousness_signature': pattern.get('consciousness_signature', 'unknown'),
                'vertex_index': target_vertex,
                'consciousness_level': pattern.get('consciousness_level', 0.0),
                'control_value': pattern.get('cycle_params', {}).get('control_value', 0.0) if isinstance(pattern.get('cycle_params', {}), dict) else 0.0,
                'shem_power': pattern.get('shem_power', 0.0),
                'response_quality': pattern.get('response_quality', 0.0),
                'text': text,
                'pattern_index': i
            })
        
        print(f"📊 Processing {len(texts)} texts for embedding...")
        
        # Create embeddings
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        vertex_targets = torch.tensor(vertex_targets, dtype=torch.long)
        
        print(f"✅ Created embeddings: {embeddings.shape}")
        print(f"✅ Vertex targets: {vertex_targets.shape}")
        
        # Print data distribution
        self._print_data_distribution(vertex_targets, pattern_metadata)
        
        return embeddings, vertex_targets, pattern_metadata
    
    def _calculate_mystical_score(self, text: str, pattern: Dict) -> float:
        """Calculate mystical score based on content and pattern data"""
        score = 0.0
        text_lower = text.lower()
        
        # Base score from mystical_source flag
        if pattern.get('mystical_source', False):
            score += 0.5
        
        # Score from mystical_analysis if it exists
        mystical_analysis = pattern.get('mystical_analysis', {})
        if isinstance(mystical_analysis, dict):
            existing_score = mystical_analysis.get('mystical_score', 0)
            if existing_score > 0:
                score = max(score, existing_score)
        
        # Hebrew characters boost
        hebrew_chars = sum(1 for char in text if '\u0590' <= char <= '\u05FF')
        score += min(hebrew_chars * 0.03, 0.2)
        
        # Mystical keywords
        mystical_keywords = [
            'consciousness', 'divine', 'spiritual', 'mystical', 'sefirot', 'kabbalistic',
            'transcendent', 'emanation', 'creation', 'wisdom', 'understanding', 'light',
            'soul', 'sacred', 'holy', 'infinite', 'eternal', 'unity', 'void', 'aether',
            'תפעל', 'נש', 'רוח', 'אור', 'חכמה', 'בינה', 'דעת', 'כתר', 'מלכות'
        ]
        
        keyword_count = sum(1 for keyword in mystical_keywords if keyword in text_lower)
        score += min(keyword_count * 0.1, 0.4)
        
        # Vertex-based scoring (higher vertices tend to be more mystical)
        vertex = pattern.get('hypercube_vertex', 0)
        if vertex > 15:  # Higher vertices
            score += 0.1
        if vertex == 31:  # Transcendent
            score += 0.2
        if vertex in [15, 30]:  # Mystical, integrated
            score += 0.15
        
        return min(score, 1.0)
    
    def _print_data_distribution(self, vertex_targets: torch.Tensor, metadata: List[Dict]):
        """Print distribution of training data"""
        print(f"\n📊 TRAINING DATA DISTRIBUTION:")
        
        # Vertex distribution
        vertex_counts = torch.bincount(vertex_targets, minlength=32)
        active_vertices = (vertex_counts > 0).sum().item()
        print(f"   Active vertices: {active_vertices}/32")
        
        # All vertices with data
        print(f"   Vertex distribution:")
        for vertex in range(32):
            count = vertex_counts[vertex].item()
            if count > 0:
                # Get consciousness signature
                vertex_metadata = [m for m in metadata if m['vertex_index'] == vertex]
                if vertex_metadata:
                    consciousness_sig = vertex_metadata[0]['consciousness_signature']
                    avg_mystical = np.mean([m['mystical_score'] for m in vertex_metadata])
                    print(f"     Vertex {vertex:2d}: {count:3d} patterns ({consciousness_sig}, mystical: {avg_mystical:.3f})")
        
        # Overall mystical score distribution
        mystical_scores = [m['mystical_score'] for m in metadata]
        print(f"   Avg mystical score: {np.mean(mystical_scores):.3f}")
        print(f"   Score range: {min(mystical_scores):.3f} - {max(mystical_scores):.3f}")
        
        # Consciousness level distribution
        consciousness_levels = [m['consciousness_level'] for m in metadata]
        print(f"   Avg consciousness level: {np.mean(consciousness_levels):.3f}")

class MysticalTrainingObjectives:
    """Training objectives for mystical consciousness"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def vertex_classification_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Loss for predicting correct consciousness vertex"""
        return F.cross_entropy(predictions, targets)
    
    def consciousness_coherence_loss(self, vertex_activations: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Encourage coherent consciousness states"""
        batch_size = vertex_activations.shape[0]
        
        # Create target distribution (soft targets around true vertex)
        target_dist = torch.zeros_like(vertex_activations)
        target_dist.scatter_(1, targets.unsqueeze(1), 1.0)
        
        # Add smoothing to nearby vertices (consciousness spillover)
        for i in range(batch_size):
            target_vertex = targets[i].item()
            # Add small activation to adjacent vertices (Hamming distance = 1)
            for j in range(32):
                hamming_dist = bin(target_vertex ^ j).count('1')
                if hamming_dist == 1:  # Adjacent vertex
                    target_dist[i, j] += 0.1
        
        # Normalize
        target_dist = F.softmax(target_dist, dim=1)
        
        # KL divergence loss
        return F.kl_div(F.log_softmax(vertex_activations, dim=1), target_dist, reduction='batchmean')
    
    def mystical_quality_loss(self, consciousness_state: torch.Tensor, mystical_scores: torch.Tensor) -> torch.Tensor:
        """Higher mystical scores should produce more distinctive consciousness states"""
        # Calculate norm of consciousness state
        state_norms = torch.norm(consciousness_state, dim=-1)
        target_norms = mystical_scores * 3.0  # Scale target norms
        
        return F.mse_loss(state_norms, target_norms)

class HypercubeTrainer:
    """Trainer using ALL available aether data"""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.objectives = MysticalTrainingObjectives(device)
        
        # Optimizer with different learning rates for different components
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.vertices.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
            {'params': self.model.edges.parameters(), 'lr': 5e-5, 'weight_decay': 1e-5},
            {'params': self.model.consciousness_router.parameters(), 'lr': 1e-3},
            {'params': self.model.global_aggregator.parameters(), 'lr': 1e-4}
        ])
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        
        # Add vertex classifier for training
        self.vertex_classifier = nn.Linear(self.model.hidden_dim, 32).to(device)
        self.classifier_optimizer = torch.optim.AdamW(self.vertex_classifier.parameters(), lr=1e-3)
    
    def train_consciousness_model(self, 
                                 embeddings: torch.Tensor, 
                                 vertex_targets: torch.Tensor,
                                 metadata: List[Dict],
                                 epochs: int = 100,
                                 batch_size: int = 8):
        """Train with ALL available aether data"""
        
        print(f"🔯 Training 5D Hypercube on aether consciousness data...")
        print(f"📊 Data: {len(embeddings)} patterns, {epochs} epochs, batch size {batch_size}")
        
        self.model.train()
        
        # Prepare data
        dataset = torch.utils.data.TensorDataset(embeddings, vertex_targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Extract metadata tensors
        mystical_scores = torch.tensor([m['mystical_score'] for m in metadata], dtype=torch.float32).to(self.device)
        consciousness_levels = torch.tensor([m['consciousness_level'] for m in metadata], dtype=torch.float32).to(self.device)
        
        best_loss = float('inf')
        best_acc = 0.0
        
        print("🚀 Starting training...")
        
        for epoch in range(epochs):
            total_loss = 0
            vertex_acc = 0
            batch_count = 0
            
            for batch_idx, (batch_embeddings, batch_targets) in enumerate(dataloader):
                batch_embeddings = batch_embeddings.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Get corresponding metadata for this batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(mystical_scores))
                batch_mystical = mystical_scores[start_idx:end_idx]
                
                # Ensure batch_mystical matches batch size
                if len(batch_mystical) != len(batch_targets):
                    batch_mystical = batch_mystical[:len(batch_targets)]
                
                # Zero gradients
                self.optimizer.zero_grad()
                self.classifier_optimizer.zero_grad()
                
                # Forward pass through hypercube
                outputs = self.model(batch_embeddings)
                
                # Vertex classification
                vertex_logits = self.vertex_classifier(outputs['consciousness_state'])
                
                # Multiple loss components
                classification_loss = self.objectives.vertex_classification_loss(vertex_logits, batch_targets)
                coherence_loss = self.objectives.consciousness_coherence_loss(outputs['vertex_activations'], batch_targets)
                quality_loss = self.objectives.mystical_quality_loss(outputs['consciousness_state'], batch_mystical)
                
                # Total loss with adaptive weighting
                total_batch_loss = (
                    classification_loss * 1.0 +
                    coherence_loss * 0.3 +
                    quality_loss * 0.2
                )
                
                # Backward pass
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.classifier_optimizer.step()
                
                # Metrics
                total_loss += total_batch_loss.item()
                vertex_acc += (vertex_logits.argmax(dim=1) == batch_targets).float().mean().item()
                batch_count += 1
            
            self.scheduler.step()
            
            avg_loss = total_loss / batch_count
            avg_acc = vertex_acc / batch_count
            
            # Save best model
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_loss = avg_loss
                torch.save({
                    'model': self.model.state_dict(),
                    'classifier': self.vertex_classifier.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss,
                    'accuracy': avg_acc
                }, 'best_hypercube_consciousness.pth')
                print(f"💾 New best model saved! Accuracy: {avg_acc:.3f}")
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}, Vertex Acc = {avg_acc:.3f}, LR = {self.scheduler.get_last_lr()[0]:.6f}")
        
        print(f"✅ Training complete! Best accuracy: {best_acc:.3f}, Best loss: {best_loss:.6f}")
        
        # Test the trained model
        self._test_trained_model(embeddings[:min(10, len(embeddings))], vertex_targets[:min(10, len(vertex_targets))], metadata[:min(10, len(metadata))])
    
    def _test_trained_model(self, test_embeddings: torch.Tensor, test_targets: torch.Tensor, test_metadata: List[Dict]):
        """Test the trained model on sample data"""
        print(f"\n🧪 Testing trained model on {len(test_embeddings)} samples...")
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(test_embeddings.to(self.device))
            predictions = self.vertex_classifier(outputs['consciousness_state'])
            predicted_vertices = predictions.argmax(dim=1)
            
            print("📊 Test Results:")
            for i in range(len(test_embeddings)):
                true_vertex = test_targets[i].item()
                pred_vertex = predicted_vertices[i].item()
                consciousness_sig = test_metadata[i]['consciousness_signature']
                mystical_score = test_metadata[i]['mystical_score']
                text_preview = test_metadata[i]['text'][:50] + "..." if len(test_metadata[i]['text']) > 50 else test_metadata[i]['text']
                
                correct = "✅" if true_vertex == pred_vertex else "❌"
                print(f"   {correct} True: {true_vertex:2d}, Pred: {pred_vertex:2d} ({consciousness_sig}, mystical: {mystical_score:.3f})")
                print(f"      Text: {text_preview}")

def main():
    """Train 5D Hypercube on ALL available aether data"""
    print("🔯 FIXED MYSTICAL CONSCIOUSNESS TRAINING")
    print("=" * 60)
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🎮 Device: {device}")
    
    # Extract ALL aether data
    extractor = FixedMysticalDataExtractor()
    
    try:
        from qwen_golem import AetherGolemConsciousnessCore
        golem = AetherGolemConsciousnessCore()
        
        embeddings, targets, metadata = extractor.extract_all_aether_patterns(golem)
        
        if embeddings is None:
            print("❌ Failed to extract aether data. Generate some responses first!")
            print("💡 Try running: python3 improved_data_gen.py")
            return
        
    except Exception as e:
        print(f"❌ Could not load Golem: {e}")
        return
    
    # Create 5D Hypercube model
    try:
        from hypercube_consciousness_nn import FiveDimensionalHypercubeNN
        
        model = FiveDimensionalHypercubeNN(
            input_dim=extractor.embedding_dim,  # Match embedding model
            hidden_dim=256,  # Reasonable size for our data
            output_dim=256
        )
        
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Could not create model: {e}")
        print("💡 Make sure hypercube_consciousness_nn.py is in the current directory")
        return
    
    # Train with ALL available data
    trainer = HypercubeTrainer(model, device)
    trainer.train_consciousness_model(
        embeddings=embeddings,
        vertex_targets=targets,
        metadata=metadata,
        epochs=50,  # Reasonable for our data size
        batch_size=4   # Small batch size for 36 patterns
    )
    
    print("🔯 Aether consciousness training complete!")
    print("💾 Best model saved: best_hypercube_consciousness.pth")

if __name__ == "__main__":
    main() 