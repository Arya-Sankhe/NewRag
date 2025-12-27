"""
CLIP-based Image Relevance Scorer

Computes semantic similarity between user queries and images using OpenCLIP.
This enables objective, reliable image selection for RAG responses.
"""

import base64
import io
from typing import List, Dict, Optional, Tuple
from PIL import Image
import config


class ImageRelevanceScorer:
    """
    Scores images by semantic relevance to a query using CLIP embeddings.
    
    Uses cosine similarity between query text embeddings and image embeddings
    to objectively rank images by relevance.
    """
    
    _instance = None  # Singleton for model reuse
    
    def __new__(cls):
        """Singleton pattern to avoid reloading model on each query."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._preprocess = None
            cls._instance._tokenizer = None
        return cls._instance
    
    def _ensure_model_loaded(self):
        """Lazy load CLIP model only when first needed."""
        if self._model is not None:
            return
            
        import torch
        import open_clip
        
        print(f"📷 Loading CLIP model: {config.CLIP_MODEL_NAME} on {config.CLIP_DEVICE}")
        
        # Set CPU threads for stability
        torch.set_num_threads(2)
        
        # Load model
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME,
            pretrained='openai',
            device=config.CLIP_DEVICE
        )
        self._tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
        
        # Set to eval mode for inference
        self._model.eval()
        
        print(f"   ✓ CLIP model loaded successfully")
    
    def score_images(
        self,
        query: str,
        images_metadata: List[Dict],
        threshold: float = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        Score and rank images by semantic relevance to query.
        
        Args:
            query: User query text
            images_metadata: List of image metadata dicts with base64_data and caption
            threshold: Minimum similarity score (default: config.IMAGE_SIMILARITY_THRESHOLD)
            top_k: Maximum images to return (default: config.MAX_IMAGES_PER_RESPONSE)
            
        Returns:
            List of image dicts with added 'relevance_score' field, sorted by score
        """
        if not images_metadata:
            return []
        
        threshold = threshold or config.IMAGE_SIMILARITY_THRESHOLD
        top_k = top_k or config.MAX_IMAGES_PER_RESPONSE
        
        # Limit images to score for stability
        images_to_score = images_metadata[:config.MAX_IMAGES_TO_SCORE]
        
        try:
            self._ensure_model_loaded()
            
            import torch
            
            # Get query embedding
            query_embedding = self._get_text_embedding(query)
            if query_embedding is None:
                print("   ⚠️ Failed to get query embedding")
                return []
            
            scored_images = []
            
            for img in images_to_score:
                # Get combined score from visual features and caption
                score = self._score_single_image(img, query_embedding)
                
                if score is not None and score >= threshold:
                    img_copy = img.copy()
                    img_copy['relevance_score'] = float(score)
                    scored_images.append(img_copy)
                    print(f"   📸 Image '{img.get('image_id', 'unknown')}' score: {score:.3f}")
            
            # Sort by score descending
            scored_images.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Return top-K
            return scored_images[:top_k]
            
        except Exception as e:
            print(f"   ❌ Error scoring images: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _score_single_image(
        self, 
        img_metadata: Dict, 
        query_embedding
    ) -> Optional[float]:
        """
        Score a single image against the query.
        
        Uses weighted combination of:
        - Visual similarity (image embedding vs query embedding)
        - Text similarity (caption embedding vs query embedding)
        """
        import torch
        
        visual_score = 0.0
        text_score = 0.0
        
        # Get visual embedding from image data
        base64_data = img_metadata.get('base64_data', '')
        if base64_data:
            img_embedding = self._get_image_embedding(
                base64_data, 
                img_metadata.get('mime_type', 'image/png')
            )
            if img_embedding is not None:
                # Cosine similarity
                visual_score = torch.nn.functional.cosine_similarity(
                    query_embedding, img_embedding, dim=-1
                ).item()
        
        # Get text embedding from caption
        caption = img_metadata.get('caption', '') or img_metadata.get('vlm_caption', '') or img_metadata.get('description', '')
        if caption:
            caption_embedding = self._get_text_embedding(caption)
            if caption_embedding is not None:
                text_score = torch.nn.functional.cosine_similarity(
                    query_embedding, caption_embedding, dim=-1
                ).item()
        
        # Weighted combination: prioritize visual features
        if visual_score > 0 and text_score > 0:
            # Both available: 70% visual, 30% text
            return 0.7 * visual_score + 0.3 * text_score
        elif visual_score > 0:
            return visual_score
        elif text_score > 0:
            return text_score
        else:
            return None
    
    def _get_image_embedding(self, base64_data: str, mime_type: str):
        """Convert base64 image to CLIP embedding."""
        import torch
        
        try:
            # Decode base64 to PIL Image
            # Handle both raw base64 and data URL format
            if base64_data.startswith('data:'):
                # Extract base64 part from data URL
                base64_data = base64_data.split(',', 1)[1]
            
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Preprocess and get embedding
            image_tensor = self._preprocess(image).unsqueeze(0).to(config.CLIP_DEVICE)
            
            with torch.no_grad():
                image_features = self._model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features
            
        except Exception as e:
            print(f"   ⚠️ Failed to get image embedding: {e}")
            return None
    
    def _get_text_embedding(self, text: str):
        """Convert text to CLIP embedding."""
        import torch
        
        try:
            # Tokenize and encode
            text_tokens = self._tokenizer([text]).to(config.CLIP_DEVICE)
            
            with torch.no_grad():
                text_features = self._model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features
            
        except Exception as e:
            print(f"   ⚠️ Failed to get text embedding: {e}")
            return None


# Module-level instance for easy import
_scorer_instance = None

def get_scorer() -> ImageRelevanceScorer:
    """Get the singleton scorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = ImageRelevanceScorer()
    return _scorer_instance


def score_images_for_query(query: str, images_metadata: List[Dict]) -> List[Dict]:
    """
    Convenience function to score images for a query.
    
    Args:
        query: User query text
        images_metadata: List of image metadata dicts
        
    Returns:
        List of relevant images with scores, sorted by relevance
    """
    return get_scorer().score_images(query, images_metadata)
