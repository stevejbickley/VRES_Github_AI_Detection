import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import math
import re
from collections import Counter
from typing import Dict, Tuple, List

class SimplifiedAIDetector:
    def __init__(self, threshold=0.65):
        """Initialize a simpler AI content detector with customizable threshold"""
        self.threshold = threshold
        
        # Try to download NLTK data without SSL verification
        try:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            nltk.download('punkt')
        except Exception as e:
            print(f"Warning: NLTK download failed, but continuing anyway: {e}")
    
    def calculate_burstiness(self, text: str) -> float:
        """Calculate sentence length variation as a measure of 'burstiness'"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 1:
                return 0.0
            
            lengths = [len(word_tokenize(s)) for s in sentences]
            return np.std(lengths) / (np.mean(lengths) + 1)
        except Exception:
            return 0.0
    
    def calculate_vocabulary_diversity(self, text: str) -> float:
        """Calculate lexical diversity using Type-Token Ratio"""
        try:
            words = word_tokenize(text.lower())
            if not words:
                return 0.0
            
            unique_words = len(set(words))
            total_words = len(words)
            
            return unique_words / total_words
        except Exception:
            return 0.0
    
    def calculate_repetition_score(self, text: str) -> float:
        """Calculate repetition of phrases (common in AI-generated text)"""
        try:
            words = word_tokenize(text.lower())
            if len(words) < 4:
                return 0.0
                
            # Create n-grams
            trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
            
            # Count occurrences
            counts = Counter(trigrams)
            
            # Calculate repetition score
            if not counts:
                return 0.0
                
            # How many trigrams are repeated
            repeated = sum(1 for count in counts.values() if count > 1)
            total = len(counts)
            
            return repeated / total if total > 0 else 0.0
        except Exception:
            return 0.0
    
    def calculate_sentence_starter_variety(self, text: str) -> float:
        """Measure variety in sentence starters (AI often uses patterns)"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 2:
                return 0.5  # Neutral for short texts
                
            starters = []
            for sentence in sentences:
                words = word_tokenize(sentence)
                if words:
                    starters.append(words[0].lower())
            
            unique_starters = len(set(starters))
            return unique_starters / len(sentences)
        except Exception:
            return 0.5
    
    def calculate_transition_usage(self, text: str) -> float:
        """Calculate usage of transition words/phrases (AI texts often overuse these)"""
        transitions = [
            'however', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'in conclusion', 'in summary', 'in short',
            'to summarize', 'overall', 'finally', 'furthermore', 'moreover'
        ]
        
        try:
            lower_text = text.lower()
            word_count = len(word_tokenize(text))
            if word_count < 5:
                return 0.5  # Neutral for very short texts
                
            transition_count = sum(lower_text.count(t) for t in transitions)
            
            # Normalize by text length
            normalized_score = transition_count / (word_count / 100)  # Per 100 words
            
            # Convert to 0-1 range (higher score = more likely AI)
            return min(1.0, normalized_score / 5)  # Cap at 1.0
        except Exception:
            return 0.5

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze text using multiple heuristic features"""
        if not text or not isinstance(text, str):
            text = str(text)
            
        metrics = {
            'burstiness': self.calculate_burstiness(text),
            'vocabulary_diversity': self.calculate_vocabulary_diversity(text),
            'repetition_score': self.calculate_repetition_score(text),
            'sentence_starter_variety': self.calculate_sentence_starter_variety(text),
            'transition_usage': self.calculate_transition_usage(text)
        }
        
        return metrics
    
    def is_ai_generated(self, metrics: Dict[str, float]) -> Tuple[bool, float, Dict[str, float]]:
        """Determine if text is likely AI-generated based on heuristic metrics"""
        # Weights for each metric
        weights = {
            'burstiness': 0.25,  # Lower burstiness = more AI-like
            'vocabulary_diversity': 0.2,  # Lower diversity = more AI-like
            'repetition_score': 0.2,  # Higher repetition = more AI-like
            'sentence_starter_variety': 0.2,  # Lower variety = more AI-like
            'transition_usage': 0.15  # Higher transition usage = more AI-like
        }
        
        # Adjust scores so higher always means more AI-like
        adjusted_scores = {
            'burstiness': 1 - metrics['burstiness'],
            'vocabulary_diversity': 1 - metrics['vocabulary_diversity'],
            'repetition_score': metrics['repetition_score'],
            'sentence_starter_variety': 1 - metrics['sentence_starter_variety'],
            'transition_usage': metrics['transition_usage']
        }
        
        # Calculate weighted contributions
        contributions = {
            k: weights[k] * v for k, v in adjusted_scores.items()
        }
        
        # Calculate final score
        ai_score = sum(contributions.values())
        
        return ai_score > self.threshold, ai_score, contributions

def analyze_commit_messages(
    csv_path: str, 
    message_column: str = 'message',
    threshold: float = 0.65,
    output_file: str = 'analyzed_commits.csv'
) -> pd.DataFrame:
    """
    Analyze commit messages in a CSV file for AI-generated content
    
    Args:
        csv_path: Path to the CSV file
        message_column: Column name containing text to analyze
        threshold: Detection threshold (0-1, higher = more sensitive)
        output_file: Path to save results
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df)} records from {csv_path}")
        
        # Initialize detector
        detector = SimplifiedAIDetector(threshold=threshold)
        
        # Store results
        results = []
        
        # Process each message
        for i, message in enumerate(df[message_column]):
            if i % 10 == 0:
                print(f"Processing message {i+1}/{len(df)}")
                
            metrics = detector.analyze_text(str(message))
            is_ai, score, contributions = detector.is_ai_generated(metrics)
            
            results.append({
                'message': message,
                'is_ai_generated': is_ai,
                'ai_score': score,
                **metrics,
                **{f"{k}_contribution": v for k, v in contributions.items()}
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original DataFrame
        final_df = pd.concat([df, results_df.drop('message', axis=1)], axis=1)
        
        # Save results
        if output_file:
            final_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"Total messages analyzed: {len(final_df)}")
        print(f"Messages likely AI-generated: {final_df['is_ai_generated'].sum()}")
        print(f"Average AI score: {final_df['ai_score'].mean():.2f}")
        
        return final_df
        
    except Exception as e:
        print(f"Error analyzing commit messages: {e}")
        raise

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = 'test.csv'
    
    threshold = 0.65
    if len(sys.argv) > 2:
        try:
            threshold = float(sys.argv[2])
        except ValueError:
            print(f"Invalid threshold value: {sys.argv[2]}. Using default: 0.65")
    
    print(f"Using threshold: {threshold}")
    results = analyze_commit_messages(csv_path, threshold=threshold)
