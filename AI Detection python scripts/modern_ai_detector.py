import json
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist
import numpy as np
from typing import Dict, List, Union, Any, Tuple
from collections import Counter
import os
from datetime import datetime
import glob
import re
import sys
import string
from scipy.stats import entropy

class ModernAIDetector:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self.setup_nltk()
        self.setup_patterns()
        
    def setup_nltk(self):
        """Initialize NLTK with SSL workaround"""
        try:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
        except Exception as e:
            print(f"Warning: NLTK download failed, but continuing: {e}")

    def setup_patterns(self):
        """Setup pattern detection for modern AI models"""
        self.gpt4_patterns = [
            r'\b(therefore|thus|hence|consequently)\b',
            r'\b(firstly|secondly|finally|lastly)\b',
            r'\b(it\'s worth noting|it\'s important to note|notably)\b',
            r'\b(in essence|essentially|fundamentally)\b',
            r'\b(let\'s explore|let\'s examine|let\'s analyze)\b',
            r'\b(to put it simply|in other words|specifically)\b',
            r'\b(for example|for instance|such as)\b',
            r'\b(however|nevertheless|nonetheless)\b'
        ]
        
        self.modern_ai_phrases = [
            "based on the provided",
            "it depends on the context",
            "there are several factors",
            "it's important to consider",
            "let me help you with that",
            "to answer your question",
            "in my analysis",
            "as an AI language model"
        ]

    def extract_text_from_json(self, json_obj: Any, current_path: str = "") -> List[Dict[str, str]]:
        """Extract text content from JSON recursively"""
        text_contents = []
        
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                
                if isinstance(value, str) and len(value.split()) > 3:
                    text_contents.append({
                        "path": new_path,
                        "content": value
                    })
                elif isinstance(value, (dict, list)):
                    text_contents.extend(self.extract_text_from_json(value, new_path))
                    
        elif isinstance(json_obj, list):
            for i, item in enumerate(json_obj):
                new_path = f"{current_path}[{i}]"
                text_contents.extend(self.extract_text_from_json(item, new_path))
                
        return text_contents

    def calculate_entropy_score(self, text: str) -> float:
        """Calculate entropy of word distributions"""
        try:
            words = word_tokenize(text.lower())
            if not words:
                return 0.0
            
            freq_dist = FreqDist(words)
            probs = [freq_dist.freq(word) for word in freq_dist]
            return entropy(probs)
        except Exception:
            return 0.0

    def calculate_consistency_score(self, text: str) -> float:
        """Measure writing style consistency"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 1:
                return 0.5
            
            # Analyze sentence structures
            sentence_lengths = [len(word_tokenize(s)) for s in sentences]
            length_variance = np.var(sentence_lengths)
            
            # Normalize variance to 0-1 range
            normalized_variance = min(length_variance / 100, 1.0)
            
            # Lower variance (more consistency) might indicate AI
            return 1 - normalized_variance
        except Exception:
            return 0.5

    def detect_modern_ai_patterns(self, text: str) -> float:
        """Detect patterns common in modern AI-generated text"""
        try:
            text_lower = text.lower()
            
            # Check for GPT-4 like patterns
            pattern_matches = sum(1 for pattern in self.gpt4_patterns 
                                if re.search(pattern, text_lower))
            
            # Check for common AI phrases
            phrase_matches = sum(1 for phrase in self.modern_ai_phrases 
                               if phrase in text_lower)
            
            # Combine scores
            pattern_score = pattern_matches / len(self.gpt4_patterns)
            phrase_score = phrase_matches / len(self.modern_ai_phrases)
            
            return (pattern_score + phrase_score) / 2
        except Exception:
            return 0.0

    def calculate_coherence_score(self, text: str) -> float:
        """Measure text coherence (too perfect might indicate AI)"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 1:
                return 0.5
            
            # Analyze transition words
            transition_patterns = [
                r'\b(however|therefore|moreover|furthermore)\b',
                r'\b(in addition|as a result|consequently)\b',
                r'\b(for example|specifically|particularly)\b'
            ]
            
            transition_counts = []
            for sentence in sentences:
                count = sum(1 for pattern in transition_patterns 
                          if re.search(pattern, sentence.lower()))
                transition_counts.append(count)
            
            # Too many transitions might indicate AI
            avg_transitions = np.mean(transition_counts)
            return min(avg_transitions / 2, 1.0)  # Normalize to 0-1
        except Exception:
            return 0.5

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Comprehensive text analysis with modern AI detection"""
        metrics = {
            'entropy_score': self.calculate_entropy_score(text),
            'consistency_score': self.calculate_consistency_score(text),
            'ai_pattern_score': self.detect_modern_ai_patterns(text),
            'coherence_score': self.calculate_coherence_score(text)
        }
        return metrics

    def is_ai_generated(self, metrics: Dict[str, float]) -> Tuple[bool, float, Dict[str, float]]:
        """Determine if text is likely AI-generated using modern detection methods"""
        weights = {
            'entropy_score': 0.25,      # Lower entropy might indicate AI
            'consistency_score': 0.25,   # Higher consistency might indicate AI
            'ai_pattern_score': 0.3,     # More patterns suggest AI
            'coherence_score': 0.2       # Too perfect coherence suggests AI
        }
        
        # Adjust scores so higher always means more AI-like
        adjusted_scores = {
            'entropy_score': 1 - (metrics['entropy_score'] / 5),  # Normalize entropy
            'consistency_score': metrics['consistency_score'],
            'ai_pattern_score': metrics['ai_pattern_score'],
            'coherence_score': metrics['coherence_score']
        }
        
        # Calculate weighted contributions
        contributions = {
            k: weights[k] * v for k, v in adjusted_scores.items()
        }
        
        ai_score = sum(contributions.values())
        return ai_score > self.threshold, ai_score, contributions

def process_json_files(file_paths: List[str], threshold: float = 0.65, output_dir: str = "json_analysis_results") -> pd.DataFrame:
    """Process JSON files with modern AI detection"""
    detector = ModernAIDetector(threshold=threshold)
    all_results = []
    total_files = len(file_paths)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, json_path in enumerate(file_paths, 1):
        print(f"\nProcessing file {idx}/{total_files}: {os.path.basename(json_path)}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            text_contents = detector.extract_text_from_json(json_data)
            print(f"Found {len(text_contents)} text segments to analyze")
            
            for i, text_item in enumerate(text_contents):
                if i % 100 == 0 and i > 0:
                    print(f"Analyzed {i} segments...")
                    
                metrics = detector.analyze_text(text_item['content'])
                is_ai, score, contributions = detector.is_ai_generated(metrics)
                
                result = {
                    'file_path': json_path,
                    'json_path': text_item['path'],
                    'content': text_item['content'],
                    'is_ai_generated': is_ai,
                    'ai_score': score,
                    **metrics,
                    **{f"{k}_contribution": v for k, v in contributions.items()}
                }
                all_results.append(result)
                
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            continue
    
    if not all_results:
        print("No results to analyze!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    output_path = os.path.join(output_dir, f"json_analysis_{timestamp}.csv")
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\nAnalysis Summary:")
    print(f"Total text segments analyzed: {len(results_df)}")
    print(f"Segments likely AI-generated: {results_df['is_ai_generated'].sum()}")
    print(f"Average AI score: {results_df['ai_score'].mean():.2f}")
    print(f"Results saved to: {output_path}")
    
    # Print detailed metrics
    print("\nDetailed Metrics:")
    print(f"Average entropy score: {results_df['entropy_score'].mean():.2f}")
    print(f"Average consistency score: {results_df['consistency_score'].mean():.2f}")
    print(f"Average AI pattern score: {results_df['ai_pattern_score'].mean():.2f}")
    print(f"Average coherence score: {results_df['coherence_score'].mean():.2f}")
    
    # Create file summary
    file_summary = results_df.groupby('file_path').agg({
        'is_ai_generated': ['count', 'sum', 'mean'],
        'ai_score': ['mean', 'max'],
        'entropy_score': 'mean',
        'consistency_score': 'mean',
        'ai_pattern_score': 'mean',
        'coherence_score': 'mean'
    }).round(3)
    
    # Save summary
    summary_path = os.path.join(output_dir, f"json_analysis_summary_{timestamp}.csv")
    file_summary.to_csv(summary_path)
    print(f"\nDetailed summary per file saved to: {summary_path}")
    
    return results_df

def main():
    """Main execution function"""
    try:
        if len(sys.argv) > 1:
            target_path = sys.argv[1]
            
            if os.path.isdir(target_path):
                print(f"Analyzing JSON files in directory: {target_path}")
                json_files = glob.glob(os.path.join(target_path, "*.json"))
                print(f"Found {len(json_files)} JSON files in {target_path}")
            else:
                json_files = [target_path]
                print(f"Analyzing single JSON file: {target_path}")
            
            if not json_files:
                print("No JSON files found!")
                return
            
            results = process_json_files(json_files)
            
        else:
            print("Please provide a JSON file or directory path")
            return
            
    except Exception as e:
        print(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()
