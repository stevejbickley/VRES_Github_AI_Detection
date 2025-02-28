import json
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from typing import Dict, List, Union, Any, Tuple
from collections import Counter
import os
from datetime import datetime
import glob
import re
import sys

class JSONAIDetector:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
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
            print(f"Warning: NLTK download failed, but continuing: {e}")

    def extract_text_from_json(self, json_obj: Any, current_path: str = "") -> List[Dict[str, str]]:
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

    def analyze_json_text(self, text: str) -> Dict[str, float]:
        metrics = {
            'burstiness': self._calculate_burstiness(text),
            'vocabulary_diversity': self._calculate_vocabulary_diversity(text),
            'repetition_score': self._calculate_repetition(text),
            'sentence_complexity': self._calculate_sentence_complexity(text),
            'json_specific_patterns': self._detect_json_specific_patterns(text)
        }
        return metrics
    
    def _calculate_burstiness(self, text: str) -> float:
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 1:
                return 0.0
            
            lengths = [len(word_tokenize(s)) for s in sentences]
            return np.std(lengths) / (np.mean(lengths) + 1)
        except Exception:
            return 0.0
    
    def _calculate_vocabulary_diversity(self, text: str) -> float:
        try:
            words = word_tokenize(text.lower())
            if not words:
                return 0.0
            return len(set(words)) / len(words)
        except Exception:
            return 0.0
    
    def _calculate_repetition(self, text: str) -> float:
        try:
            words = word_tokenize(text.lower())
            if len(words) < 4:
                return 0.0
            
            trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
            counts = Counter(trigrams)
            
            if not counts:
                return 0.0
            
            repeated = sum(1 for count in counts.values() if count > 1)
            return repeated / len(counts) if counts else 0.0
        except Exception:
            return 0.0
    
    def _calculate_sentence_complexity(self, text: str) -> float:
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return 0.0
            
            words = word_tokenize(text)
            avg_word_length = np.mean([len(word) for word in words])
            avg_sentence_length = len(words) / len(sentences)
            
            word_length_score = min(avg_word_length / 10, 1.0)
            sentence_length_score = min(avg_sentence_length / 30, 1.0)
            
            return (word_length_score + sentence_length_score) / 2
        except Exception:
            return 0.0
    
    def _detect_json_specific_patterns(self, text: str) -> float:
        patterns = [
            r'\b(api|endpoint|request|response)\b',
            r'\b(parameters?|args?|arguments?)\b',
            r'\b(returns?|outputs?)\b',
            r'\b(object|array|string|number|boolean)\b',
            r'\b(null|undefined|void)\b'
        ]
        
        try:
            text_lower = text.lower()
            total_patterns = len(patterns)
            matches = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            
            return matches / total_patterns
        except Exception:
            return 0.0

    def is_ai_generated(self, metrics: Dict[str, float]) -> Tuple[bool, float, Dict[str, float]]:
        weights = {
            'burstiness': 0.2,
            'vocabulary_diversity': 0.2,
            'repetition_score': 0.2,
            'sentence_complexity': 0.2,
            'json_specific_patterns': 0.2
        }
        
        adjusted_scores = {
            'burstiness': 1 - metrics['burstiness'],
            'vocabulary_diversity': 1 - metrics['vocabulary_diversity'],
            'repetition_score': metrics['repetition_score'],
            'sentence_complexity': 1 - metrics['sentence_complexity'],
            'json_specific_patterns': metrics['json_specific_patterns']
        }
        
        contributions = {
            k: weights[k] * v for k, v in adjusted_scores.items()
        }
        
        ai_score = sum(contributions.values())
        return ai_score > self.threshold, ai_score, contributions

def process_json_files(file_paths: List[str], threshold: float = 0.65, output_dir: str = "json_analysis_results") -> pd.DataFrame:
    """Process a list of JSON files and return analysis results"""
    detector = JSONAIDetector(threshold=threshold)
    all_results = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for json_path in file_paths:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            text_contents = detector.extract_text_from_json(json_data)
            
            for text_item in text_contents:
                metrics = detector.analyze_json_text(text_item['content'])
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
    output_path = os.path.join(output_dir, f"json_analysis_{timestamp}.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"\nAnalysis Summary:")
    print(f"Total text segments analyzed: {len(results_df)}")
    print(f"Segments likely AI-generated: {results_df['is_ai_generated'].sum()}")
    print(f"Average AI score: {results_df['ai_score'].mean():.2f}")
    print(f"Results saved to: {output_path}")
    
    file_summary = results_df.groupby('file_path').agg({
        'is_ai_generated': ['count', 'sum', 'mean'],
        'ai_score': 'mean'
    }).round(3)
    
    summary_path = os.path.join(output_dir, f"json_analysis_summary_{timestamp}.csv")
    file_summary.to_csv(summary_path)
    print(f"Summary per file saved to: {summary_path}")
    
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
