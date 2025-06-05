```python
#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from datetime import datetime
import fitz
import traceback
import re
import unicodedata
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ConfidenceInterval:
    """Class to store confidence interval information."""
    lower: float
    upper: float
    confidence_level: float
    raw_text: str
    context: str
    section: str

@dataclass
class SampleSize:
    """Class to store sample size information."""
    value: int
    group: Optional[str]  # For when groups are specified (e.g., "n₁ = 50")
    raw_text: str
    context: str
    section: str

@dataclass
class EffectSize:
    """Class to store effect size information."""
    value: float
    type: str  # e.g., "Cohen's d", "Hedges' g", "r", etc.
    raw_text: str
    context: str
    section: str

class StatisticalExtractor:
    """Class to handle extraction of various statistical measures."""
    
    def __init__(self):
        # Confidence Interval patterns
        self.ci_patterns = [
            # 95% CI [1.2, 2.3]
            r'(\d+)%\s*CI\s*\[([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\]',
            # (95% confidence interval: 1.2-2.3)
            r'(\d+)%\s*confidence\s*interval\s*[:=]\s*([-+]?\d*\.?\d+)\s*[-–]\s*([-+]?\d*\.?\d+)',
            # CI95% = [1.2, 2.3]
            r'CI\s*(\d+)%\s*[:=]\s*\[([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\]'
        ]
        
        # Sample Size patterns
        self.n_patterns = [
            # N = 100
            r'[Nn]\s*[:=]\s*(\d+)',
            # (n₁ = 50)
            r'[Nn](?:₁|₂|₃|_1|_2|_3)\s*[:=]\s*(\d+)',
            # total sample size of 150
            r'(?:total\s+)?sample\s+size\s+(?:of|[:=])\s*(\d+)',
            # 100 participants
            r'(\d+)\s+(?:participants|subjects|patients|individuals)'
        ]
        
        # Effect Size patterns
        self.effect_patterns = [
            # Cohen's d = 0.5
            r"Cohen['']s\s*d\s*[:=]\s*([-+]?\d*\.?\d+)",
            # Hedges' g = 0.6
            r"Hedges['']?\s*g\s*[:=]\s*([-+]?\d*\.?\d+)",
            # r = 0.3
            r'(?<![\w])r\s*[:=]\s*([-+]?\d*\.?\d+)',
            # eta² = 0.2
            r'(?:partial\s+)?η²\s*[:=]\s*([-+]?\d*\.?\d+)',
            # R² = 0.25
            r'R²\s*[:=]\s*([-+]?\d*\.?\d+)'
        ]
        
        # Compile all patterns
        self.ci_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.ci_patterns]
        self.n_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.n_patterns]
        self.effect_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.effect_patterns]
    
    def extract_statistics(self, text: str, section: str = "unknown") -> Dict[str, List[Any]]:
        """
        Extract all statistical measures from text.
        
        Args:
            text (str): Text to analyze
            section (str): Section name where the text was found
            
        Returns:
            Dict containing lists of extracted statistical measures
        """
        return {
            "confidence_intervals": self.extract_confidence_intervals(text, section),
            "sample_sizes": self.extract_sample_sizes(text, section),
            "effect_sizes": self.extract_effect_sizes(text, section),
            "p_values": extract_p_values(text, section)  # From previous implementation
        }
    
    def get_context(self, text: str, match: re.Match, window: int = 10) -> str:
        """Extract surrounding context for a match."""
        words = text.split()
        try:
            match_start = text.index(match.group(0))
            word_pos = len(text[:match_start].split())
            context_start = max(0, word_pos - window)
            context_end = min(len(words), word_pos + window)
            return ' '.join(words[context_start:context_end])
        except ValueError:
            return text.strip()
    
    def extract_confidence_intervals(self, text: str, section: str) -> List[ConfidenceInterval]:
        """Extract confidence intervals from text."""
        intervals = []
        
        for pattern in self.ci_patterns:
            for match in pattern.finditer(text):
                try:
                    confidence_level = float(match.group(1))
                    lower = float(match.group(2))
                    upper = float(match.group(3))
                    
                    intervals.append(ConfidenceInterval(
                        lower=lower,
                        upper=upper,
                        confidence_level=confidence_level,
                        raw_text=match.group(0),
                        context=self.get_context(text, match),
                        section=section
                    ))
                except (ValueError, IndexError):
                    continue
        
        return intervals
    
    def extract_sample_sizes(self, text: str, section: str) -> List[SampleSize]:
        """Extract sample sizes from text."""
        samples = []
        
        for pattern in self.n_patterns:
            for match in pattern.finditer(text):
                try:
                    value = int(match.group(1))
                    group = None
                    
                    # Check for group indicators
                    raw_text = match.group(0)
                    if any(indicator in raw_text.lower() for indicator in ['₁', '_1', '₂', '_2', '₃', '_3']):
                        group = re.search(r'[Nn](₁|₂|₃|_1|_2|_3)', raw_text).group(1)
                    
                    samples.append(SampleSize(
                        value=value,
                        group=group,
                        raw_text=raw_text,
                        context=self.get_context(text, match),
                        section=section
                    ))
                except (ValueError, IndexError):
                    continue
        
        return samples
    
    def extract_effect_sizes(self, text: str, section: str) -> List[EffectSize]:
        """Extract effect sizes from text."""
        effects = []
        
        effect_type_mapping = {
            "cohen": "Cohen's d",
            "hedges": "Hedges' g",
            "r": "Correlation coefficient",
            "η²": "Eta squared",
            "R²": "R squared"
        }
        
        for pattern in self.effect_patterns:
            for match in pattern.finditer(text):
                try:
                    value = float(match.group(1))
                    raw_text = match.group(0).lower()
                    
                    # Determine effect size type
                    effect_type = next(
                        (v for k, v in effect_type_mapping.items() if k.lower() in raw_text),
                        "Unknown"
                    )
                    
                    effects.append(EffectSize(
                        value=value,
                        type=effect_type,
                        raw_text=match.group(0),
                        context=self.get_context(text, match),
                        section=section
                    ))
                except (ValueError, IndexError):
                    continue
        
        return effects

def analyze_statistics(stats: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Generate summary analysis of extracted statistics.
    
    Args:
        stats: Dictionary containing lists of extracted statistical measures
        
    Returns:
        Dictionary containing summary statistics
    """
    analysis = {
        "confidence_intervals": {
            "count": len(stats["confidence_intervals"]),
            "levels": defaultdict(int),
            "by_section": defaultdict(int)
        },
        "sample_sizes": {
            "count": len(stats["sample_sizes"]),
            "total": sum(s.value for s in stats["sample_sizes"]),
            "by_group": defaultdict(list),
            "by_section": defaultdict(int)
        },
        "effect_sizes": {
            "count": len(stats["effect_sizes"]),
            "by_type": defaultdict(list),
            "by_section": defaultdict(int)
        }
    }
    
    # Analyze confidence intervals
    for ci in stats["confidence_intervals"]:
        analysis["confidence_intervals"]["levels"][ci.confidence_level] += 1
        analysis["confidence_intervals"]["by_section"][ci.section] += 1
    
    # Analyze sample sizes
    for n in stats["sample_sizes"]:
        if n.group:
            analysis["sample_sizes"]["by_group"][n.group].append(n.value)
        analysis["sample_sizes"]["by_section"][n.section] += 1
    
    # Analyze effect sizes
    for es in stats["effect_sizes"]:
        analysis["effect_sizes"]["by_type"][es.type].append(es.value)
        analysis["effect_sizes"]["by_section"][es.section] += 1
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description='Extract statistical measures from PDF documents.')
    parser.add_argument('pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--raw', action='store_true', help='Skip text cleaning')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='Output format')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Extract text from PDF
        success, content = extract_text_from_pdf(args.pdf_path, clean=not args.raw)
        
        if not success:
            logging.error(f"Failed to extract text: {content}")
            sys.exit(1)
        
        # Extract sections
        sections = identify_sections(content)
        
        # Initialize statistical extractor
        extractor = StatisticalExtractor()
        
        # Process each section
        all_stats = defaultdict(list)
        for section_name, section_text in sections.items():
            section_stats = extractor.extract_statistics(section_text, section_name)
            for key, values in section_stats.items():
                all_stats[key].extend(values)
        
        # Analyze results
        analysis = analyze_statistics(all_stats)
        
        # Output results
        if args.format == 'json':
            import json
            # Convert dataclass objects to dictionaries
            output = {
                "statistics": {
                    k: [vars(item) for item in v] for k, v in all_stats.items()
                },
                "analysis": analysis
            }
            print(json.dumps(output, indent=2))
        else:
            print("\nStatistical Analysis Results:")
            print("-" * 80)
            
            # Print confidence intervals
            print("\nConfidence Intervals:")
            for ci in all_stats["confidence_intervals"]:
                print(f"\n{ci.confidence_level}% CI [{ci.lower}, {ci.upper}]")
                print(f"Context: ...{ci.context}...")
                print(f"Section: {ci.section}")
            
            # Print sample sizes
            print("\nSample Sizes:")
            for n in all_stats["sample_sizes"]:
                print(f"\n{n.raw_text}")
                if n.group:
                    print(f"Group: {n.group}")
                print(f"Context: ...{n.context}...")
                print(f"Section: {n.section}")
            
            # Print effect sizes
            print("\nEffect Sizes:")
            for es in all_stats["effect_sizes"]:
                print(f"\n{es.type}: {es.value}")
                print(f"Context: ...{es.context}...")
                print(f"Section: {es.section}")
            
            # Print summary
            print("\nSummary:")
            print(f"Total confidence intervals: {analysis['confidence_intervals']['count']}")
            print(f"Total sample sizes: {analysis['sample_sizes']['count']}")
            print(f"Total effect sizes: {analysis['effect_sizes']['count']}")
            
    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

```

Let's also add tests for the new statistical extraction functionality:

```python
import unittest
from extract_stats import StatisticalExtractor

class TestStatisticalExtraction(unittest.TestCase):
    def setUp(self):
        self.extractor = StatisticalExtractor()
    
    def test_confidence_interval_extraction(self):
        """Test extraction of confidence intervals."""
        text = """
        The results showed a significant effect (95% CI [1.2, 2.3]).
        Using a 90% confidence interval: 0.5-1.5
        The CI99% = [-0.3, 0.7] was calculated.
        """
        
        intervals = self.extractor.extract_confidence_intervals(text, "results")
        self.assertEqual(len(intervals), 3)
        
        # Check specific values
        ci = intervals[0]
        self.assertEqual(ci.confidence_level, 95)
        self.assertEqual(ci.lower, 1.2)
        self.assertEqual(ci.upper, 2.3)
    
    def test_sample_size_extraction(self):
        """Test extraction of sample sizes."""
        text = """
        The study included N = 100 participants.
        Group sizes were n₁ = 50 and n₂ = 50.
        We recruited 75 patients for the control group.
        """
        
        sizes = self.extractor.extract_sample_sizes(text, "methods")
        self.assertEqual(len(sizes), 4)
        
        # Check total sample size
        total_n = next(n for n in sizes if n.value == 100)
        self.assertIsNone(total_n.group)
        
        # Check group sizes
        group_1 = next(n for n in sizes if n.group in ['₁', '_1'])
        self.assertEqual(group_1.value, 50)
    
    def test_effect_size_extraction(self):
        """Test extraction of effect sizes."""
        text = """
        The analysis revealed a medium effect (Cohen's d =
```
