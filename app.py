"""
Ultra-minimal PDF Outline Extractor for <1GB Docker image
Uses custom lightweight semantic similarity with no PyTorch dependencies.
Follows the proper output structure from the original app.py
"""

import pdfplumber
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Constants
INPUT_DOCS = "input/docs"
INPUT_META = "input/challenge1b_input.json"
OUTPUT_FILE = "output/output.json"
TOP_SECTIONS = 5

@dataclass
class ExtractedHeading:
    """Data class for extracted heading information."""
    text: str
    document: str
    page: int
    font_size: float
    confidence: float = 0.0

class UltraLightEmbedder:
    """Ultra-lightweight similarity using keyword overlap and basic patterns."""
    
    def __init__(self):
        # South of France specific categories for travel planning
        self.travel_categories = {
            'activities': ['activity', 'activities', 'do', 'visit', 'see', 'tour', 'trip', 'experience', 'adventure'],
            'places': ['city', 'town', 'village', 'place', 'location', 'destination', 'region', 'area'],
            'culture': ['culture', 'history', 'art', 'museum', 'festival', 'tradition', 'heritage'],
            'food': ['restaurant', 'food', 'cuisine', 'wine', 'dining', 'eat', 'drink', 'taste'],
            'accommodation': ['hotel', 'stay', 'accommodation', 'lodge', 'resort', 'room'],
            'transport': ['travel', 'transport', 'train', 'bus', 'car', 'flight', 'boat']
        }
    
    def calculate_similarity(self, query: str, text: str) -> float:
        """Calculate similarity using multiple lightweight methods."""
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Extract words
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        
        if not query_words or not text_words:
            return 0.0
        
        # Method 1: Direct keyword overlap (50% weight)
        direct_overlap = len(query_words.intersection(text_words)) / len(query_words.union(text_words))
        
        # Method 2: Category matching (30% weight)
        category_score = 0.0
        query_categories = self._identify_categories(query_lower)
        text_categories = self._identify_categories(text_lower)
        
        if query_categories and text_categories:
            category_overlap = len(query_categories.intersection(text_categories))
            category_score = category_overlap / max(len(query_categories), len(text_categories))
        
        # Method 3: Character n-gram similarity (20% weight)
        char_score = self._char_similarity(query_lower, text_lower)
        
        # Combine with weights
        final_score = (direct_overlap * 0.5 + category_score * 0.3 + char_score * 0.2)
        
        return min(final_score, 1.0)
    
    def _identify_categories(self, text: str) -> set:
        """Identify which categories the text belongs to."""
        categories = set()
        for category, keywords in self.travel_categories.items():
            if any(keyword in text for keyword in keywords):
                categories.add(category)
        return categories
    
    def _char_similarity(self, text1: str, text2: str) -> float:
        """Simple character-based similarity."""
        if len(text1) < 3 or len(text2) < 3:
            return 0.0
        
        # Create character trigrams
        trigrams1 = set(text1[i:i+3] for i in range(len(text1)-2))
        trigrams2 = set(text2[i:i+3] for i in range(len(text2)-2))
        
        if not trigrams1 or not trigrams2:
            return 0.0
        
        return len(trigrams1.intersection(trigrams2)) / len(trigrams1.union(trigrams2))

class PDFProcessor:
    """Main PDF processing class with ultra-light similarity."""
    
    def __init__(self):
        self.embedder = UltraLightEmbedder()
    
    def extract_headings_from_pdf(self, pdf_path: str, job_context: str = "") -> List[ExtractedHeading]:
        """Extract potential headings from PDF using font analysis."""
        headings = []
        filename = Path(pdf_path).name
        
        with pdfplumber.open(pdf_path) as pdf:
            # Analyze font sizes
            font_sizes = {}
            for page in pdf.pages:
                for char in page.chars:
                    if 'size' in char and char['size'] > 0:
                        size = round(float(char['size']), 1)
                        font_sizes[size] = font_sizes.get(size, 0) + 1
            
            if not font_sizes:
                return headings
            
            max_font_size = max(font_sizes.keys())
            font_threshold = max_font_size * 0.7  # Only consider larger fonts as headings
            
            print(f"  📄 Processing {filename} - Font threshold: {font_threshold}pt")
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_headings = self._extract_page_headings(
                    page, page_num, filename, font_threshold
                )
                headings.extend(page_headings)
        
        print(f"    Extracted {len(headings)} potential headings")
        return headings
    
    def _extract_page_headings(self, page, page_num: int, filename: str, font_threshold: float) -> List[ExtractedHeading]:
        """Extract headings from a single page."""
        headings = []
        chars = page.chars
        
        if not chars:
            return headings
        
        # Group characters by line
        lines = {}
        for char in chars:
            if all(key in char for key in ['top', 'x0', 'text', 'size']):
                top = round(char['top'])
                if top not in lines:
                    lines[top] = []
                lines[top].append(char)
        
        # Process each line
        for line_chars in lines.values():
            line_chars.sort(key=lambda x: x['x0'])
            text = ''.join(c['text'] for c in line_chars).strip()
            
            if not text or len(text) < 3:
                continue
            
            avg_font_size = sum(float(c['size']) for c in line_chars) / len(line_chars)
            
            # Check if this looks like a heading
            if self._is_likely_heading(text, avg_font_size, font_threshold):
                headings.append(ExtractedHeading(
                    text=text,
                    document=filename,
                    page=page_num,
                    font_size=avg_font_size,
                    confidence=1.0
                ))
        
        return headings
    
    def _is_likely_heading(self, text: str, font_size: float, threshold: float) -> bool:
        """Determine if text is likely a heading."""
        if font_size < threshold:
            return False
        
        # Must be reasonably short
        if len(text) > 80 or len(text.split()) > 8:
            return False
        
        # Exclude common non-heading patterns
        text_lower = text.lower()
        if any(pattern in text_lower for pattern in ['•', 'step', 'note:', 'tip:', 'warning:']):
            return False
        
        # Prefer title case or all caps
        if text.istitle() or text.isupper():
            return True
        
        return False
    
    def rank_by_semantic_similarity(self, headings: List[ExtractedHeading], persona: str, job: str) -> List[Dict[str, Any]]:
        """Rank headings by semantic similarity using ultra-light method."""
        if not headings:
            return []
        
        query = f"{persona}: {job}"
        
        # Calculate similarities
        ranked_results = []
        # Calculate similarities and create results with temporary sort key
        ranked_results = []
        similarity_scores = []
        
        for i, heading in enumerate(headings):
            similarity = self.embedder.calculate_similarity(query, heading.text)
            
            ranked_results.append({
                'document': heading.document,
                'section_title': heading.text,
                'importance_rank': i + 1,
                'page_number': heading.page
            })
            similarity_scores.append(similarity)
        
        # Sort by similarity score using separate list
        sorted_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)
        ranked_results = [ranked_results[i] for i in sorted_indices]
        
        # Update ranks and return top sections
        for i, result in enumerate(ranked_results):
            result['importance_rank'] = i + 1
        
        return ranked_results[:TOP_SECTIONS]

class SubsectionExtractor:
    """Extract subsection content around identified headings."""
    
    def extract_context(self, pdf_path: str, page_num: int, job: str, 
                       target_heading: str = "", max_chars: int = 500) -> str:
        """Extract contextual text from around a heading."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]
                    full_text = page.extract_text() or ""
                    
                    if not full_text:
                        return ""
                    
                    # Find content around the target heading
                    if target_heading:
                        lines = full_text.split('\n')
                        for i, line in enumerate(lines):
                            if target_heading.lower() in line.lower():
                                # Extract surrounding context
                                start_idx = max(0, i - 2)
                                end_idx = min(len(lines), i + 5)
                                context = '\n'.join(lines[start_idx:end_idx])
                                return context[:max_chars]
                    
                    # Fallback: return first part of page text
                    return full_text[:max_chars]
        except Exception:
            pass
        return ""

def load_input_metadata() -> Tuple[str, str]:
    """Load persona and job requirements from input file."""
    try:
        with open(INPUT_META, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        persona = data.get('persona', {}).get('role', 'General User')
        job = data.get('job_to_be_done', {}).get('task', 'Extract relevant information')
        
        return persona, job
    except Exception as e:
        print(f"⚠  Could not load metadata: {e}")
        return "General User", "Extract relevant information"

def main():
    """Main processing function with proper output structure."""
    print("🚀 Starting Ultra-Minimal PDF Outline Extraction")
    
    # Load input metadata
    persona, job = load_input_metadata()
    print(f"👤 Persona: {persona}")
    print(f"🎯 Job: {job}")
    
    # Initialize processor
    processor = PDFProcessor()
    
    # Process PDF files
    all_headings = []
    pdf_files = list(Path(INPUT_DOCS).glob("*.pdf"))
    
    print(f"\n📚 Processing {len(pdf_files)} PDF files...")
    
    for pdf_path in pdf_files:
        try:
            headings = processor.extract_headings_from_pdf(pdf_path, job)
            all_headings.extend(headings)
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")
    
    print(f"\n✅ Extracted {len(all_headings)} total headings")
    
    # Rank by semantic similarity
    ranked_sections = processor.rank_by_semantic_similarity(all_headings, persona, job)
    
    # Extract subsection content
    extractor = SubsectionExtractor()
    subsections = []
    
    for section in ranked_sections:
        pdf_path = Path(INPUT_DOCS) / section['document']
        context = extractor.extract_context(
            pdf_path,
            section['page_number'],
            job,
            section['section_title']
        )
        if context:
            subsections.append({
                'document': section['document'],
                'refined_text': context,
                'page_number': section['page_number']
            })
    
    # Create output in the proper structure
    output = {
        'metadata': {
            'input_documents': [f.name for f in pdf_files],
            'persona': {'role': persona},
            'job_to_be_done': {'task': job},
            'processing_timestamp': datetime.now().isoformat()
        },
        'extracted_sections': ranked_sections,
        'subsection_analysis': subsections
    }
    
    # Save output
    os.makedirs(Path(OUTPUT_FILE).parent, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Output written to: {OUTPUT_FILE}")
    print("📊 Processing completed using ultra-minimal semantic similarity (no PyTorch)")

if __name__ == "__main__":
    main()
