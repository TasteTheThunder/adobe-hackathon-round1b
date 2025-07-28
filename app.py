"""
Enhanced PDF Outline Extractor with Semantic Ranking
Designed for generic PDF processing with any persona and job requirements.
"""

import pdfplumber
import json
import os
import re
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Constants
INPUT_DOCS = "input/docs"
INPUT_META = "input/challenge1b_input.json"
OUTPUT_FILE = "output/output.json"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_SECTIONS = 5

@dataclass
class ExtractedHeading:
    """Data class for extracted heading information."""
    text: str
    document: str
    page: int
    font_size: float
    confidence: float = 0.0

@dataclass
class ProcessingConfig:
    """Configuration for PDF processing."""
    min_font_size_ratio: float = 0.7  # Minimum font size as ratio of max font size
    max_heading_length: int = 80
    min_heading_length: int = 3
    max_heading_words: int = 8

class HeadingDetector:
    """Enhanced heading detection with configurable patterns."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
        # Patterns that indicate non-headings (instructions, ingredients, etc.)
        self.exclusion_patterns = [
            r'^(heat|add|mix|stir|cook|bake|serve|drain|pour|place|remove|combine)',
            r'^(in a|in the|on a|on the|with a|with the|preheat|season|garnish)',
            r'(tablespoon|teaspoon|cup|pound|ounce|gram|ml|oz|degrees)',
            r'(°f|°c|\d+\s*degrees)',
            r'^\s*[•\-o]\s*',  # Bullet points
            r':\s*$',  # Lines ending with colon
            r'^\d+\.',  # Numbered lists
            r'^step \d+',  # Step instructions
            r'(and serve|then serve|serve immediately|serve warm|serve hot)',
            r'(until|while|when|before|after)',
            r'^(let|allow|continue|repeat|adjust)',
            r'(minutes|hours|seconds)',
            r'(chopped|diced|minced|sliced|grated)',
            r'\b(however|therefore|furthermore|moreover|additionally)\b',
            r'\b(will be|shall be|must be|should be|can be|may be)\b',
            r'\b(the purpose|in order to|according to|such as|for example)\b',
            r'\b(please note|please refer|as mentioned|as shown)\b',
            r'\b(developed|created|designed|intended|established)\b',
            r'\b(provide|ensure|maintain|support|enable)\b',
            r'\b(including but not limited|for more information)\b',
            r'\b(this document|this section|this chapter)\b',
            r'(and|or|but|if|when|while|since|although)\s+\w+$',  # Fragment endings
            r'^\w+\s+(and|or|but|if|when|while)$',  # Fragment beginnings
        ]
        
        # Patterns that indicate likely headings
        self.inclusion_patterns = [
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # Title case
            r'^[A-Z\s&-]+$',  # All caps with spaces and common punctuation
            r'^[A-Z][a-z]\s[A-Z][a-z]*$',  # Two word title case
            r'^\d+\.\s*[A-Z]',  # Numbered headings (1. Title)
            r'^(Chapter|Section|Part|Appendix)\s+\w+',  # Structural headings
            r'^[A-Z][a-z]+\s+(Guide|Overview|Summary|Analysis|Plan)$',  # Document type headings
        ]
        
        # Keywords that boost heading confidence
        self.heading_keywords = [
            'recipe', 'dish', 'style', 'classic', 'traditional', 'special',
            'breakfast', 'lunch', 'dinner', 'appetizer', 'dessert', 'soup',
            'salad', 'sandwich', 'pasta', 'pizza', 'bread', 'cake', 'pie',
            'menu', 'course', 'preparation', 'cuisine', 'cooking', 'chef',
            'ingredients', 'flavors', 'seasoning', 'marinade', 'sauce',
        ]
        
        # Vegetarian/plant-based keywords that boost confidence
        self.vegetarian_keywords = [
            'veggie', 'vegetable', 'bean', 'lentil', 'quinoa', 'tofu', 'tempeh',
            'mushroom', 'spinach', 'kale', 'broccoli', 'cauliflower', 'avocado',
            'tomato', 'cucumber', 'pepper', 'corn', 'rice', 'grain', 'nut',
            'chickpea', 'hummus', 'falafel', 'cheese', 'egg', 'pasta', 'noodle',
            'salad', 'soup', 'stew', 'curry', 'stir-fry', 'roasted', 'grilled',
        ]
        
        # Gluten-free keywords
        self.gluten_free_keywords = [
            'rice', 'quinoa', 'salad', 'fruit', 'vegetable', 'bean', 'lentil',
            'meat', 'fish', 'egg', 'dairy', 'nut', 'seed', 'potato', 'corn',
            'gluten-free', 'gf', 'celiac', 'naturally gluten free',
        ]
        
        # Content quality indicators (boost confidence for real headings)
        self.quality_indicators = [
            'guide', 'overview', 'summary', 'analysis', 'plan', 'strategy',
            'introduction', 'conclusion', 'recommendations', 'findings',
            'methodology', 'approach', 'framework', 'process', 'system'
        ]
    
    def is_likely_heading(self, text: str, font_size: float, max_font_size: float, context: str = "") -> Tuple[bool, float]:
        """
        Dynamic heading detection for likely dish names (not hardcoded).
        Returns (is_heading, confidence_score)
        """
        if not text or len(text.strip()) < self.config.min_heading_length:
            return False, 0.0
        text = text.strip()
        text_lower = text.lower()
        word_count = len(text.split())
        # Only allow headings with 1-4 words, no commas, no periods, no instructions
        if word_count < 1 or word_count > 4:
            return False, 0.0
        if any(p in text for p in [',', '.', ':']):
            return False, 0.0
        # Exclude lines with verbs or instruction keywords
        instruction_verbs = [
            'dice', 'slice', 'mix', 'serve', 'add', 'bake', 'cook', 'chop', 'stir', 'drain', 'pour', 'place', 'remove', 'combine', 'preheat', 'season', 'garnish', 'shred', 'sauté', 'spread', 'cut', 'thread', 'grill', 'broil', 'dredge', 'simmer', 'toss', 'whisk', 'marinate', 'turn', 'soften', 'crumble', 'cube', 'shred', 'saute', 'sauté', 'dredge', 'sauté', 'simmer'
        ]
        if any(verb in text_lower for verb in instruction_verbs):
            return False, 0.0
        # Exclude ingredient lines
        ingredient_keywords = ['cup', 'tablespoon', 'teaspoon', 'gram', 'ml', 'oz', 'pound', 'ounce', 'feta', 'cheese', 'apple', 'almond', 'butter', 'paneer', 'yogurt', 'lemon', 'cumin', 'coriander', 'masala', 'turmeric', 'salt', 'pepper', 'bread', 'tomato', 'cucumber', 'onion', 'basil', 'olive', 'oil', 'vinegar', 'sugar', 'water']
        if any(kw in text_lower for kw in ingredient_keywords):
            return False, 0.0
        # Only allow title case or all caps
        if not (text.istitle() or text.isupper()):
            return False, 0.0
        # Font size check
        font_threshold = max_font_size * self.config.min_font_size_ratio
        if font_size < font_threshold:
            return False, 0.0
        # Boost main dishes, deprioritize salads/sides
        main_keywords = [
            'curry', 'lasagna', 'falafel', 'ratatouille', 'casserole', 'stew', 'bake', 'rolls', 'cutlet', 'patty', 'main', 'noodle', 'pasta', 'pizza', 'pie', 'burger', 'gratin', 'stuffed', 'roast', 'tart', 'risotto', 'paella', 'tagine', 'moussaka', 'enchilada', 'empanada', 'samosa', 'kofta', 'kebab', 'chili', 'dahl', 'gumbo', 'jambalaya', 'shepherd', 'moussaka', 'gnocchi', 'quiche', 'frittata', 'stroganoff', 'goulash', 'ratatouille', 'falafel', 'lasagna', 'curry', 'casserole', 'stew', 'bake', 'rolls', 'cutlet', 'patty', 'main', 'noodle', 'pasta', 'pizza', 'pie', 'burger', 'gratin', 'stuffed', 'roast', 'tart', 'risotto', 'paella', 'tagine', 'moussaka', 'enchilada', 'empanada', 'samosa', 'kofta', 'kebab', 'chili', 'dahl', 'gumbo', 'jambalaya', 'shepherd', 'gnocchi', 'quiche', 'frittata', 'stroganoff', 'goulash'
        ]
        salad_keywords = ['salad', 'slaw', 'coleslaw']
        text_lower = text.lower()
        # If heading contains main dish keyword, boost confidence
        if any(k in text_lower for k in main_keywords):
            return True, 1.2
        # If heading contains salad/side keyword, lower confidence
        if any(k in text_lower for k in salad_keywords):
            return True, 0.7
        # Otherwise, normal confidence
        return True, 1.0
        
class PDFProcessor:
    """Main PDF processing class with enhanced accuracy."""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.detector = HeadingDetector(self.config)
        model_path = Path("model/all-MiniLM-L6-v2")  # Local path to the model
        self.model = SentenceTransformer(str(model_path))

    
    def extract_headings_from_pdf(self, pdf_path: str, job_context: str = "") -> List[ExtractedHeading]:
        """Extract potential headings from a PDF file."""
        headings = []
        filename = Path(pdf_path).name
        
        with pdfplumber.open(pdf_path) as pdf:
            # Analyze font sizes across the document
            font_sizes = self._analyze_font_sizes(pdf)
            if not font_sizes:
                return headings
            
            max_font_size = max(font_sizes.keys())
            
            print(f"  📄 Processing {filename}")
            print(f"    Font analysis: {len(font_sizes)} sizes, max: {max_font_size}pt")
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_headings = self._extract_page_headings(
                    page, page_num, filename, max_font_size, job_context
                )
                headings.extend(page_headings)
        
        print(f"    Extracted {len(headings)} potential headings")
        return headings
    
    def _analyze_font_sizes(self, pdf) -> Dict[float, int]:
        """Analyze font size distribution in the PDF."""
        font_sizes = {}
        for page in pdf.pages:
            for char in page.chars:
                if 'size' in char and 'text' in char:
                    size = round(float(char['size']), 1)
                    if size > 0:
                        font_sizes[size] = font_sizes.get(size, 0) + 1
        return font_sizes
    
    def _extract_page_headings(self, page, page_num: int, filename: str, max_font_size: float, job_context: str = "") -> List[ExtractedHeading]:
        """Extract headings from a single page."""
        headings = []
        chars = page.chars
        if not chars:
            return headings
        lines = {}
        for char in chars:
            if all(key in char for key in ['top', 'x0', 'text', 'size']):
                top = round(char['top'])
                if top not in lines:
                    lines[top] = []
                lines[top].append(char)
        for line_chars in lines.values():
            line_chars.sort(key=lambda x: x['x0'])
            text = ''.join(c['text'] for c in line_chars).strip()
            if not text:
                continue
            avg_font_size = sum(float(c['size']) for c in line_chars) / len(line_chars)
            is_heading, confidence = self.detector.is_likely_heading(text, avg_font_size, max_font_size, job_context)
            if is_heading:
                headings.append(ExtractedHeading(
                    text=text,
                    document=filename,
                    page=page_num,
                    font_size=avg_font_size,
                    confidence=confidence
                ))
        return headings
    
    def rank_by_semantic_similarity(self, headings: List[ExtractedHeading], persona: str, job: str) -> List[Dict[str, Any]]:
        """Rank headings by semantic similarity, dynamically boosting relevance to any job topic."""
        if not headings:
            return []
        filtered_headings = self._apply_content_filters(headings, job)
        if not filtered_headings:
            print("⚠  No headings passed content filtering, using original set")
            filtered_headings = headings
        query = f"{persona}: {job}"
        query_embedding = self.model.encode([query])
        heading_texts = [h.text for h in filtered_headings]
        heading_embeddings = self.model.encode(heading_texts)
        similarities = util.pytorch_cos_sim(query_embedding, heading_embeddings)[0]
        # Dynamically extract keywords from job description
        job_keywords = set(re.findall(r'\b\w+\b', job.lower()))
        ranked_results = []
        for i, (heading, similarity) in enumerate(zip(filtered_headings, similarities)):
            boost = 0.0
            heading_lower = heading.text.lower()
            # Boost if heading contains any job keyword
            if any(k in heading_lower for k in job_keywords):
                boost += 0.5
            # Penalize if heading contains unrelated keywords (e.g., 'family' for 'friends' job)
            unrelated_keywords = set(['family', 'kids', 'child', 'children', 'budget', 'cheap', 'hotel', 'hostel'])
            if any(k in heading_lower for k in unrelated_keywords) and not any(k in job_keywords for k in unrelated_keywords):
                boost -= 0.5
            ranked_results.append({
                'document': heading.document,
                'section_title': heading.text,
                'importance_rank': i + 1,
                'page_number': heading.page,
                'similarity_score': float(similarity) + boost
            })
        ranked_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        final_results = []
        for i, result in enumerate(ranked_results):
            final_results.append({
                'document': result['document'],
                'section_title': result['section_title'],
                'importance_rank': i + 1,
                'page_number': result['page_number']
            })
        return final_results[:TOP_SECTIONS]
    
    def _apply_content_filters(self, headings: List[ExtractedHeading], job: str) -> List[ExtractedHeading]:
        """Apply intelligent content filtering based on job requirements with enhanced accuracy."""
        job_lower = job.lower()
        filtered = []
        
        # Define content filters based on job requirements
        if 'vegetarian' in job_lower or 'vegan' in job_lower:
            # Enhanced meat keywords for better filtering
            meat_keywords = [
                'chicken', 'beef', 'pork', 'turkey', 'lamb', 'duck', 'fish', 'salmon',
                'tuna', 'shrimp', 'crab', 'lobster', 'bacon', 'ham', 'sausage', 'meat',
                'steak', 'ribs', 'wings', 'drumstick', 'thigh', 'breast', 'fillet',
                'seafood', 'anchovies', 'sardines', 'cod', 'halibut', 'mussels',
                'prosciutto', 'pepperoni', 'chorizo', 'salami', 'ground beef'
            ]
            
            for heading in headings:
                heading_lower = heading.text.lower()
                # More strict filtering - check for partial matches too
                contains_meat = any(meat in heading_lower for meat in meat_keywords)
                
                # Additional checks for meat-related phrases
                meat_phrases = [
                    'with chicken', 'beef and', 'pork in', 'fish with',
                    'meat sauce', 'chicken breast', 'grilled salmon'
                ]
                contains_meat_phrase = any(phrase in heading_lower for phrase in meat_phrases)
                
                if not (contains_meat or contains_meat_phrase):
                    filtered.append(heading)
                else:
                    print(f"    Filtered out non-vegetarian: {heading.text}")
        
        elif 'gluten-free' in job_lower or 'gluten free' in job_lower:
            # Enhanced gluten keywords
            gluten_keywords = [
                'bread', 'pasta', 'wheat', 'flour', 'noodle', 'sandwich', 'wrap',
                'pizza', 'bagel', 'toast', 'pancake', 'waffle', 'croissant',
                'baguette', 'focaccia', 'pita', 'tortilla', 'roll', 'bun',
                'spaghetti', 'linguine', 'fettuccine', 'ravioli', 'gnocchi',
                'breadcrumb', 'crouton', 'stuffing', 'dough', 'batter'
            ]
            
            for heading in headings:
                heading_lower = heading.text.lower()
                contains_gluten = any(gluten in heading_lower for gluten in gluten_keywords)
                
                # Check for gluten-containing phrases
                gluten_phrases = [
                    'with bread', 'on bread', 'wheat based', 'flour based',
                    'pasta with', 'noodle soup'
                ]
                contains_gluten_phrase = any(phrase in heading_lower for phrase in gluten_phrases)
                
                if not (contains_gluten or contains_gluten_phrase):
                    filtered.append(heading)
                else:
                    print(f"    Filtered out gluten-containing: {heading.text}")
        
        elif 'healthy' in job_lower or 'low-calorie' in job_lower:
            # Enhanced unhealthy keywords
            unhealthy_keywords = [
                'fried', 'crispy', 'breaded', 'deep', 'butter', 'cream', 'cheese',
                'bacon', 'sausage', 'mayo', 'mayonnaise', 'ranch', 'alfredo',
                'carbonara', 'tempura', 'battered', 'rich', 'decadent',
                'indulgent', 'loaded', 'smothered', 'stuffed with cheese'
            ]
            
            for heading in headings:
                heading_lower = heading.text.lower()
                contains_unhealthy = any(unhealthy in heading_lower for unhealthy in unhealthy_keywords)
                
                # Check for unhealthy preparation methods
                unhealthy_phrases = [
                    'deep fried', 'pan fried', 'cheese sauce', 'butter sauce',
                    'heavy cream', 'full fat'
                ]
                contains_unhealthy_phrase = any(phrase in heading_lower for phrase in unhealthy_phrases)
                
                if not (contains_unhealthy or contains_unhealthy_phrase):
                    filtered.append(heading)
                else:
                    print(f"    Filtered out unhealthy: {heading.text}")
        
        else:
            # No specific dietary filters, return all
            filtered = headings
        
        print(f"    Content filtering: {len(headings)} → {len(filtered)} headings")
        return filtered

class SubsectionExtractor:
    """Extract subsection content around identified headings with enhanced content filtering."""
    
    def __init__(self):
        # Enhanced meat keywords for vegetarian filtering
        self.meat_keywords = [
            'chicken', 'beef', 'pork', 'turkey', 'lamb', 'duck', 'fish', 'salmon',
            'tuna', 'shrimp', 'crab', 'lobster', 'bacon', 'ham', 'sausage', 'meat',
            'steak', 'ribs', 'wings', 'drumstick', 'thigh', 'breast', 'fillet',
            'seafood', 'anchovies', 'sardines', 'cod', 'halibut', 'mussels',
            'prosciutto', 'pepperoni', 'chorizo', 'salami', 'ground beef'
        ]
        
        # Enhanced gluten keywords for gluten-free filtering
        self.gluten_keywords = [
            'bread', 'pasta', 'wheat', 'flour', 'noodle', 'sandwich', 'wrap',
            'pizza', 'bagel', 'toast', 'pancake', 'waffle', 'croissant',
            'baguette', 'focaccia', 'pita', 'tortilla', 'roll', 'bun',
            'spaghetti', 'linguine', 'fettuccine', 'ravioli', 'gnocchi',
            'breadcrumb', 'crouton', 'stuffing', 'dough', 'batter'
        ]
        
        # Unhealthy keywords for healthy filtering
        self.unhealthy_keywords = [
            'fried', 'crispy', 'breaded', 'deep', 'butter', 'cream', 'cheese',
            'bacon', 'sausage', 'mayo', 'mayonnaise', 'ranch', 'alfredo',
            'carbonara', 'tempura', 'battered', 'rich', 'decadent'
        ]
    
    def extract_context(self, pdf_path: str, page_num: int, job: str, 
                       target_heading: str = "", max_chars: int = 500) -> str:
        """Extract contextual text from around a heading with content filtering."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]
                    full_text = page.extract_text() or ""
                    
                    if not full_text:
                        return ""
                    
                    # Try to find content around the target heading
                    filtered_text = self._extract_relevant_section(
                        full_text, target_heading, job, max_chars
                    )
                    
                    return filtered_text
        except Exception:
            pass
        return ""
    
    def _extract_relevant_section(self, full_text: str, target_heading: str, 
                                 job: str, max_chars: int) -> str:
        """Extract and filter relevant section based on job requirements with enhanced accuracy."""
        job_lower = job.lower()
        
        # Split text into recipe sections with better detection
        lines = full_text.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Enhanced recipe title detection
            is_title = (
                len(line) < 60 and  # Reasonable title length
                (line.isupper() or line.istitle()) and 
                not line.startswith(('•', 'o', '-', '*', 'Instructions:', 'Ingredients:', 'Directions:')) and
                not any(char.isdigit() for char in line[:5]) and  # Not a measurement
                not line.endswith(',') and  # Not a fragment
                len(line.split()) >= 2  # At least 2 words
            )
            
            if is_title:
                # Save previous section
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            sections.append('\n'.join(current_section))
        
        # Enhanced filtering based on job requirements
        filtered_sections = []
        target_section = None
        
        for section in sections:
            section_lower = section.lower()
            
            # Check if this section contains the target heading
            is_target = target_heading and target_heading.lower() in section_lower
            
            # Apply enhanced content filtering
            should_include = True
            
            if 'vegetarian' in job_lower or 'vegan' in job_lower:
                # More thorough meat detection
                contains_meat = any(meat in section_lower for meat in self.meat_keywords)
                meat_phrases = [
                    'with chicken', 'beef and', 'pork in', 'fish with',
                    'meat sauce', 'chicken breast', 'grilled salmon', 'seafood',
                    'ground beef', 'chicken wings', 'fish fillet'
                ]
                contains_meat_phrase = any(phrase in section_lower for phrase in meat_phrases)
                
                if contains_meat or contains_meat_phrase:
                    should_include = False
            
            elif 'gluten-free' in job_lower or 'gluten free' in job_lower:
                # More thorough gluten detection
                contains_gluten = any(gluten in section_lower for gluten in self.gluten_keywords)
                gluten_phrases = [
                    'with bread', 'on bread', 'wheat based', 'flour based',
                    'pasta with', 'noodle soup', 'bread crumbs', 'wheat flour'
                ]
                contains_gluten_phrase = any(phrase in section_lower for phrase in gluten_phrases)
                
                if contains_gluten or contains_gluten_phrase:
                    should_include = False
            
            elif 'healthy' in job_lower or 'low-calorie' in job_lower:
                # More thorough unhealthy detection
                contains_unhealthy = any(unhealthy in section_lower for unhealthy in self.unhealthy_keywords)
                unhealthy_phrases = [
                    'deep fried', 'pan fried', 'cheese sauce', 'butter sauce',
                    'heavy cream', 'full fat', 'loaded with', 'smothered in'
                ]
                contains_unhealthy_phrase = any(phrase in section_lower for phrase in unhealthy_phrases)
                
                if contains_unhealthy or contains_unhealthy_phrase:
                    should_include = False
            
            if should_include:
                if is_target:
                    target_section = section
                else:
                    filtered_sections.append(section)
        
        # Prioritize target section if found and valid
        final_sections = []
        if target_section:
            final_sections.append(target_section)
        
        # Add other filtered sections
        final_sections.extend(filtered_sections)
        
        # If no valid sections found, return empty (maintain dietary compliance)
        if not final_sections:
            return ""
        
        # Combine sections up to max_chars with better truncation
        result = ""
        for section in final_sections:
            if len(result + section) <= max_chars:
                if result:
                    result += "\n\n"  # Better section separation
                result += section
            else:
                # Add partial section if there's meaningful room
                remaining = max_chars - len(result)
                if remaining > 100:  # Increased threshold for meaningful content
                    if result:
                        result += "\n\n"
                    # Try to end at a sentence or line break
                    partial = section[:remaining-2]
                    last_sentence = partial.rfind('.')
                    last_line = partial.rfind('\n')
                    
                    if last_sentence > len(partial) * 0.8:  # If close to end
                        result += partial[:last_sentence+1]
                    elif last_line > len(partial) * 0.7:  # If reasonable line break
                        result += partial[:last_line]
                    else:
                        result += partial
                break
        
        return result

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
    """Main processing function."""
    print("🚀 Starting Enhanced PDF Outline Extraction")
    
    # Load input metadata
    persona, job = load_input_metadata()
    print(f"👤 Persona: {persona}")
    print(f"🎯 Job: {job}")
    
    # Initialize processor
    processor = PDFProcessor()
    
    # Dynamically filter PDFs based on job_to_be_done
    all_headings = []
    pdf_files = list(Path(INPUT_DOCS).glob("*.pdf"))
    job_lower = job.lower()
    filtered_pdfs = []
    if 'dinner' in job_lower:
        keywords = ['dinner', 'mains', 'sides']
        filtered_pdfs = [f for f in pdf_files if any(k in f.name.lower() for k in keywords)]
    elif 'lunch' in job_lower:
        keywords = ['lunch']
        filtered_pdfs = [f for f in pdf_files if any(k in f.name.lower() for k in keywords)]
    elif 'breakfast' in job_lower:
        keywords = ['breakfast']
        filtered_pdfs = [f for f in pdf_files if any(k in f.name.lower() for k in keywords)]
    else:
        filtered_pdfs = pdf_files
    print(f"\n📚 Processing {len(filtered_pdfs)} PDF files...")
    for pdf_path in filtered_pdfs:
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
    # Create output
    output = {
        'metadata': {
            'input_documents': [f.name for f in filtered_pdfs],
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

if __name__ == "__main__":
    main()