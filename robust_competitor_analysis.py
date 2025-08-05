import re
import json
from collections import Counter, defaultdict
from typing import List, Dict, Union, Tuple
import openai
from difflib import SequenceMatcher

class RobustCompetitorAnalyzer:
    """
    A more robust system for extracting and analyzing competitors from AI search results.
    Uses multiple extraction methods and AI validation for better accuracy.
    """
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.brand_name = None
        self.results = []
        
    def analyze_competitors(self, results: List[Dict], brand_name: str, skip_ai_validation: bool = False) -> Dict:
        """
        Main method to analyze competitors with multiple validation steps.
        """
        self.brand_name = brand_name
        self.results = results
        
        # Step 1: Extract potential competitors using multiple methods
        candidates = self._extract_candidates_multiple_methods()
        
        # Step 2: Clean and validate candidates
        cleaned_candidates = self._clean_and_validate_candidates(candidates)
        
        # Step 3: Use AI to validate and categorize competitors (optional)
        if skip_ai_validation:
            validated_competitors = cleaned_candidates[:10]  # Take top 10 without AI validation
        else:
            validated_competitors = self._ai_validate_competitors(cleaned_candidates)
        
        # Step 4: Count mentions with fuzzy matching
        mention_counts = self._count_mentions_with_fuzzy_matching(validated_competitors)
        
        # Step 5: Calculate market share
        market_shares = self._calculate_market_shares(mention_counts)
        
        return {
            'competitors': validated_competitors,
            'mention_counts': mention_counts,
            'market_shares': market_shares,
            'total_mentions': sum(mention_counts.values())
        }
    
    def _extract_candidates_multiple_methods(self) -> List[str]:
        """
        Extract potential competitor names using multiple regex patterns and AI assistance.
        """
        combined_text = "\n".join([r.get('response', '') for r in self.results])
        candidates = []
        
        # Method 1: Traditional regex patterns
        patterns = [
            r'\*\*\[([^]]+)\]\(',  # **[Name](URL)
            r'^##\s+([^\n]+)',     # ## Name
            r'^[•–-]\s+([^(\n]+?)(?:\s*\([^)]*\))?\s*$',  # • Name, – Name, - Name
            r'^\d+\.\s+\*\*([^*]+)\*\*',  # 1. **Name**
            r'(?<!\[)\*\*([^*\[\]]+)\*\*(?!\])',  # **Name** (plain bold)
            r'^#\s+([^\n]+)',      # # Name
            r'^##\s+([^\n]+)',     # ## Name
            r'^###\s+([^\n]+)',    # ### Name
            r'^\*\s+([^\n]+)',     # * Name
            r'^-\s+([^\n]+)',      # - Name
            r'^\d+\.\s+([^\n]+)',  # 1. Name
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, combined_text, re.MULTILINE)
            candidates.extend(matches)
        
        # Method 2: Extract capitalized phrases (potential company names)
        capitalized_phrases = re.findall(r'\b[A-Z][a-zA-Z\s&]+(?:Inc\.|Ltd\.|LLC|Corp\.|Company|Co\.)?\b', combined_text)
        candidates.extend(capitalized_phrases)
        
        # Method 3: Extract phrases in quotes
        quoted_names = re.findall(r'"([^"]+)"', combined_text)
        candidates.extend(quoted_names)
        
        return candidates
    
    def _clean_and_validate_candidates(self, candidates: List[str]) -> List[str]:
        """
        Clean and validate candidate names using multiple filters.
        """
        cleaned = []
        
        for candidate in candidates:
            name = candidate.strip()
            
            # Skip if too short or too long
            if len(name) < 3 or len(name) > 100:
                continue
                
            # Skip if contains obvious non-company patterns
            skip_patterns = [
                r'Rate:', r'Price:', r'Cost:', r'Fee:', r'Free:', r'Discount:',
                r'Contact:', r'Email:', r'Phone:', r'Address:', r'Website:',
                r'Click here', r'Learn more', r'Read more', r'Visit',
                r'^\d+$',  # Just numbers
                r'^[A-Z\s]+$',  # All caps (likely not a company name)
            ]
            
            if any(re.search(pattern, name, re.IGNORECASE) for pattern in skip_patterns):
                continue
            
            # Skip if it's the user's brand
            if self._is_same_brand(name):
                continue
            
            # Skip if it's a common word or phrase
            if self._is_common_word(name):
                continue
            
            # Skip likely client companies (non-video production companies)
            if self._is_likely_client_not_competitor(name):
                continue
            
            # Clean up the name
            name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
            name = re.sub(r'[^\w\s&.-]', '', name)  # Remove special chars except &, ., -
            name = name.strip()
            
            if name and len(name) >= 3:
                cleaned.append(name)
        
        return list(set(cleaned))  # Remove duplicates
    
    def _is_same_brand(self, name: str) -> bool:
        """
        Check if a name is the same as the user's brand using fuzzy matching.
        """
        if not self.brand_name:
            return False
        
        # Exact match
        if name.lower() == self.brand_name.lower():
            return True
        
        # Fuzzy match (similarity > 0.8)
        similarity = SequenceMatcher(None, name.lower(), self.brand_name.lower()).ratio()
        return similarity > 0.8
    
    def _is_common_word(self, name: str) -> bool:
        """
        Check if a name is a common word that's unlikely to be a company name.
        """
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'within', 'without',
            'best', 'top', 'leading', 'premium', 'quality', 'service', 'solutions',
            'company', 'business', 'enterprise', 'group', 'team', 'agency',
            'melbourne', 'sydney', 'brisbane', 'perth', 'adelaide', 'australia',
            'australian', 'local', 'national', 'international', 'global'
        }
        
        words = name.lower().split()
        return all(word in common_words for word in words)
    
    def _is_likely_client_not_competitor(self, name: str) -> bool:
        """
        Check if a name is likely a client company rather than a video production competitor.
        Uses industry keywords and known major brands to filter out non-competitors.
        """
        name_lower = name.lower()
        
        # Major corporations that are likely clients, not video production competitors
        major_client_brands = {
            'telstra', 'anz', 'nab', 'westpac', 'cba', 'commonwealth bank',
            'qantas', 'jetstar', 'virgin', 'toyota', 'ford', 'holden', 'bmw', 'mercedes',
            'woolworths', 'coles', 'bunnings', 'harvey norman', 'jb hi-fi',
            'netflix', 'disney', 'amazon', 'google', 'microsoft', 'apple', 'facebook', 'meta',
            'coca-cola', 'pepsi', 'mcdonalds', 'kfc', 'subway', 'dominos',
            'university of melbourne', 'monash university', 'rmit', 'deakin university',
            'australian government', 'victorian government', 'city of melbourne',
            'australian red cross', 'salvation army', 'red bull', 'nike', 'adidas',
            'myer', 'david jones', 'target', 'kmart', 'big w',
            'channel nine', 'channel seven', 'channel ten', 'abc', 'sbs',
            'medibank', 'bupa', 'ahm', 'hcf', 'nib',
            'seek', 'realestate.com.au', 'domain', 'carsguide', 'autotrader',
            'ibm', 'oracle', 'salesforce', 'atlassian', 'canva',
            'bhp', 'rio tinto', 'fortescue', 'santos', 'origin energy',
            'cricket australia', 'afl', 'nrl', 'tennis australia'
        }
        
        # Check if it's a known major brand that would be a client
        for brand in major_client_brands:
            if brand in name_lower or name_lower in brand:
                return True
        
        # Industry keywords that suggest non-video production companies
        non_video_industry_keywords = [
            'bank', 'insurance', 'airline', 'university', 'government', 'council',
            'hospital', 'medical', 'pharmacy', 'retail', 'supermarket', 'restaurant',
            'hotel', 'resort', 'real estate', 'mining', 'energy', 'telecommunications',
            'automotive', 'fashion', 'sports', 'tourism', 'finance', 'investment',
            'construction', 'engineering', 'consulting', 'accounting', 'legal',
            'healthcare', 'pharmaceutical', 'technology', 'software', 'hardware'
        ]
        
        # Check if name contains industry keywords suggesting it's not a video production company
        words = name_lower.split()
        for word in words:
            if any(keyword in word for keyword in non_video_industry_keywords):
                return True
        
        return False
    
    def _ai_validate_competitors(self, candidates: List[str]) -> List[str]:
        """
        Use AI to validate and filter competitor names.
        """
        if not candidates:
            return []
        
        # Create a prompt for AI validation
        prompt = f"""
        I have extracted potential competitor names from video production search results. 
        Please identify which ones are actual VIDEO PRODUCTION/MEDIA/FILM companies (not client brands).
        
        EXCLUDE these types of companies:
        - Banks (ANZ, NAB, Telstra)
        - Retail brands (Woolworths, Myer, Harvey Norman)
        - Tech companies (Google, Microsoft, Netflix)
        - Universities and government agencies
        - Any non-media/video production companies
        
        INCLUDE only:
        - Video production companies
        - Film studios
        - Animation studios
        - Media agencies
        - Creative production houses
        
        User's brand: {self.brand_name}
        Potential competitors: {', '.join(candidates[:20])}  # Limit to first 20
        
        Return only the valid VIDEO PRODUCTION company names as a JSON array.
        Example: ["Video Company A", "Film Studio B", "Media Agency C"]
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at identifying company and brand names from text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                validated = json.loads(result)
                if isinstance(validated, list):
                    return validated
            except json.JSONDecodeError:
                pass
            
            # Fallback: extract names from text response
            names = re.findall(r'"([^"]+)"', result)
            if names:
                return names
            
            # Final fallback: return original candidates
            return candidates[:10]  # Limit to top 10
            
        except Exception as e:
            print(f"[Error] AI validation failed: {e}")
            return candidates[:10]  # Return top 10 as fallback
    
    def _count_mentions_with_fuzzy_matching(self, competitors: List[str]) -> Dict[str, int]:
        """
        Count mentions using fuzzy matching to handle variations in company names.
        """
        counts = defaultdict(int)
        
        for result in self.results:
            text = result.get('response', '').lower()
            
            for competitor in competitors:
                # Exact match
                if competitor.lower() in text:
                    counts[competitor] += 1
                    continue
                
                # Fuzzy match for variations
                words = competitor.lower().split()
                if len(words) >= 2:
                    # Check if most words match
                    matches = sum(1 for word in words if word in text)
                    if matches >= len(words) * 0.7:  # 70% match threshold
                        counts[competitor] += 1
                
                # Check for common variations
                variations = self._get_name_variations(competitor)
                for variation in variations:
                    if variation.lower() in text:
                        counts[competitor] += 1
                        break
        
        return dict(counts)
    
    def _get_name_variations(self, name: str) -> List[str]:
        """
        Generate common variations of a company name.
        """
        variations = [name]
        
        # Remove common suffixes
        suffixes = [' Inc', ' LLC', ' Ltd', ' Corp', ' Company', ' Co']
        for suffix in suffixes:
            if name.endswith(suffix):
                variations.append(name[:-len(suffix)])
        
        # Add common suffixes
        if not any(name.endswith(suffix) for suffix in suffixes):
            variations.extend([name + suffix for suffix in suffixes])
        
        # Abbreviate multi-word names
        words = name.split()
        if len(words) > 1:
            abbreviation = ''.join(word[0].upper() for word in words)
            variations.append(abbreviation)
        
        return variations
    
    def _calculate_market_shares(self, mention_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate market share percentages for each competitor.
        """
        total_mentions = sum(mention_counts.values())
        if total_mentions == 0:
            return {}
        
        shares = {}
        for competitor, count in mention_counts.items():
            shares[competitor] = (count / total_mentions) * 100
        
        return shares

def create_robust_competitor_analyzer(openai_client):
    """
    Factory function to create a RobustCompetitorAnalyzer instance.
    """
    return RobustCompetitorAnalyzer(openai_client) 