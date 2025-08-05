from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
import tempfile
from datetime import datetime
import re
from typing import List, Dict, Union
import pandas as pd
import concurrent.futures
from collections import Counter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import markdown2
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
from dotenv import load_dotenv
import openai
import google.generativeai as genai
from robust_competitor_analysis import create_robust_competitor_analyzer


# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="AI Brand Visibility Checker")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files (create static directory if it doesn't exist)
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/download/{filename}")
async def download_pdf(filename: str):
    """
    Serve the generated PDF file for download.
    """
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/pdf', filename=filename)
    return {"error": "File not found"}

# Templates
templates = Jinja2Templates(directory="templates")

# Progress tracking
progress_store = {}
analysis_results = {}

# Model Configuration
MODELS = {
    "GPT-4o": "gpt-4o",
    "GPT-4o-mini": "gpt-4o-mini",
    "Gemini-2.5-Flash": "gemini-2.5-flash"
}

# Assign model for each task (optimised for speed + functionality)
PROMPT_MODEL     = MODELS["Gemini-2.5-Flash"]    # Fast for prompts
SEARCH_MODEL     = MODELS["GPT-4o-mini"]         # Must use OpenAI for web search
SENTIMENT_MODEL  = MODELS["Gemini-2.5-Flash"]    # Fast for sentiment
COMPETITOR_MODEL = MODELS["Gemini-2.5-Flash"]    # Fast for competitor analysis
INSIGHT_MODEL    = MODELS["Gemini-2.5-Flash"]    # Fast for insights
RECOMMEND_MODEL  = MODELS["GPT-4o-mini"]         # Keep quality for recommendations

# Search-capable models (only OpenAI models support web search)
SEARCH_CAPABLE_MODELS = ["GPT-4o", "GPT-4o-mini"]  # Only these can do web search
SEARCH_MODELS_FAST = ["GPT-4o-mini", "GPT-4o-mini", "GPT-4o-mini"]  # Fast search models

# All models for comparison (includes Gemini without web search)
ALL_MODELS_FOR_COMPARISON = ["GPT-4o", "GPT-4o-mini", "Gemini-2.5-Flash"]

# App Configuration
MAX_PARALLEL = 6  # Increased from 3 - test your OpenAI rate limits
temperature = 0.5

# Australian spellings
AUS_SPELLINGS = [
    (r'(?i)optimi[sz]e', 'optimise'),
    (r'(?i)optimi[sz]ation', 'optimisation'),
    (r'(?i)analy[sz]e', 'analyse'),
    (r'(?i)analy[sz]ing', 'analysing'),
    (r'(?i)analy[sz]ed', 'analysed'),
    (r'(?i)analy[sz]er', 'analyser'),
    (r'(?i)color', 'colour'),
    (r'(?i)favor', 'favour'),
    (r'(?i)organize', 'organise'),
    (r'(?i)organizing', 'organising'),
    (r'(?i)organizer', 'organiser'),
    (r'(?i)center', 'centre'),
    (r'(?i)locali[sz]e', 'localise'),
    (r'(?i)locali[sz]ation', 'localisation'),
    (r'(?i)behavior', 'behaviour'),
    (r'(?i)summarize', 'summarise'),
    (r'(?i)summarizing', 'summarising'),
    (r'(?i)summarized', 'summarised'),
    (r'(?i)program', 'programme'), # only for non-software context
]

def aus_spell(text):
    for pattern, replacement in AUS_SPELLINGS:
        text = re.sub(pattern, replacement, text)
    return text

def call_llm(model_name: str, system_prompt: str, user_prompt: str, max_tokens: int = 150, temperature: float = 0.5) -> str:
    """
    Unified function to call either OpenAI or Gemini models
    """
    try:
        print(f"[DEBUG] Calling LLM: {model_name}")  # Debug print
        if model_name.startswith("gemini"):
            # Use Gemini
            print(f"[DEBUG] Using Gemini model: {model_name}")
            # Configure safety settings to be less restrictive
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ]
            
            model = genai.GenerativeModel(model_name)
            full_prompt = f"{system_prompt}\n\nUser: {user_prompt}"
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                ),
                safety_settings=safety_settings
            )
            
            # Check if response is valid
            if response.parts:
                result = response.text.strip()
                print(f"[DEBUG] Gemini response length: {len(result)}")
                return result
            else:
                print(f"[WARNING] Gemini blocked response (finish_reason: {response.candidates[0].finish_reason}), falling back to OpenAI")
                # Fallback to OpenAI if Gemini blocks the content
                resp = openai.chat.completions.create(
                    model="gpt-4o-mini",  # Use GPT-4o-mini as fallback
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                result = resp.choices[0].message.content.strip()
                print(f"[DEBUG] OpenAI fallback response length: {len(result)}")
                return result
        else:
            # Use OpenAI
            print(f"[DEBUG] Using OpenAI model: {model_name}")
            resp = openai.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            result = resp.choices[0].message.content.strip()
            print(f"[DEBUG] OpenAI response length: {len(result)}")
            return result
    except Exception as e:
        print(f"[Error] LLM call failed for {model_name}: {e}")
        return ""

def validate_business_inputs(keyword: str, brand_name: str) -> dict:
    """
    Validate that the keyword and brand name are legitimate business-related inputs.
    Returns a dictionary with 'valid' boolean and 'error' message if invalid.
    """
    # Clean inputs
    keyword = keyword.strip()
    brand_name = brand_name.strip()
    
    # Check for empty inputs
    if not keyword or not brand_name:
        return {"valid": False, "error": "Please provide both a keyword and brand name."}
    
    # Check minimum length
    if len(keyword) < 2 or len(brand_name) < 2:
        return {"valid": False, "error": "Keyword and brand name must be at least 2 characters long."}
    
    # Check for gibberish patterns
    gibberish_patterns = [
        r'^[aeiou]{5,}$',  # Too many consecutive vowels
        r'^[bcdfghjklmnpqrstvwxyz]{5,}$',  # Too many consecutive consonants
        r'^(.)\1{4,}$',  # Same character repeated 5+ times
        r'^[^a-zA-Z\s]{3,}$',  # Only symbols/numbers (3+ chars)
        r'^\d+$',  # Only numbers
        r'^[!@#$%^&*(),.?":{}|<>]+$',  # Only special characters
    ]
    
    for pattern in gibberish_patterns:
        if re.search(pattern, keyword.lower()) or re.search(pattern, brand_name.lower()):
            return {"valid": False, "error": "Please enter meaningful business-related keywords and brand names."}
    
    # Check for common non-business words
    non_business_keywords = {
        'test', 'testing', 'hello', 'world', 'example', 'sample', 'demo',
        'asdf', 'qwerty', 'abcd', 'xyz', 'lorem', 'ipsum', 'dummy',
        'random', 'gibberish', 'nonsense', 'blah', 'whatever', 'nothing',
        'something', 'anything', 'stuff', 'thing', 'things', 'junk'
    }
    
    keyword_words = set(keyword.lower().split())
    brand_words = set(brand_name.lower().split())
    
    if keyword_words.issubset(non_business_keywords) or brand_words.issubset(non_business_keywords):
        return {"valid": False, "error": "Please enter a business service or industry as your keyword, and a real brand name."}
    
    # Check for potentially business-related keywords using LLM validation
    try:
        validation_prompt = f"""
        Determine if these inputs are related to legitimate business services, industries, or brands:
        
        Keyword: "{keyword}"
        Brand Name: "{brand_name}"
        
        Valid business keywords include: services, products, industries, professions, etc.
        Valid brand names include: company names, business names, organization names, etc.
        
        Invalid examples: random words, gibberish, personal names unrelated to business, test data, etc.
        
        Respond with only "VALID" or "INVALID".
        """
        
        result = call_llm(
            PROMPT_MODEL,
            "You are a business input validator. Determine if inputs are legitimate business-related terms.",
            validation_prompt,
            10,
            0.1
        )
        
        if "INVALID" in result.upper():
            return {"valid": False, "error": "Please enter a legitimate business service/industry as your keyword and a real brand/company name."}
            
    except Exception as e:
        print(f"[Warning] LLM validation failed: {e}")
        # Continue with basic validation if LLM fails
    
    return {"valid": True, "error": ""}

def validate_custom_prompt(prompt: str) -> dict:
    """
    Validate that a custom search prompt is business-related and meaningful.
    """
    prompt = prompt.strip()
    
    if len(prompt) < 5:
        return {"valid": False, "error": "Search queries must be at least 5 characters long."}
    
    # Check for gibberish patterns
    if re.search(r'^(.)\1{4,}$', prompt.lower()) or re.search(r'^[^a-zA-Z\s]{5,}$', prompt):
        return {"valid": False, "error": "Please enter a meaningful search query."}
    
    # Check for test/dummy content
    test_words = {'test', 'testing', 'hello', 'asdf', 'qwerty', 'gibberish', 'random', 'dummy'}
    if any(word in prompt.lower() for word in test_words):
        return {"valid": False, "error": "Please enter a real business search query, not test data."}
    
    # Should contain business-related terms
    business_indicators = [
        'company', 'companies', 'business', 'service', 'services', 'agency', 'agencies',
        'firm', 'firms', 'provider', 'providers', 'consultant', 'consultants',
        'studio', 'studios', 'shop', 'shops', 'store', 'stores', 'professional',
        'expert', 'specialists', 'contractor', 'contractors', 'vendor', 'vendors',
        'best', 'top', 'leading', 'affordable', 'cheap', 'premium', 'quality',
        'local', 'near me', 'melbourne', 'sydney', 'australia', 'australian'
    ]
    
    if not any(indicator in prompt.lower() for indicator in business_indicators):
        return {"valid": False, "error": "Search queries should be business-related (e.g., 'best web design companies')."}
    
    return {"valid": True, "error": ""}

def generate_prompts_for_keyword(keyword: str) -> list[str]:
    if not keyword.strip():
        return []
    system_prompt = (
        "You are an expert in generating realistic business search queries that people use when looking for local companies and services. "
        "Create 5 diverse search queries based on a user-provided keyword using this structure: "
        "[Descriptor] + [Service Keyword] + [Location if provided]. "
        "Use descriptors like: 'Top', 'Best', 'Leading', 'Affordable', 'Award-winning', 'Budget-friendly', 'Professional', 'Expert', 'Recommended', 'Local', 'Premier', 'High-end', 'Reliable'. "
        "Examples: 'Top video production companies in Melbourne', 'Leading corporate video production services Melbourne', 'Best affordable web design agencies Sydney'. "
        "Make queries sound like real people searching for businesses to hire. "
        "Include the year 2025 in only one query (like 'Best [service] companies 2025'). "
        "Return as a numbered list, each on a new line."
    )
    user_prompt = f"Generate 5 business search queries for '{keyword}'. Focus on queries people use when looking for companies to hire, using descriptors like 'top', 'best', 'leading', etc."
    try:
        print(f"[DEBUG] generate_prompts_for_keyword using model: {PROMPT_MODEL}")
        content = call_llm(PROMPT_MODEL, system_prompt, user_prompt, 150, temperature)
        print(f"[DEBUG] Generated content: {content[:200]}...")  # Show first 200 chars
        prompts = re.findall(r"^\d+\.\s*(.*)", content, re.MULTILINE)
        print(f"[DEBUG] Extracted prompts: {prompts}")
        return [p.strip() for p in prompts if p.strip()]
    except Exception as e:
        print(f"[Error] generate_prompts_for_keyword: {e}")
        return []

def query_model_with_or_without_search(model_name: str, prompt: str, brand_name: str, timeout: int = 30) -> dict:
    import time
    
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            start_time = time.time()
            model_id = MODELS.get(model_name, model_name)
            
            # Check if this model can do web search
            if model_name in SEARCH_CAPABLE_MODELS:
                # Use OpenAI API with search functionality
                resp = openai.responses.create(
                    model=model_id,
                    tools=[{
                        "type": "web_search_preview",
                        "user_location": {
                            "type": "approximate",
                            "country": "AU"
                        }
                    }],
                    input=prompt
                )
                text = resp.output_text
                print(f"[DEBUG] {model_name} web search response length: {len(text)}")
            else:
                # Use regular LLM call without web search (for Gemini)
                system_prompt = (
                    "You are a comprehensive business directory expert with extensive knowledge of companies across industries. "
                    "When asked about companies, provide detailed responses with specific company names, descriptions, and key details. "
                    "Include information about their services, reputation, notable clients, and any distinguishing features. "
                    "Focus on real, well-known businesses that would appear in search results. "
                    "Provide at least 10-15 companies with substantial details about each. "
                    "Do not give generic advice about how to find companies - instead, list actual company names with comprehensive information."
                )
                text = call_llm(model_id, system_prompt, prompt, 1500, 0.7)  # Increased from 300 to 1500 tokens for fuller responses
                print(f"[DEBUG] {model_name} non-search response length: {len(text)}")
            
            duration = time.time() - start_time
            print(f"[TIMING] {model_name} took: {duration:.2f}s")
            
            mentioned = bool(re.search(rf"\b{re.escape(brand_name)}\b", text, re.IGNORECASE))
            return {"model": model_name, "prompt": prompt, "response": text, "brand_mentioned": mentioned}
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"[Error] query_model ({model_name}) attempt {attempt + 1}: {e} (took {duration:.2f}s)")
            
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"[Retry] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                return {"model": model_name, "prompt": prompt, "response": "", "brand_mentioned": False}

def batch_sentiment_analysis(texts: tuple[str], brand_name: str) -> list[str]:
    """
    OPTIMIZED sentiment analysis with parallel processing, smaller batches, and faster model
    Returns list of sentiment classifications: 'Positive', 'Negative', or 'Neutral'
    """
    if not texts:
        return []
    
    def analyze_single_text(text: str) -> str:
        """Analyse sentiment for a single text"""
        try:
            sentiment = call_llm(
                SENTIMENT_MODEL,
                "Analyse sentiment. Reply only: Positive, Negative, or Neutral",
                text[:500],  # Limit text length
                5,  # max tokens
                0.0  # temperature
            )
            if sentiment not in ["Positive", "Negative", "Neutral"]:
                sentiment = "Neutral"
            return sentiment
            
        except Exception as e:
            print(f"[Error] Sentiment analysis failed: {e}")
            return "Neutral"
    
    # PARALLEL PROCESSING with smaller batches
    results = []
    batch_size = 3  # Reduced from processing all at once
    
    # Process in smaller batches with parallel execution
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"[TIMING] Processing sentiment batch {i//batch_size + 1} ({len(batch)} items)")
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch), 3)) as executor:
            batch_results = list(executor.map(analyze_single_text, batch))
        
        results.extend(batch_results)
        print(f"[TIMING] Batch {i//batch_size + 1} completed")
    
    return results

def extract_top_competitors_by_mentions(
    results: List[Union[Dict, str]],
    brand_name: str,
    top_n: int = 10
) -> List[str]:
    """
    Finds the top N competitor names by parsing out:
      - **[Name](URL)
      - ## Name
      - • Name  or  – Name
      - - Name
      - 1. **Name**
      - **Name** (plain bold)
    and then cleans out listicle artifacts (e.g. "Rate: $150"), standalone lowercase words, etc.
    """
    # 1) Combine all responses into one text blob
    texts = []
    for r in results:
        if isinstance(r, dict):
            texts.append(r.get('response', ''))
        elif isinstance(r, str):
            texts.append(r)
    combined = "\n".join(texts)

    # 2) Pull every candidate name via regex
    candidates = []
    candidates += re.findall(r"\*\*\[([^]]+)\]\(", combined)                     # **[Name](URL)
    candidates += re.findall(r"^##\s+([^\n]+)", combined, re.MULTILINE)           # ## Name
    bullets = re.findall(r"^[•–-]\s+([^(\n]+?)(?:\s*\([^)]*\))?\s*$",
                         combined, re.MULTILINE)
    candidates += [b.strip() for b in bullets]                                   # • Name, – Name or - Name
    candidates += re.findall(r"^\d+\.\s+\*\*([^*]+)\*\*", combined, re.MULTILINE) # 1. **Name**
    bolds = re.findall(r"(?<!\[)\*\*([^*\[\]]+)\*\*(?!\])", combined)            # **Name** plain
    candidates += [b.strip() for b in bolds if len(b.strip()) > 3]

    # 3) Clean & filter out junk
    cleaned = []
    for name in candidates:
        n = name.strip()
        # strip trailing "(…)" or "– description"
        n = re.sub(r'\s*\([^)]*\)$', '', n)
        n = re.sub(r'\s*–.*$', '', n).strip()
        # drop anything still containing parentheses or asterisks
        if '(' in n or '*' in n:
            continue
        # drop single-word all-lowercase (e.g. "melbourne")
        if n.islower():
            continue
        # require multi-word or ends with uppercase (proper nouns)
        if ' ' not in n and not n[-1].isupper():
            continue
        # drop obvious listicle lines like "Rate: $150" or "Rate: $100"
        if re.match(r'Rate:', n):
            continue
        # drop your own brand
        if n.lower() == brand_name.lower():
            continue

        cleaned.append(n)

    # 4) Count and take top N
    freq = Counter(cleaned)
    print("[DEBUG] extracted competitors:", freq.most_common(15))
    return [name for name, _ in freq.most_common(top_n)]

def count_competitor_mentions(
    results: List[Dict[str, str]],
    competitors: List[str]
) -> Dict[str, int]:
    """
    Unified function to count competitor mentions across all results.
    Uses consistent regex patterns to ensure plot and summary data match.
    """
    counts = {}
    
    for comp in competitors:
        esc = re.escape(comp)
        cnt = 0
        
        for r in results:
            text = r.get('response', '')
            # Use the same patterns as in analyze_competitor_insights
            patterns = [
                rf"\*\*\[{esc}\]\(",                    # **[Name](URL)
                rf"^##\s+{esc}",                        # ## Name
                rf"^[•–-]\s+{esc}",                     # • Name, – Name, - Name
                rf"^\d+\.\s+\*\*{esc}\*\*",            # 1. **Name**
                rf"(?<!\[)\*\*{esc}\*\*(?!\])"          # **Name** (plain bold)
            ]
            
            for p in patterns:
                cnt += len(re.findall(p, text, re.MULTILINE))
        
        counts[comp] = cnt
    
    return counts

def analyze_competitor_insights(
    results: List[Dict[str, str]],
    brand_name: str
) -> str:
    # 1) Get top competitors
    competitors = extract_top_competitors_by_mentions(results, brand_name, top_n=10)

    # 2) Count competitor mentions and user's brand mentions
    competitor_counts = count_competitor_mentions(results, competitors)
    brand_mentions = sum(1 for r in results if r['brand_mentioned'])
    
    # 3) Combine all brands for comprehensive analysis
    all_counts = {brand_name: brand_mentions, **competitor_counts}
    total = sum(all_counts.values())

    if total == 0:
        return "## Competitor Analysis\n\n**No brand mentions found in AI search results.**"

    # 4) Compute share% and categorize all brands (including user's brand)
    shares = {c: (cnt/total)*100 for c, cnt in all_counts.items()}
    sorted_brands = sorted(shares.items(), key=lambda x: x[1], reverse=True)
    
    # Categorize all brands by visibility (including user's brand)
    dominant = [c for c, s in sorted_brands if s >= 20]  # 20%+ share
    strong = [c for c, s in sorted_brands if 10 <= s < 20]  # 10-19% share
    emerging = [c for c, s in sorted_brands if 5 <= s < 10]  # 5-9% share
    niche = [c for c, s in sorted_brands if s < 5]  # <5% share
    
    # 5) Build structured analysis prompt including user's brand
    brand_data = "\n".join([f"{c}: {shares[c]:.1f}% visibility" for c, _ in sorted_brands[:11]])  # Include user brand + top 10 competitors
    
    prompt = f"""
Analyse this competitive landscape data (including {brand_name}'s position) and provide structured insights:

{brand_data}

Provide analysis in this EXACT structure:

## Market Leadership Analysis
[Identify the top 2-3 competitors and what makes them dominant in AI search results]

## Competitive Positioning Insights
[Analyse the competitive landscape - who are the key players, market concentration, gaps]

Do NOT include specific percentages or mention counts in your response. Focus on market positioning.
"""

    # 5) Send to model for structured analysis
    try:
        analysis = call_llm(
            INSIGHT_MODEL,
            "You are a strategic marketing analyst specialising in competitive intelligence and market positioning. Provide structured, actionable insights.",
            prompt,
            800,
            0.2
        )
        
        # 6) Add brand positioning and competitor categorization summary
        category_summary = "\n\n---\n\n### Brand Positioning Overview\n\n"
        
        # Highlight user's brand position
        user_share = shares.get(brand_name, 0)
        user_rank = next((i+1 for i, (c, _) in enumerate(sorted_brands) if c == brand_name), "Not ranked")
        category_summary += f"**🏢 Your Brand ({brand_name}):** {user_share:.1f}% visibility (Rank #{user_rank})\n\n"
        
        # Separate competitors from user's brand
        competitor_dominant = [c for c in dominant if c != brand_name]
        competitor_strong = [c for c in strong if c != brand_name]
        competitor_emerging = [c for c in emerging if c != brand_name]
        competitor_niche = [c for c in niche if c != brand_name]
        
        category_summary += "### Competitor Categories\n\n"
        if competitor_dominant:
            category_summary += f"**🏆 Market Leaders:** {', '.join(competitor_dominant)}\n\n"
        if competitor_strong:
            category_summary += f"**💪 Strong Competitors:** {', '.join(competitor_strong)}\n\n"
        if competitor_emerging:
            category_summary += f"**📈 Emerging Players:** {', '.join(competitor_emerging)}\n\n"
        if competitor_niche:
            category_summary += f"**🎯 Niche Players:** {', '.join(competitor_niche[:5])}" + (" and others" if len(competitor_niche) > 5 else "") + "\n\n"
        
        return analysis + category_summary
        
    except Exception as e:
        print("[Error] analyze_competitor_insights:", e)
        return "## Competitor Analysis\n\n**Error generating competitor insights. Please try again.**"

def generate_competitor_insights_from_data(all_counts: dict, all_shares: dict, brand_name: str) -> str:
    """
    Generate competitor insights using the exact same data as the pie chart
    """
    try:
        if not all_counts or not all_shares:
            return "## Competitor Analysis\n\n**No competitor data available for analysis.**"
        
        total_mentions = sum(all_counts.values())
        if total_mentions == 0:
            return "## Competitor Analysis\n\n**No brand mentions found in AI search results.**"
        
        # Sort brands by share (same as pie chart)
        sorted_brands = sorted(all_shares.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize all brands by visibility (same logic as original)
        dominant = [c for c, s in sorted_brands if s >= 20]  # 20%+ share
        strong = [c for c, s in sorted_brands if 10 <= s < 20]  # 10-19% share
        emerging = [c for c, s in sorted_brands if 5 <= s < 10]  # 5-9% share
        niche = [c for c, s in sorted_brands if s < 5]  # <5% share
        
        # Debug output to verify data consistency
        print(f"[DEBUG] Pie chart data - Total mentions: {total_mentions}")
        print(f"[DEBUG] Pie chart data - All counts: {all_counts}")
        print(f"[DEBUG] Pie chart data - All shares: {dict(list(all_shares.items())[:5])}")
        
        # Build structured analysis prompt with exact same data
        brand_data = "\n".join([f"{c}: {s:.1f}% visibility ({all_counts.get(c, 0)} mentions)" 
                               for c, s in sorted_brands[:11]])  # Include user brand + top 10 competitors
        
        prompt = f"""
Analyse this competitive landscape data (including {brand_name}'s position) and provide structured insights:

{brand_data}

Provide analysis in this EXACT structure:

## Market Leadership Analysis
[Identify the top 2-3 competitors and what makes them dominant in AI search results]

## Competitive Positioning Insights
[Analyse the competitive landscape - who are the key players, market concentration, gaps]

Do NOT include specific percentages or mention counts in your response. Focus on market positioning.
"""

        # Generate insights using LLM
        analysis = call_llm(
            INSIGHT_MODEL,
            "You are a strategic marketing analyst specialising in competitive intelligence and market positioning. Provide structured, actionable insights.",
            prompt,
            800,
            0.2
        )
        
        # Add brand positioning summary with exact same percentages as pie chart
        category_summary = "\n\n---\n\n### Brand Positioning Overview\n\n"
        
        # Highlight user's brand position using pie chart data
        user_share = all_shares.get(brand_name, 0)
        user_rank = next((i+1 for i, (c, _) in enumerate(sorted_brands) if c == brand_name), "Not ranked")
        user_mentions = all_counts.get(brand_name, 0)
        category_summary += f"**🏢 Your Brand ({brand_name}):** {user_share:.1f}% visibility ({user_mentions} mentions) - Rank #{user_rank}\n\n"
        
        # Separate competitors from user's brand
        competitor_dominant = [c for c in dominant if c != brand_name]
        competitor_strong = [c for c in strong if c != brand_name]
        competitor_emerging = [c for c in emerging if c != brand_name]
        competitor_niche = [c for c in niche if c != brand_name]
        
        category_summary += "### Competitor Categories\n\n"
        if competitor_dominant:
            category_summary += f"**🏆 Market Leaders:** {', '.join(competitor_dominant)}\n\n"
        if competitor_strong:
            category_summary += f"**💪 Strong Competitors:** {', '.join(competitor_strong)}\n\n"
        if competitor_emerging:
            category_summary += f"**📈 Emerging Players:** {', '.join(competitor_emerging)}\n\n"
        if competitor_niche:
            category_summary += f"**🎯 Niche Players:** {', '.join(competitor_niche[:5])}" + (" and others" if len(competitor_niche) > 5 else "") + "\n\n"
        
        return analysis + category_summary
        
    except Exception as e:
        print(f"[Error] generate_competitor_insights_from_data: {e}")
        return "## Competitor Analysis\n\n**Error generating competitor insights. Please try again.**"

def generate_recommendations(brand_name: str, keyword: str) -> str:
    """
    Generate structured GEO/AEO best practices personalised for the brand and keyword.
    No AI model needed - uses predefined best practices with personalization.
    """
    
    # Personalize the content based on brand and keyword
    brand_context = f"{brand_name} in the {keyword} industry"
    keyword_context = f"{keyword} businesses"
    
    recommendations = f"""# 🧠 Generative Engine Optimization (GEO) / AEO Best Practices for {brand_name}

Here's a list of **best practices for Generative Engine Optimisation (GEO)** - also referred to as **Answer Engine Optimisation (AEO)** - for {brand_context}, optimised for tools like **ChatGPT, Google's SGE (Search Generative Experience), Perplexity, and other LLM-powered search engines**.

---

## 🏗️ 1. **Use Structured, Fact-Based Content**

* Provide **clear answers** to {keyword}-related questions.
* Use **concise headings** (H2/H3) to match query patterns:
  *e.g., "What is {keyword} for {brand_name}?"*
* Bullet points, tables, and FAQs help LLMs extract answers better.

---

## 🔍 2. **Target Long-Tail, Question-Based Keywords**

* Optimize content around **natural-language queries**, e.g.:

  * "How does {keyword} work for {brand_name}?"
  * "Best {keyword} solutions for {brand_name} 2025"
* Use tools like **AlsoAsked, Answer the Public**, and **Google's 'People Also Ask'** for inspiration.

---

## 📑 3. **Implement Schema Markup**

* Use structured data (JSON-LD) like:

  * `FAQ`, `HowTo`, `Product`, `Review`, `Organization`
* Helps both LLMs and traditional search engines understand {brand_name}'s context.

---

## ✍️ 4. **Write with Retrieval-Augmented Generation (RAG) in Mind**

* Use **explicit mentions** of {brand_name} and {keyword}:

  > "At {brand_name}, we offer tailored {keyword} solutions..."
* Helps LLMs **retrieve** and **attribute** answers to your site.

---

## 🔗 5. **Create Internal Answer Hubs**

* Build **cluster content** around {keyword} topics:

  * One main pillar page + multiple supporting pages
* Cross-link them semantically:

  > "Learn more about [{keyword} solutions](#)"

---

## 📱 6. **Optimize for Voice & Conversational Queries**

* Use natural phrasing and full-sentence questions about {keyword}
* Include featured snippets-style answers near the top of your page

---

## 🔁 7. **Keep Content Fresh and Updated**

* LLMs often prioritize **current and verified** content.
* Add date references:

  > "As of August 2025, {brand_name}'s {keyword} solutions..."
* Update statistics and citations regularly.

---

## 🔍 8. **Gain Citations from Reputable Sources**

* LLMs trust pages **referenced elsewhere**.
* Aim for:

  * Backlinks from authority sites in {keyword} industry
  * Mentions in news/media or respected directories

---

## 🧩 9. **Be Brand-Identifiable in LLMs**

* Include {brand_name} **with key {keyword} topics**:

  > "{brand_name} explains that {keyword}..."
* Mention your **location and niche** for better contextual attribution

---

## 🛠️ 10. **Monitor and Test Appearance in AI Tools**

* Regularly test how your {keyword} content appears in:

  * Google SGE
  * Bing Copilot
  * Perplexity
  * ChatGPT (e.g., via browsing plugins or web tools)
* Adjust based on visibility gaps

---

## 📋 BONUS: Technical SEO Still Matters

* Fast-loading, mobile-friendly pages
* Clean HTML, crawlable structure
* Use `<title>`, `<meta description>`, `<h1>` properly for {keyword} content

---

### 🎯 **Priority Actions for {brand_name}**

1. **Immediate (Week 1-2):** Implement schema markup for {keyword} pages
2. **Short-term (Month 1):** Create FAQ content around {keyword} questions
3. **Medium-term (Month 2-3):** Build internal linking structure for {keyword} topics
4. **Long-term (Ongoing):** Monitor AI tool visibility and adjust strategy

*These best practices are specifically tailored for {brand_name} in the {keyword} industry to maximize visibility in AI-powered search engines.*"""
    
    return recommendations

# Helper to add markdown (except tables) to story
def add_markdown_to_story(md_text, story, style):
    html = markdown2.markdown(md_text)
    # Split on <p> tags, <ul>, <ol>, <li>, or double newlines for paragraphs and lists
    blocks = re.split(r'(?:<p>|</p>|<ul>|</ul>|<ol>|</ol>|<li>|</li>|\n\n)', html)
    for block in blocks:
        block = block.strip()
        if block:
            story.append(Paragraph(block, style))
            story.append(Spacer(1, 8))  # Add spacing after each paragraph or block

# Helper to add markdown tables to story
def add_markdown_table_to_story(md_table, story):
    lines = [line for line in md_table.splitlines() if '|' in line]
    if not lines:
        return
    data = [ [cell.strip() for cell in re.split(r'\s*\|\s*', line)[1:-1]] for line in lines ]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

def generate_pdf_report(brand_name, keyword, main_score, summary, avg_sent, perception, summary_df, comp_insights, recommendations, pie_img_path):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.2*inch, bottomMargin=1*inch)
    styles = getSampleStyleSheet()
    story = []
    warning_message = ""

    # Custom Styles
    styles.add(ParagraphStyle(name='Justify', alignment=TA_LEFT, fontName='Helvetica', fontSize=10, leading=14))
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER, fontName='Helvetica', fontSize=10, leading=14))
    styles.add(ParagraphStyle(name='Title', fontName='Helvetica-Bold', fontSize=24, alignment=TA_CENTER, spaceAfter=16))
    styles.add(ParagraphStyle(name='h1', fontName='Helvetica-Bold', fontSize=18, spaceAfter=12, textColor=colors.HexColor('#4f46e5')))
    styles.add(ParagraphStyle(name='h2', fontName='Helvetica-Bold', fontSize=14, spaceAfter=10, textColor=colors.HexColor('#4f46e5')))
    styles.add(ParagraphStyle(name='h3', fontName='Helvetica-Bold', fontSize=12, spaceAfter=8, textColor=colors.HexColor('#4f46e5')))

    # Title
    story.append(Paragraph(aus_spell("AI Brand Visibility Report"), styles['Title']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(aus_spell(f"<b>Brand:</b> {brand_name}"), styles['Center']))
    story.append(Paragraph(aus_spell(f"<b>Keyword:</b> {keyword}"), styles['Center']))
    story.append(Spacer(1, 0.3*inch))

    # Main Score Section
    story.append(Paragraph(aus_spell("Your Visibility Grade"), styles['h1']))
    story.append(Paragraph(main_score, ParagraphStyle('Score', parent=styles['h1'], alignment=TA_CENTER, fontSize=48, textColor=colors.HexColor('#10b981'))))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(summary, styles['Justify']))
    story.append(Spacer(1, 0.3*inch))

    # Sentiment
    story.append(Paragraph("Sentiment Analysis", styles['h1']))
    story.append(Paragraph(f"<b>Average Sentiment Score:</b> {avg_sent:.2f}", styles['Justify']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Sentiment Summary</b>", styles['h2']))
    add_markdown_to_story(perception, story, styles['Justify'])
    story.append(Spacer(1, 0.3*inch))

    # Brand Mentions Table
    if not summary_df.empty:
        story.append(Paragraph(aus_spell("Brand Mentions Table"), styles['h1']))
        story.append(Spacer(1, 0.2*inch))
        table_data = [summary_df.columns.tolist()] + summary_df.values.tolist()
        table = Table(table_data, repeatRows=1, colWidths=[doc.width/len(summary_df.columns)]*len(summary_df.columns))
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))

    # Competitor Insights
    if comp_insights:
        story.append(Paragraph(aus_spell("Competitor Insights"), styles['h1']))
        story.append(Spacer(1, 0.2*inch))
        if pie_img_path and os.path.exists(pie_img_path):
            try:
                pie_img = Image(pie_img_path, width=6*inch, height=4.5*inch)
                pie_img.hAlign = 'CENTER'
                story.append(pie_img)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"[Warning] Could not load competitor pie chart image: {e}")
        add_markdown_to_story(comp_insights, story, styles['Justify'])
        story.append(Spacer(1, 0.3*inch))

    # Recommendations
    if recommendations:
        story.append(Paragraph(aus_spell("GEO/AEO Best Practices"), styles['h1']))
        story.append(Spacer(1, 0.2*inch))
        add_markdown_to_story(recommendations, story, styles['Justify'])

    def header_footer(canvas, doc):
        canvas.saveState()
        # Header
        logo_path = os.path.join(os.path.dirname(__file__), "apws_logo.png")
        if os.path.exists(logo_path):
            canvas.drawImage(logo_path, 40, doc.height + 0.5*inch, width=0.8*inch, height=0.8*inch, mask='auto')
        canvas.setFont('Helvetica', 9)
        canvas.drawString(1.5*inch, doc.height + 0.75*inch, "A.P. Web Solutions")
        canvas.drawString(1.5*inch, doc.height + 0.6*inch, "AI Brand Visibility Report")
        canvas.line(40, doc.height + 0.4*inch, doc.width + 40, doc.height + 0.4*inch)
        
        # Footer
        canvas.setFont('Helvetica', 9)
        canvas.drawString(40, 0.75*inch, "www.apwebsolutions.com.au")
        canvas.drawRightString(doc.width + 40, 0.75*inch, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    pdf_data = buffer.getvalue()
    buffer.close()
    temp_dir = tempfile.gettempdir()
    clean_brand_name = re.sub(r'[^a-zA-Z0-9]', '_', brand_name)
    filename = f"AI_Brand_Visibility_Score_{clean_brand_name}_APWS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "wb") as f:
        f.write(pdf_data)
    return file_path, warning_message

def make_competitor_pie(labels, values, brand_name=None, image_path=None):
    # Create pull effect - highlight user's brand more prominently
    pull_values = []
    colors = []
    
    for i, label in enumerate(labels):
        if brand_name and label == brand_name:
            pull_values.append(0.15)  # Pull user's brand slice out more
            colors.append('#4f46e5')  # Distinctive color for user's brand
        else:
            pull_values.append(0.05)  # Standard pull for competitors
            colors.append(None)  # Use default colors
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=0.3,
        marker_colors=colors if any(colors) else None,
        textinfo='percent+label',
        pull=pull_values
    )])
    
    fig.update_layout(
        title={
            'text': f"AI Search Visibility: {brand_name} vs Competitors" if brand_name else "Competitor Visibility Share",
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
    )
    
    if image_path:
        fig.write_image(image_path, width=500, height=400, engine='kaleido')
    return fig

def save_matplotlib_pie(labels, values, image_path):
    fig, ax = plt.subplots(figsize=(8, 8))  # Increased size for better PDF visibility
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax.set_title("AI Search Visibility Share", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(image_path, dpi=300, bbox_inches='tight')  # Higher DPI for better quality
    plt.close(fig)

def create_seaborn_pie(labels, values, brand_name=None, image_path=None):
    """
    Create a beautiful pie chart using Seaborn with custom styling.
    More reliable and visually appealing than Plotly.
    """
    # Set Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Create custom color palette with more variety
    vibrant_colors = [
        '#FF6B6B',  # Coral Red
        '#4ECDC4',  # Turquoise
        '#45B7D1',  # Sky Blue
        '#96CEB4',  # Mint Green
        '#FFEAA7',  # Warm Yellow
        '#DDA0DD',  # Plum
        '#98D8C8',  # Sea Green
        '#F7DC6F',  # Golden Yellow
        '#BB8FCE',  # Lavender
        '#85C1E9',  # Light Blue
        '#F8C471',  # Orange
        '#82E0AA',  # Light Green
        '#F1948A',  # Light Coral
        '#D7BDE2',  # Light Purple
        '#F39C12',  # Orange
        '#E74C3C',  # Red
        '#9B59B6',  # Purple
        '#3498DB',  # Blue
        '#1ABC9C',  # Teal
        '#F1C40F',  # Yellow
        '#E67E22',  # Carrot Orange
        '#34495E',  # Wet Asphalt
        '#16A085',  # Green Sea
        '#8E44AD',  # Wisteria
        '#2C3E50',  # Midnight Blue
    ]
    
    if brand_name and brand_name in labels:
        # Highlight user's brand with distinctive color
        colors = []
        for i, label in enumerate(labels):
            if label == brand_name:
                colors.append('#4f46e5')  # Distinctive blue for user's brand
            else:
                colors.append(vibrant_colors[i % len(vibrant_colors)])
    else:
        colors = [vibrant_colors[i % len(vibrant_colors)] for i in range(len(labels))]
    
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add some spacing between slices for better visual separation
    explode = [0.05 if label == brand_name else 0.02 for label in labels]
    
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=labels, 
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        pctdistance=0.85
    )
    
    # Enhance text appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # Add title
    title = f"AI Search Visibility: {brand_name} vs Competitors" if brand_name else "Competitor Visibility Share"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
    
    # Add legend
    ax.legend(
        wedges, 
        labels,
        title="Brands",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=10
    )
    
    plt.tight_layout()
    
    if image_path:
        plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    return fig

def create_plotly_enhanced_pie(labels, values, brand_name=None, image_path=None):
    """
    Enhanced Plotly pie chart with better styling and reliability.
    """
    # Create custom colors with more variety
    vibrant_colors = [
        '#FF6B6B',  # Coral Red
        '#4ECDC4',  # Turquoise
        '#45B7D1',  # Sky Blue
        '#96CEB4',  # Mint Green
        '#FFEAA7',  # Warm Yellow
        '#DDA0DD',  # Plum
        '#98D8C8',  # Sea Green
        '#F7DC6F',  # Golden Yellow
        '#BB8FCE',  # Lavender
        '#85C1E9',  # Light Blue
        '#F8C471',  # Orange
        '#82E0AA',  # Light Green
        '#F1948A',  # Light Coral
        '#D7BDE2',  # Light Purple
        '#F39C12',  # Orange
        '#E74C3C',  # Red
        '#9B59B6',  # Purple
        '#3498DB',  # Blue
        '#1ABC9C',  # Teal
        '#F1C40F',  # Yellow
        '#E67E22',  # Carrot Orange
        '#34495E',  # Wet Asphalt
        '#16A085',  # Green Sea
        '#8E44AD',  # Wisteria
        '#2C3E50',  # Midnight Blue
    ]
    
    if brand_name and brand_name in labels:
        colors = []
        for i, label in enumerate(labels):
            if label == brand_name:
                colors.append('#4f46e5')  # Distinctive blue
            else:
                colors.append(vibrant_colors[i % len(vibrant_colors)])
    else:
        colors = [vibrant_colors[i % len(vibrant_colors)] for i in range(len(labels))]
    
    # Create pull effect for user's brand
    pull_values = [0.15 if label == brand_name else 0.05 for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=0.3,
        marker_colors=colors[:len(labels)],
        textinfo='percent+label',
        pull=pull_values,
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{label}</b><br>Share: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': f"AI Search Visibility: {brand_name} vs Competitors" if brand_name else "Competitor Visibility Share",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        showlegend=True,
        legend=dict(
            orientation="v", 
            yanchor="middle", 
            y=0.5, 
            xanchor="left", 
            x=1.05,
            font=dict(size=12)
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=60, b=50, l=50, r=200),
        width=800,  # Explicit width
        height=600  # Explicit height
    )
    
    if image_path:
        try:
            fig.write_image(image_path, width=600, height=500, engine='kaleido')
        except Exception as e:
            print(f"[Warning] Could not save Plotly image: {e}")
    
    return fig

def analyze_competitors_robust(results: List[Dict[str, str]], brand_name: str) -> Dict:
    """
    Use the robust competitor analyzer for better accuracy.
    """
    try:
        analyzer = create_robust_competitor_analyzer(openai)
        analysis = analyzer.analyze_competitors(results, brand_name)
        
        print(f"[DEBUG] Robust analysis found {len(analysis['competitors'])} competitors")
        print(f"[DEBUG] Competitors: {analysis['competitors']}")
        print(f"[DEBUG] Mention counts: {analysis['mention_counts']}")
        
        return analysis
    except Exception as e:
        print(f"[Error] Robust competitor analysis failed: {e}")
        # Fallback to original method
        return analyze_competitors_fallback(results, brand_name)

def analyze_competitors_fallback(results: List[Dict[str, str]], brand_name: str) -> Dict:
    """
    Fallback method using the original approach.
    """
    competitors = extract_top_competitors_by_mentions(results, brand_name, top_n=10)
    competitor_counts = count_competitor_mentions(results, competitors)
    total_mentions = sum(competitor_counts.values())
    
    market_shares = {}
    if total_mentions > 0:
        market_shares = {comp: (count / total_mentions) * 100 for comp, count in competitor_counts.items()}
    
    return {
        'competitors': competitors,
        'mention_counts': competitor_counts,
        'market_shares': market_shares,
        'total_mentions': total_mentions
    }

def generate_demo_data(keyword: str, brand_name: str) -> dict:
    """
    Generate demo data for quick testing without API calls.
    """
    import random
    
    # Sample prompts
    prompts = [
        f"Best {keyword} companies in Australia 2025",
        f"How to find reliable {keyword} services",
        f"Top {keyword} providers near me"
    ]
    
    # Sample models
    models = list(MODELS.keys())
    
    # Generate sample results
    results = []
    summary_map = {p: {} for p in prompts}
    
    for prompt in prompts:
        for model in models:
            # Randomly decide if brand is mentioned (70% chance for demo)
            brand_mentioned = random.choice([True, True, True, False])  # 75% chance
            response_text = f"Sample response for {prompt} using {model}. "
            if brand_mentioned:
                response_text += f"{brand_name} is mentioned as a leading provider."
            else:
                response_text += "Other companies are mentioned."
            
            results.append({
                "model": model,
                "prompt": prompt,
                "response": response_text,
                "brand_mentioned": brand_mentioned
            })
            summary_map[prompt][model] = brand_mentioned
    
    # Calculate summary
    rows, mentions = [], 0
    for p, vals in summary_map.items():
        cnt = sum(vals.values())
        mentions += cnt
        rows.append({'Prompt': p, **{m: ('✅' if v else '❌') for m, v in vals.items()}, 'Score': f"{cnt}/{len(MODELS)}"})
    summary_df = pd.DataFrame(rows)
    
    # Sample sentiment
    avg_sent = random.uniform(0.3, 0.8)
    perception = f"Demo sentiment analysis for {brand_name}: Generally positive perception with some neutral mentions."
    
    # Sample competitor data
    sample_competitors = ["Competitor A", "Competitor B", "Competitor C", "Competitor D"]
    competitor_counts = {comp: random.randint(1, 5) for comp in sample_competitors}
    brand_mentions = sum(1 for r in results if r['brand_mentioned'])
    
    # Calculate shares
    all_counts = {brand_name: brand_mentions, **competitor_counts}
    total_mentions = sum(all_counts.values())
    all_shares = {c: (cnt / total_mentions) * 100 for c, cnt in all_counts.items()}
    
    # Generate pie chart
    pie_img_path = os.path.join(tempfile.gettempdir(), f"demo_pie_{brand_name}_{keyword}.png")
    try:
        create_seaborn_pie(list(all_shares.keys()), list(all_shares.values()), brand_name, pie_img_path)
        plotly_pie = create_plotly_enhanced_pie(list(all_shares.keys()), list(all_shares.values()), brand_name)
    except Exception as e:
        print(f"[Warning] Demo pie chart creation failed: {e}")
        plotly_pie = None
    
    # Sample competitor insights
    comp_insights = f"""## Market Leadership Analysis

**Competitor A** and **Competitor B** dominate the {keyword} market with strong brand recognition and comprehensive service offerings.

## Competitive Positioning Insights

The {keyword} industry shows moderate market concentration with several established players. {brand_name} has opportunities to differentiate through specialised services and local market focus.

---

### Brand Positioning Overview

**🏢 Your Brand ({brand_name}):** {all_shares.get(brand_name, 0):.1f}% visibility (Rank #2)

### Competitor Categories

**🏆 Market Leaders:** Competitor A, Competitor B

**💪 Strong Competitors:** Competitor C

**📈 Emerging Players:** Competitor D

**🎯 Niche Players:** Various smaller providers"""
    
    # Generate recommendations
    recommendations = generate_recommendations(brand_name, keyword)
    
    # Calculate visibility percentage
    visibility_pct = mentions / (len(prompts) * len(MODELS)) * 100
    
    # Calculate demo grade using same logic as main function
    if visibility_pct >= 90:
        grade = "A+"
        summary = f"🎉 Demo: Excellent! Your brand achieved {visibility_pct:.1f}% visibility across AI search results. Outstanding performance!"
    elif visibility_pct >= 80:
        grade = "A"
        summary = f"🎉 Demo: Excellent! Your brand achieved {visibility_pct:.1f}% visibility across AI search results. Strong AI presence!"
    elif visibility_pct >= 70:
        grade = "B+"
        summary = f"👍 Demo: Very Good! Your brand achieved {visibility_pct:.1f}% visibility across AI search results. Above average performance."
    elif visibility_pct >= 60:
        grade = "B"
        summary = f"👍 Demo: Good! Your brand achieved {visibility_pct:.1f}% visibility across AI search results. Solid presence with room to grow."
    elif visibility_pct >= 50:
        grade = "C+"
        summary = f"⚠️ Demo: Fair. Your brand achieved {visibility_pct:.1f}% visibility across AI search results. Moderate presence - improvement needed."
    else:
        grade = "C"
        summary = f"⚠️ Demo: Fair. Your brand achieved {visibility_pct:.1f}% visibility across AI search results. Below average - focus on AI optimization."
    
    # Generate PDF report
    try:
        pdf_path, logo_warning = generate_pdf_report(
            brand_name, keyword, grade, summary, 
            avg_sent, perception, summary_df, comp_insights, recommendations, pie_img_path
        )
    except Exception as e:
        print(f"[Warning] Demo PDF generation failed: {e}")
        pdf_path = None
        logo_warning = None
    
    # Generate demo sentiment analysis
    demo_sentiment_analysis = {
        "strengths": [
            "High-quality products and services",
            "Excellent customer support team",
            "Competitive pricing for the value provided",
            "Modern and user-friendly approach"
        ],
        "weaknesses": [
            "Limited availability during peak times",
            "Could improve website navigation"
        ]
    }
    
    # Generate demo sentiment drivers
    demo_sentiment_drivers = [
        {"title": "Customer Service Quality", "strength": 8},
        {"title": "Product Range", "strength": 7},
        {"title": "Pricing Competitiveness", "strength": 6},
        {"title": "Brand Recognition", "strength": 6},
        {"title": "Digital Presence", "strength": 5}
    ]
    
    return {
        "score": grade,
        "visibility_percentage": visibility_pct,
        "summary": summary,
        "sentiment": avg_sent,
        "perception": perception,
        "sentiment_analysis": demo_sentiment_analysis,
        "sentiment_drivers": demo_sentiment_drivers,
        "summary_df": summary_df.to_dict('records'),
        "raw_results": results,
        "competitor_plot": plotly_pie.to_json() if plotly_pie else None,
        "competitor_insights": comp_insights,
        "recommendations": recommendations,
        "pdf_path": pdf_path
    }

def run_full_analysis(keyword: str, brand_name: str, progress_callback=None, performance_mode: bool = False, prompt_mode: str = "auto", custom_prompts: list = None):
    import time
    start_time = time.time()
    
    def update_progress(step, message, percentage):
        if progress_callback:
            progress_callback(step, message, percentage)
    
    update_progress(1, "Generating search queries...", 10)
    step_start = time.time()
    
    # Handle prompt generation based on mode
    if prompt_mode == "manual" and custom_prompts:
        prompts = custom_prompts
        print(f"[DEBUG] Using {len(prompts)} custom prompts: {prompts}")
    else:
        prompts = generate_prompts_for_keyword(keyword)
        print(f"[DEBUG] Auto-generated {len(prompts)} prompts: {prompts}")
    
    print(f"[TIMING] Generate prompts took: {time.time() - step_start:.2f}s")
    if not prompts:
        return {
            "error": "No prompts generated; please refine your keyword or provide custom search queries.",
            "score": 0,
            "summary": "",
            "sentiment": 0,
            "perception": "",
            "summary_df": [],
            "raw_results": [],
            "competitor_plot": None,
            "competitor_insights": "",
            "pdf_path": None
        }

    update_progress(2, "Searching across AI models...", 20)
    step_start = time.time()
    
    # Performance mode: Use faster models and fewer searches
    if performance_mode:
        models_to_use = SEARCH_MODELS_FAST
        max_workers = min(MAX_PARALLEL * 2, 9)  # Double parallelism in performance mode
        print(f"[PERFORMANCE MODE] Using faster models: {models_to_use}")
    else:
        models_to_use = ALL_MODELS_FOR_COMPARISON  # Use all models for comparison
        max_workers = MAX_PARALLEL
    
    total = len(prompts) * len(models_to_use)
    results = []
    summary_map = {p:{} for p in prompts}
    
    completed_searches = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(query_model_with_or_without_search, m, p, brand_name):(p,m) for p in prompts for m in models_to_use}
        for fut in concurrent.futures.as_completed(futures):
            p,m = futures[fut]
            r = fut.result()
            results.append(r)
            summary_map[p][m] = r['brand_mentioned']
            completed_searches += 1
            # Update progress based on completed searches (20% to 50%)
            search_progress = 20 + (completed_searches / total) * 30
            update_progress(2, f"Searching across AI models... ({completed_searches}/{total})", search_progress)

    print(f"[TIMING] AI model searches took: {time.time() - step_start:.2f}s")
    
    rows,mentions = [],0
    for p,vals in summary_map.items():
        cnt = sum(vals.values()); mentions += cnt
        rows.append({'Prompt':p, **{m:('✅' if v else '❌') for m,v in vals.items()}, 'Score':f"{cnt}/{len(MODELS)}"})
    summary_df = pd.DataFrame(rows)

    update_progress(3, "Analysing sentiment...", 55)
    step_start = time.time()
    
    # Get ALL mentions (not just brand mentions) for comprehensive analysis
    all_mentions = [r['response'] for r in results if r['response'].strip()]
    brand_mentions = [r['response'] for r in results if r['brand_mentioned']]
    
    print(f"[DEBUG] Total results: {len(results)}")
    print(f"[DEBUG] Brand mentioned in: {len(brand_mentions)} results")
    print(f"[DEBUG] All mentions length: {len(all_mentions)}")
    
    # Extract strengths and weaknesses from brand mentions (where brand was actually mentioned)
    if brand_mentions:
        try:
            step_start = time.time()
            print(f"[DEBUG] Starting sentiment analysis from {len(brand_mentions)} brand mentions...")
            print(f"[DEBUG] Sample brand mention: {brand_mentions[0][:200]}..." if brand_mentions else "[DEBUG] No brand mentions")
            sentiment_analysis = extract_strengths_and_weaknesses(brand_mentions, brand_name)
            sentiment_drivers = generate_sentiment_drivers(brand_mentions, brand_name)
            print(f"[DEBUG] Extracted {len(sentiment_analysis.get('strengths', []))} strengths and {len(sentiment_analysis.get('weaknesses', []))} weaknesses")
            print(f"[DEBUG] Generated {len(sentiment_drivers)} sentiment drivers")
            print(f"[TIMING] Strengths/weaknesses extraction took: {time.time() - step_start:.2f}s")
        except Exception as e:
            print(f"[Error] extract_strengths_and_weaknesses failed: {e}")
            sentiment_analysis = {"strengths": [], "weaknesses": []}
            sentiment_drivers = []
    elif all_mentions:
        # Fallback to all mentions if no brand mentions found
        try:
            step_start = time.time()
            print(f"[TIMING] No brand mentions found, using all mentions for sentiment analysis...")
            sentiment_analysis = extract_strengths_and_weaknesses(all_mentions, brand_name)
            sentiment_drivers = generate_sentiment_drivers(all_mentions, brand_name)
            print(f"[TIMING] Strengths/weaknesses extraction took: {time.time() - step_start:.2f}s")
        except Exception as e:
            print(f"[Error] extract_strengths_and_weaknesses failed: {e}")
            sentiment_analysis = {"strengths": [], "weaknesses": []}
            sentiment_drivers = []
    else:
        print("[DEBUG] Skipping sentiment analysis - no mentions found")
        sentiment_analysis = {"strengths": [], "weaknesses": []}
        sentiment_drivers = []
    
    # Calculate simple sentiment score for compatibility
    avg_sent = 0.7 if sentiment_analysis.get("strengths") else 0.3

    # Generate sentiment summary from strengths and weaknesses
    try:
        strengths = sentiment_analysis.get("strengths", [])
        weaknesses = sentiment_analysis.get("weaknesses", [])
        
        print(f"[DEBUG] Generating perception from strengths: {strengths[:3]}")  # Show first 3
        print(f"[DEBUG] Generating perception from weaknesses: {weaknesses[:3]}")  # Show first 3
        
        if strengths or weaknesses:
            # Generate a natural paragraph summary using LLM
            system_prompt = (
                "You are an expert at creating professional sentiment analysis summaries. "
                "Create a 1-2 paragraph summary of the brand's perception based on the strengths and weaknesses provided. "
                "Write in a professional, analytical tone without using bullet points or structured lists. "
                "Make it flow naturally as a narrative summary of the brand's reputation."
            )
            
            strengths_text = ", ".join(strengths) if strengths else "No specific strengths mentioned"
            weaknesses_text = ", ".join(weaknesses) if weaknesses else "No specific weaknesses mentioned"
            
            user_prompt = f"Create a sentiment summary for '{brand_name}' based on these findings. Strengths: {strengths_text}. Weaknesses: {weaknesses_text}. Write as flowing paragraphs, not bullet points."
            
            try:
                perception = call_llm(
                    SENTIMENT_MODEL,
                    system_prompt,
                    user_prompt,
                    300,
                    0.5
                )
            except Exception as llm_error:
                print(f"[Error] LLM perception generation failed: {llm_error}")
                # Fallback to simple summary
                if strengths and weaknesses:
                    perception = f"Analysis of '{brand_name}' reveals a mixed sentiment profile. The brand demonstrates several positive attributes including {', '.join(strengths[:3])}. However, there are areas that could benefit from attention, particularly {', '.join(weaknesses[:2])}. Overall, the brand maintains a presence in search results with both positive recognition and opportunities for improvement."
                elif strengths:
                    perception = f"'{brand_name}' shows generally positive sentiment in search results. The brand is recognised for {', '.join(strengths[:3])}. This positive perception suggests strong market positioning and customer satisfaction in key areas."
                else:
                    perception = f"'{brand_name}' appears in search results with limited specific sentiment indicators. While this suggests a neutral to positive baseline perception, there may be opportunities to strengthen the brand's reputation through more prominent positive associations."
        else:
            perception = f"The analysis of '{brand_name}' in search results shows a neutral sentiment profile with limited specific feedback indicators. This baseline positioning provides opportunities to build stronger positive associations and enhance brand recognition in search contexts."
            
    except Exception as e:
        print(f"[Error] sentiment summary generation: {e}")
        perception = "Error generating sentiment summary."

    print(f"[TIMING] Sentiment analysis took: {time.time() - step_start:.2f}s")

    update_progress(4, "Extracting competitors...", 70)
    step_start = time.time()
    print(f"[TIMING] Starting competitor extraction...")
    # Replace the old competitor analysis with the robust version
    competitor_analyzer = create_robust_competitor_analyzer(openai)
    competitor_analysis = competitor_analyzer.analyze_competitors(results, brand_name, skip_ai_validation=performance_mode)
    
    # Use the robust analysis results
    competitors = competitor_analysis['competitors']
    competitor_counts = competitor_analysis['mention_counts']
    market_shares = competitor_analysis['market_shares']
    
    # Include user's brand in the analysis
    brand_mentions = sum(1 for r in results if r['brand_mentioned'])
    all_counts = {brand_name: brand_mentions, **competitor_counts}
    total_mentions = sum(all_counts.values())
    
    # Calculate shares including user's brand
    all_shares = {}
    if total_mentions > 0:
        all_shares = {c: (cnt / total_mentions) * 100 for c, cnt in all_counts.items()}
    
    safe_brand = re.sub(r'[^a-zA-Z0-9]', '_', brand_name)
    safe_keyword = re.sub(r'[^a-zA-Z0-9]', '_', keyword)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plotly_pie = None
    pie_img_path = os.path.join(tempfile.gettempdir(), f"competitor_pie_{safe_brand}_{safe_keyword}_{timestamp}.png")

    if all_shares:
        # Use Seaborn for more reliable and visually appealing pie chart
        try:
            print("[DEBUG] Creating seaborn pie chart...")
            seaborn_pie = create_seaborn_pie(list(all_shares.keys()), list(all_shares.values()), brand_name, pie_img_path)
            print("[DEBUG] Seaborn pie chart created successfully")
        except Exception as e:
            print(f"[ERROR] Seaborn pie chart creation failed: {e}")
            # Create a simple fallback pie chart path
            pie_img_path = None
        
        # Keep Plotly for web display (fallback)
        try:
            print("[DEBUG] Creating plotly pie chart...")
            plotly_pie = create_plotly_enhanced_pie(list(all_shares.keys()), list(all_shares.values()), brand_name)
            print("[DEBUG] Plotly pie chart created successfully")
        except Exception as e:
            print(f"[Warning] Plotly chart creation failed: {e}")
            plotly_pie = None

    print(f"[TIMING] Competitor extraction took: {time.time() - step_start:.2f}s")

    update_progress(5, "Generating insights...", 85)
    step_start = time.time()
    print("[DEBUG] Generating competitor insights using same data as pie chart...")
    print(f"[DEBUG] Competitor counts: {dict(list(all_counts.items())[:5])}")  # Show first 5
    print(f"[DEBUG] Competitor shares: {dict(list(all_shares.items())[:5])}")  # Show first 5
    comp_insights = generate_competitor_insights_from_data(all_counts, all_shares, brand_name)
    print("[DEBUG] Competitor insights generation finished.")
    print(f"[TIMING] Competitor insights took: {time.time() - step_start:.2f}s")

    update_progress(6, "Generating recommendations...", 90)
    step_start = time.time()
    recommendations = generate_recommendations(brand_name, keyword)
    print(f"[TIMING] Recommendations took: {time.time() - step_start:.2f}s")

    update_progress(7, "Finalising analysis...", 95)
    step_start = time.time()
    visibility_pct = mentions / total * 100 if total else 0
    
    # Debug the different counting methods
    print(f"[DEBUG VISIBILITY CALCULATION]")
    print(f"  Main visibility: {mentions} brand mentions out of {total} total searches = {visibility_pct:.1f}%")
    print(f"  Competitor pie chart: {brand_mentions} brand mentions out of {total_mentions} total mentions = {all_shares.get(brand_name, 0):.1f}%")
    print(f"  Difference: mentions variable ({mentions}) vs brand_mentions count ({brand_mentions})")
    print(f"  All counts for pie chart: {all_counts}")
    print(f"  All shares for pie chart: {dict(list(all_shares.items())[:5]) if all_shares else 'None'}")  # Show first 5 for readability
    
    # Calculate letter grade
    if visibility_pct >= 90:
        grade = "A+"
        summary = f"🎉 Excellent! Your brand achieved {visibility_pct:.1f}% visibility across AI search results. Outstanding performance!"
    elif visibility_pct >= 80:
        grade = "A"
        summary = f"🎉 Excellent! Your brand achieved {visibility_pct:.1f}% visibility across AI search results. Strong AI presence!"
    elif visibility_pct >= 70:
        grade = "B+"
        summary = f"👍 Very Good! Your brand achieved {visibility_pct:.1f}% visibility across AI search results. Above average performance."
    elif visibility_pct >= 60:
        grade = "B"
        summary = f"👍 Good! Your brand achieved {visibility_pct:.1f}% visibility across AI search results. Solid presence with room to grow."
    elif visibility_pct >= 50:
        grade = "C+"
        summary = f"⚠️ Fair. Your brand achieved {visibility_pct:.1f}% visibility across AI search results. Moderate presence - improvement needed."
    elif visibility_pct >= 40:
        grade = "C"
        summary = f"⚠️ Fair. Your brand achieved {visibility_pct:.1f}% visibility across AI search results. Below average - focus on AI optimization."
    elif visibility_pct >= 30:
        grade = "D+"
        summary = f"📉 Poor. Your brand achieved only {visibility_pct:.1f}% visibility across AI search results. Significant improvement needed."
    elif visibility_pct >= 20:
        grade = "D"
        summary = f"📉 Poor. Your brand achieved only {visibility_pct:.1f}% visibility across AI search results. Urgent attention required."
    else:
        grade = "F"
        summary = f"❌ Very Poor. Your brand achieved only {visibility_pct:.1f}% visibility across AI search results. Immediate action required to improve AI presence."

    # Generate the PDF report
    try:
        print("[DEBUG] Starting PDF generation...")
        pdf_path, logo_warning = generate_pdf_report(brand_name, keyword, grade, summary, avg_sent, perception, summary_df, comp_insights, recommendations, pie_img_path)
        print("[DEBUG] PDF generation completed successfully")
        print(f"[TIMING] PDF generation took: {time.time() - step_start:.2f}s")
    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        pdf_path = None
        logo_warning = f"PDF generation failed: {e}"

    # Optionally, delete the pie chart image after PDF is generated
    try:
        if os.path.exists(pie_img_path):
            os.remove(pie_img_path)
    except Exception as e:
        print(f"[Warning] Could not delete pie chart image: {e}")

    print(f"[TIMING] TOTAL ANALYSIS TIME: {time.time() - start_time:.2f}s")

    return {
        "score": grade,
        "visibility_percentage": visibility_pct,
        "summary": summary,
        "sentiment": avg_sent,
        "perception": perception,
        "sentiment_analysis": sentiment_analysis,
        "sentiment_drivers": sentiment_drivers,
        "summary_df": summary_df.to_dict('records'),
        "raw_results": results,
        "competitor_plot": plotly_pie.to_json() if plotly_pie else None,
        "competitor_insights": comp_insights,
        "recommendations": recommendations,
        "pdf_path": pdf_path
    }

def generate_sentiment_drivers(all_mentions: list[str], brand_name: str) -> list[dict]:
    """
    Generate sentiment drivers with strength ratings for visualization.
    Returns a list of sentiment drivers with titles and strength scores.
    """
    if not all_mentions:
        return []
    
    try:
        combined_text = " ".join(all_mentions)
        
        system_prompt = (
            "You are an expert at analysing brand sentiment and identifying key sentiment drivers. "
            "Analyse the text and identify 3-5 key sentiment drivers (topics that affect brand perception). "
            "For each driver, provide a title and rate its strength from 1-10 based on how positively it's mentioned. "
            "Return a JSON array of objects with 'title' and 'strength' fields. "
            "Example: [{'title': 'Customer Service Quality', 'strength': 8}, {'title': 'Product Range', 'strength': 6}]"
        )
        
        user_prompt = f"Analyse these mentions about {brand_name} and identify key sentiment drivers with strength ratings: {combined_text[:1500]}"
        
        result_text = call_llm(
            SENTIMENT_MODEL,
            system_prompt,
            user_prompt,
            300,
            0.3
        )
        
        # Try to extract JSON from the response
        import json
        import re
        
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if json_match:
            drivers = json.loads(json_match.group())
            return drivers[:5]  # Limit to 5 drivers
        else:
            # Fallback - create some default drivers
            return [
                {"title": "Overall Brand Presence", "strength": 6},
                {"title": "Service Quality", "strength": 5},
                {"title": "Market Position", "strength": 4}
            ]
            
    except Exception as e:
        print(f"[Error] generate_sentiment_drivers: {e}")
        return []

def extract_positive_attributes_directly(text: str, brand_name: str) -> list[str]:
    """
    Direct extraction of positive attributes from text when LLM parsing fails.
    """
    import re
    
    strengths = []
    
    # Find sentences that mention the brand
    brand_pattern = re.compile(rf'\b{re.escape(brand_name)}\b', re.IGNORECASE)
    sentences = re.split(r'[.!?]', text)
    
    for sentence in sentences:
        if brand_pattern.search(sentence):
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Look for positive indicators
            positive_indicators = [
                'known for', 'renowned for', 'specializes in', 'offers', 'provides',
                'comprehensive', 'high-quality', 'professional', 'excellent', 'leading',
                'top-rated', 'award-winning', 'experienced', 'expertise', 'quality',
                'trusted', 'reliable', 'innovative', 'creative', 'dedicated',
                'client-centric', 'tailored', 'scalable', 'notable clients',
                'comprehensive services', 'full-service', 'end-to-end'
            ]
            
            if any(indicator in sentence.lower() for indicator in positive_indicators):
                # Clean up the sentence
                clean_sentence = re.sub(r'\*\*\[.*?\]\(.*?\)\*\*', brand_name, sentence)
                clean_sentence = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_sentence)
                clean_sentence = clean_sentence.strip()
                
                if len(clean_sentence) > 20 and len(strengths) < 5:
                    strengths.append(clean_sentence)
    
    return strengths

def extract_strengths_and_weaknesses(all_mentions: list[str], brand_name: str) -> dict:
    """
    Extract specific strengths and weaknesses mentioned about the brand.
    Returns a dictionary with strengths and weaknesses arrays.
    """
    if not all_mentions:
        return {"strengths": [], "weaknesses": []}
    
    try:
        # Combine all mentions for analysis
        combined_text = " ".join(all_mentions)
        
        system_prompt = (
            "You are an expert at analysing brand mentions to extract specific strengths and weaknesses. "
            "Analyse the text and identify specific strengths and weaknesses mentioned about the brand. "
            "Return a JSON object with 'strengths' and 'weaknesses' arrays containing specific quotes or paraphrased points. "
            "Focus on actionable insights like 'excellent customer service', 'high-quality products', 'expensive pricing', 'slow delivery', etc. "
            "Limit to top 5 strengths and top 5 weaknesses. If no weaknesses are mentioned, return empty array."
        )
        
        user_prompt = f"Analyse these mentions about {brand_name} and extract specific strengths and weaknesses: {combined_text[:1500]}"
        
        print(f"[DEBUG] Extracting strengths/weaknesses for {brand_name}")
        print(f"[DEBUG] Combined text length: {len(combined_text)}")
        print(f"[DEBUG] Sample text: {combined_text[:300]}...")
        
        result_text = call_llm(
            SENTIMENT_MODEL,
            system_prompt,
            user_prompt,
            400,
            0.3
        )
        
        print(f"[DEBUG] LLM response: {result_text}")
        
        # Try to extract JSON from the response
        import json
        import re
        
        # Look for JSON in the response
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                print(f"[DEBUG] Parsed JSON successfully: {data}")
                return {
                    "strengths": data.get("strengths", [])[:5],
                    "weaknesses": data.get("weaknesses", [])[:5]
                }
            except json.JSONDecodeError as e:
                print(f"[DEBUG] JSON parsing failed: {e}")
                # Continue to fallback parsing
        else:
            # Fallback parsing - look for strengths and weaknesses sections
            strengths = []
            weaknesses = []
            
            lines = result_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if 'strength' in line.lower() or 'positive' in line.lower():
                    current_section = 'strengths'
                elif 'weakness' in line.lower() or 'negative' in line.lower() or 'downside' in line.lower():
                    current_section = 'weaknesses'
                elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    point = line[1:].strip()
                    if current_section == 'strengths' and len(strengths) < 5:
                        strengths.append(point)
                    elif current_section == 'weaknesses' and len(weaknesses) < 5:
                        weaknesses.append(point)
            
            # If no structured parsing worked, try to extract from original text
            if not strengths and not weaknesses:
                print("[DEBUG] No strengths/weaknesses found via parsing, trying direct extraction...")
                strengths = extract_positive_attributes_directly(combined_text, brand_name)
                print(f"[DEBUG] Direct extraction found {len(strengths)} strengths: {strengths}")
            
            return {"strengths": strengths, "weaknesses": weaknesses}
        
    except Exception as e:
        print(f"[Error] Strengths/weaknesses extraction: {e}")
        # Try direct extraction as final fallback
        try:
            strengths = extract_positive_attributes_directly(" ".join(all_mentions), brand_name)
            return {"strengths": strengths, "weaknesses": []}
        except:
            return {"strengths": [], "weaknesses": []}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def run_analysis_background(analysis_id: str, keyword: str, brand_name: str, performance_mode: bool, demo_mode: bool, prompt_mode: str = "auto", custom_prompts: list = None):
    """Background task to run the analysis"""
    def progress_callback(step, message, percentage):
        progress_store[analysis_id] = {"step": step, "message": message, "percentage": percentage}
        print(f"[PROGRESS] Step {step}: {message} ({percentage}%)")  # Debug print
    
    try:
        if demo_mode:
            # Use demo data for quick testing
            def update_progress(step, message, percentage):
                if progress_callback:
                    progress_callback(step, message, percentage)
            
            update_progress(1, "Generating demo data...", 20)
            update_progress(2, "Creating sample results...", 50)
            update_progress(3, "Building charts...", 80)
            update_progress(4, "Finalizing demo...", 95)
            result = generate_demo_data(keyword, brand_name)
        else:
            # Use real API calls
            result = run_full_analysis(keyword, brand_name, progress_callback, performance_mode, prompt_mode, custom_prompts)
        
        # Store the final result
        analysis_results[analysis_id] = result
        progress_store[analysis_id] = {"step": 7, "message": "Analysis complete!", "percentage": 100}
        
    except Exception as e:
        analysis_results[analysis_id] = {"error": str(e)}
        progress_store[analysis_id] = {"step": 0, "message": f"Error: {str(e)}", "percentage": 0}

@app.post("/analyze")
async def analyze_brand(background_tasks: BackgroundTasks, keyword: str = Form(...), brand_name: str = Form(...), performance_mode: bool = Form(False), demo_mode: bool = Form(False), prompt_mode: str = Form("auto"), custom_prompts: str = Form(None)):
    # Validate inputs first (skip for demo mode to allow testing)
    if not demo_mode:
        validation_result = validate_business_inputs(keyword, brand_name)
        if not validation_result["valid"]:
            return {"error": validation_result["error"]}
    
    # Validate custom prompts if provided
    custom_prompts_list = []
    if prompt_mode == "manual" and custom_prompts:
        try:
            custom_prompts_list = json.loads(custom_prompts)
            
            # Validate custom prompts for business relevance (skip for demo mode)
            if not demo_mode:
                for i, prompt in enumerate(custom_prompts_list):
                    if not prompt.strip():
                        continue
                    prompt_validation = validate_custom_prompt(prompt)
                    if not prompt_validation["valid"]:
                        return {"error": f"Custom query {i+1}: {prompt_validation['error']}"}
                        
        except json.JSONDecodeError:
            return {"error": "Invalid custom prompts format"}
    
    # Generate unique analysis ID
    import uuid
    analysis_id = str(uuid.uuid4())
    
    # Initialize progress
    progress_store[analysis_id] = {"step": 0, "message": "Starting analysis...", "percentage": 0}
    print(f"[DEBUG] Analysis started for {brand_name} - {keyword} (ID: {analysis_id})")  # Debug print
    print(f"[DEBUG] Prompt mode: {prompt_mode}, Custom prompts: {len(custom_prompts_list) if custom_prompts_list else 0}")
    
    # Start background analysis
    background_tasks.add_task(run_analysis_background, analysis_id, keyword, brand_name, performance_mode, demo_mode, prompt_mode, custom_prompts_list)
    
    return {"analysis_id": analysis_id, "status": "started"}

@app.get("/progress/{analysis_id}")
async def get_progress(analysis_id: str):
    """Get current progress for a specific analysis"""
    if analysis_id in progress_store:
        current_progress = progress_store[analysis_id]
        print(f"[PROGRESS ENDPOINT] Returning for {analysis_id}: {current_progress}")  # Debug print
        return current_progress
    else:
        print(f"[PROGRESS ENDPOINT] No analysis found for ID: {analysis_id}")  # Debug print
        return {"step": 0, "message": "Analysis not found", "percentage": 0}

@app.get("/results/{analysis_id}")
async def get_results(analysis_id: str):
    """Get results for a specific analysis"""
    if analysis_id in analysis_results:
        result = analysis_results[analysis_id]
        # Clean up after retrieving results
        if analysis_id in progress_store:
            del progress_store[analysis_id]
        if analysis_id in analysis_results:
            del analysis_results[analysis_id]
        return result
    else:
        return {"error": "Results not ready or analysis not found"}

@app.get("/download/{filename}")
async def download_pdf(filename: str):
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    print("Starting AI Brand Visibility Checker on http://127.0.0.1:8003")
    print("Open your browser and go to: http://127.0.0.1:8003")
    uvicorn.run(app, host="127.0.0.1", port=8003)