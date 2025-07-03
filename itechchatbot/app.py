import json
import os
import re
import logging
from collections import Counter 
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import random

import torch
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util

# --- Flask App Setup ---
# CRITICAL FIX: Adjusted template_folder path in Flask constructor
# Ensure no leading whitespace before 'app = Flask(...)'
app = Flask(__name__,
            static_folder=str(Path(__file__).parent / 'modules' / 'static'),
            template_folder=str(Path(__file__).parent / 'modules' / 'templates')) 
CORS(app) # Enable CORS for all routes, or specify origins if needed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app_logger = logging.getLogger(__name__)

# --- Global variables for model and embeddings ---
# These will be loaded once when the application starts
model = None
corpus_embeddings = None
corpus_texts = []
corpus_urls = []
# cached_scraped_data = {} # This variable is not used, can be removed
service_page_map = {}
MAJOR_SECTION_KEYWORDS = []

# --- Global variables for Intent Data ---
intents_data = {}
# training_data = [] # This variable is not used directly, can be removed
training_phrases_map = {} # {'normalized_text': 'intent_name'}
homepage_url = "" # To store the normalized homepage URL from scraped data

# Define file paths relative to the current script
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / 'modules' / 'static' 
DATA_DIR = STATIC_DIR / 'data' # Data is still inside static

EMBEDDINGS_FILE = DATA_DIR / 'scraped_content_embeddings.json'
CONTENT_FILE = DATA_DIR / 'scraped_content.json'
INTENTS_FILE = BASE_DIR / 'intents.json' # Assuming intents.json is in the same directory as app.py
TRAINING_FILE = BASE_DIR / 'training.json' # Assuming training.json is in the same directory as app.py
CHATBOT_CONFIG_FILE = BASE_DIR / 'config.json' # Assuming config.json is in the base directory

# Ensure the data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Global variable for scraped data ---
scraped_data = {}

# --- Helper Functions ---

def load_json_data(filepath):
    """Loads JSON data from a specified file path."""
    if not filepath.exists():
        app_logger.error(f"File not found: {filepath}. Please ensure your file is in the correct directory and named correctly.")
        # Return appropriate empty structure based on file type for robustness
        if 'intents.json' in str(filepath) or 'scraped_content.json' in str(filepath) or 'config.json' in str(filepath):
            return {} 
        else: # For training.json which is a list
            return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        app_logger.error(f"Error decoding JSON from {filepath}: {e}. Check if the JSON is valid.")
        if 'intents.json' in str(filepath) or 'scraped_content.json' in str(filepath) or 'config.json' in str(filepath):
            return {} 
        else:
            return []
    except Exception as e:
        app_logger.error(f"An unexpected error occurred while loading {filepath}: {e}")
        if 'intents.json' in str(filepath) or 'scraped_content.json' in str(filepath) or 'config.json' in str(filepath):
            return {} 
        else:
            return []

def load_scraped_content():
    """Loads the scraped content from the JSON file and identifies the homepage URL."""
    global scraped_data, homepage_url
    try:
        scraped_data = load_json_data(CONTENT_FILE)
        
        if not scraped_data:
            app_logger.error("Scraped content data is empty or could not be loaded.")
            return False
        
        app_logger.info(f"Successfully loaded scraped content from {CONTENT_FILE}. Total pages: {len(scraped_data)}")

        # Load root_url from config.json (if available, otherwise fallback)
        root_url_from_config = 'https://ktgsoftware.com/' # Default fallback
        try:
            chatbot_config = load_json_data(CHATBOT_CONFIG_FILE)
            if chatbot_config is not None: # Ensure config was loaded successfully
                root_url_from_config = chatbot_config.get('root_url', root_url_from_config)
        except Exception as e:
            app_logger.warning(f"Could not load chatbot config file from {CHATBOT_CONFIG_FILE}: {e}. Using default root URL.")

        normalized_root = normalize_url(root_url_from_config)
        
        if normalized_root in scraped_data:
            homepage_url = normalized_root
            app_logger.info(f"Identified homepage URL: {homepage_url}")
        else:
            # Fallback: try to find a likely homepage (e.g., shortest path)
            if scraped_data:
                candidate_urls = [url for url in scraped_data.keys() if urlparse(url).path in ['', '/', '/index.html']]
                if candidate_urls:
                    if normalized_root in candidate_urls:
                        homepage_url = normalized_root
                    else:
                        homepage_url = sorted(candidate_urls, key=lambda url: len(url))[0]
                elif scraped_data:
                     homepage_url = list(scraped_data.keys())[0]

                app_logger.warning(f"Could not find exact root URL '{normalized_root}' in scraped data. Falling back to identified homepage: {homepage_url}")
            else:
                app_logger.error("Scraped data is empty, cannot identify homepage.")

        return True
    except Exception as e:
        app_logger.error(f"An unexpected error occurred while loading scraped content: {e}")
        return False


def load_intent_data():
    """Loads intents and training phrases into global variables."""
    global intents_data, training_phrases_map

    app_logger.info("Loading intent data from intents.json and training.json...")
    
    intents_data = load_json_data(INTENTS_FILE)
    raw_training_data = load_json_data(TRAINING_FILE)

    if not intents_data:
        app_logger.warning(f"Could not load intents from {INTENTS_FILE}. Intent-based responses will not work.")
    if not raw_training_data:
        app_logger.warning(f"Could not load training data from {TRAINING_FILE}. Intent matching will be limited.")

    # Process raw_training_data into training_phrases_map for quick lookup
    if raw_training_data:
        for entry in raw_training_data:
            text = entry.get('text')
            intent = entry.get('intent')
            if text and intent:
                normalized_text = normalize_text(text)
                training_phrases_map[normalized_text] = intent
        app_logger.info(f"Loaded {len(training_phrases_map)} training phrases for intent matching.")

def normalize_text(text):
    """Normalizes text for intent matching (lowercase, remove extra whitespace)."""
    if text is None:
        return ""
    return re.sub(r'\s+', ' ', text).strip().lower()

def normalize_url(url):
    """Normalizes a URL for consistent lookup."""
    parsed = urlparse(url)
    # Remove trailing slash unless it's just the root domain
    path = parsed.path.rstrip('/') if parsed.path != '/' else parsed.path
    # Remove common homepage file names
    if path.endswith('/index.html'):
        path = path[:-11].rstrip('/') # Remove /index.html and any trailing slash that results
    elif path.endswith('/index.htm'):
        path = path[:-10].rstrip('/')
    
    clean_url = parsed.scheme + "://" + parsed.netloc.replace('www.', '') + path
    
    # Ensure root domain without path has a trailing slash for consistency if that's the scraper's norm
    if not path and not clean_url.endswith('/'):
        clean_url += '/'
        
    return clean_url

def get_current_time():
    """Returns the current time and approximate location."""
    now = datetime.now()
    current_time_str = now.strftime("%I:%M %p") # e.g., 01:06 PM
    # Assuming the server is located in Coimbatore based on previous context
    return f"The current time is {current_time_str} in Coimbatore, India."

def get_clients_info():
    """Returns a professional sentence about clients and the client page link."""
    client_page_url = "https://ktgsoftware.com/client.html"
    return f"We collaborate with a diverse range of clients, delivering tailored software solutions that drive their success. You can explore our client portfolio and success stories on our dedicated client page: {client_page_url}"


def match_intent(query):
    """
    Attempts to match the user query to a predefined intent using exact or close text matching.
    Returns the intent name if a match is found, otherwise None.
    """
    normalized_query = normalize_text(query)
    app_logger.debug(f"Normalized query for intent matching: '{normalized_query}'")

    # Exact match lookup
    if normalized_query in training_phrases_map:
        matched_intent = training_phrases_map[normalized_query]
        app_logger.info(f"Exact match found for query '{query}' to intent '{matched_intent}'")
        return matched_intent

    # Check for partial matches or keyword presence for robustness (simple example)
    for phrase, intent in training_phrases_map.items():
        if phrase in normalized_query: # User query contains the training phrase
            app_logger.info(f"Partial match found: Query '{query}' contains training phrase '{phrase}' for intent '{intent}'")
            return intent
        elif normalized_query in phrase: # Training phrase contains the user query
            app_logger.info(f"Partial match found: Training phrase '{phrase}' contains query '{query}' for intent '{intent}'")
            return intent

    return None

def get_intent_response(intent_name):
    """Retrieves the response for a given intent."""
    intent_info = intents_data.get(intent_name)
    if not intent_info:
        app_logger.warning(f"No info found for intent: {intent_name}")
        # MODIFIED: Professional message
        return intents_data.get('fallback', {}).get('response', "I apologize, but I couldn't find a specific response for that request. Please try rephrasing your query or ask about our services, products, or expertise.")

    if "response" in intent_info:
        return intent_info["response"]
    elif "response_func" in intent_info:
        func_name = intent_info["response_func"]
        # Correctly call the function based on its name (handle the trailing dot from JSON if present)
        if func_name == "get_current_time" or func_name == "get_current_time.": 
            app_logger.info(f"Calling dynamic response function: {func_name}")
            return get_current_time()
        elif func_name == "get_clients_info": # Added client info function call
            app_logger.info(f"Calling dynamic response function: {func_name}")
            return get_clients_info() # CRITICAL FIX: Call the new clients function
        else:
            app_logger.warning(f"Unknown response function: {func_name} for intent {intent_name}")
            # MODIFIED: Professional message
            return intents_data.get('fallback', {}).get('response', "I'm sorry, I encountered an issue fulfilling that request. Please try again.")
    # MODIFIED: Professional message
    return intents_data.get('fallback', {}).get('response', "I'm sorry, I couldn't find a direct answer to your question. Please try rephrasing it.")


def load_or_generate_embeddings():
    """
    Loads pre-computed embeddings or generates them if they don't exist,
    or if the content file has been updated.
    """
    global model, corpus_embeddings, corpus_texts, corpus_urls

    # Check if already loaded in this process (e.g., for subsequent requests)
    if model is not None and corpus_embeddings is not None and len(corpus_texts) > 0:
        app_logger.info("Model and embeddings already loaded in this process. Skipping regeneration.")
        # Ensure dynamic maps are built if not already
        if not service_page_map:
            _build_dynamic_service_map()
        if not MAJOR_SECTION_KEYWORDS:
            _infer_major_section_keywords()
        return

    app_logger.info("Checking for existing embeddings to load or generate...")

    embeddings_exist_and_not_outdated = False
    if EMBEDDINGS_FILE.exists() and os.path.getsize(EMBEDDINGS_FILE) > 0:
        try:
            with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                corpus_embeddings = torch.tensor(data['embeddings'])
                corpus_texts = data['texts']
                corpus_urls = data['urls']
            app_logger.info(f"Loaded {len(corpus_embeddings)} existing embeddings from {EMBEDDINGS_FILE}.")
            
            # Check if content file is newer than embeddings file
            if CONTENT_FILE.exists() and CONTENT_FILE.stat().st_mtime > EMBEDDINGS_FILE.stat().st_mtime:
                app_logger.warning("Scraped content file is newer than embeddings file. Regenerating embeddings.")
                embeddings_exist_and_not_outdated = False # Force regeneration
            else:
                embeddings_exist_and_not_outdated = True
        except (json.JSONDecodeError, KeyError, TypeError, FileNotFoundError) as e:
            app_logger.warning(f"Error loading embeddings file ({e}). Embeddings will be regenerated.")
            corpus_embeddings = None
            corpus_texts = []
            corpus_urls = []
        except Exception as e: # CRITICAL FIX: Added general exception for robustness
            app_logger.warning(f"An unexpected error occurred while loading embeddings ({e}). Embeddings will be regenerated.")
            corpus_embeddings = None
            corpus_texts = []
            corpus_urls = []

    if embeddings_exist_and_not_outdated:
        if model is None:
            try:
                app_logger.info("Model was None, loading SentenceTransformer model 'all-MiniLM-L6-v2' after loading embeddings...")
                # Load model from the bundled directory
                model = SentenceTransformer('all-MiniLM-L6-v2')
                app_logger.info("SentenceTransformer model loaded successfully (after embeddings).")
            except Exception as e:
                app_logger.error(f"Failed to load SentenceTransformer model after embeddings: {e}")
                model = None
                corpus_embeddings = None
                corpus_texts = []
                corpus_urls = []
        if model is not None and corpus_embeddings is not None:
            _build_dynamic_service_map() # Ensure maps are built
            _infer_major_section_keywords() # Ensure keywords are inferred
            return # Successfully loaded existing embeddings and model.

    app_logger.info("Embeddings not found/loaded or content updated. Generating new embeddings.")

    # Ensure scraped_data is loaded before generating embeddings
    if not scraped_data: 
        if not load_scraped_content(): 
            app_logger.error("Cannot generate embeddings: Scraped data is empty or missing after attempting to load.")
            corpus_embeddings = None
            corpus_texts = []
            corpus_urls = []
            return

    _build_dynamic_service_map() # Build dynamic maps before embedding generation
    _infer_major_section_keywords() # Infer keywords before embedding generation

    texts_for_embedding = []
    urls_for_embedding = []
    seen_snippets = set() 

    for url, page_content in scraped_data.items():
        paragraphs = page_content.get('paragraphs', [])
        for p_text in paragraphs:
            cleaned_p_text = p_text.strip()
            if cleaned_p_text and cleaned_p_text not in seen_snippets:
                texts_for_embedding.append(cleaned_p_text)
                urls_for_embedding.append(url)
                seen_snippets.add(cleaned_p_text)

        headings = page_content.get('headings', [])
        for h_text in headings:
            cleaned_h_text = h_text.strip()
            if cleaned_h_text and cleaned_h_text not in seen_snippets:
                texts_for_embedding.append(cleaned_h_text)
                urls_for_embedding.append(url)
                seen_snippets.add(cleaned_h_text)
        
        for category_key in ['services', 'products', 'use_cases', 'benefits', 'features']:
            items = page_content.get(category_key, [])
            if isinstance(items, list): 
                for item_text in items:
                    cleaned_item_text = str(item_text).strip()
                    if cleaned_item_text: 
                        page_title = page_content.get('page_title', url.split('/')[-1].replace('.html', '').replace('-', ' ').title())
                        text_to_embed = f"{category_key.replace('_', ' ').title()} of {page_title}: {cleaned_item_text}"
                        
                        if text_to_embed not in seen_snippets:
                            texts_for_embedding.append(text_to_embed)
                            urls_for_embedding.append(url)
                            seen_snippets.add(text_to_embed) 

        extracted_full_text = page_content.get('extracted_text', '')
        if extracted_full_text and extracted_full_text.strip() and extracted_full_text.strip() not in seen_snippets:
            texts_for_embedding.append(extracted_full_text.strip())
            urls_for_embedding.append(url)
            seen_snippets.add(extracted_full_text.strip())

    if not texts_for_embedding:
        app_logger.error("Cannot generate embeddings: Extracted texts for embedding are empty after processing JSON structure. Check your scraped_content.json.")
        corpus_embeddings = None
        corpus_texts = []
        corpus_urls = []
        return

    try:
        if model is None:
            app_logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2' for generation...")
            model = SentenceTransformer( 'all-MiniLM-L6-v2')
            app_logger.info("SentenceTransformer model loaded successfully.")
        else:
            app_logger.info("SentenceTransformer model already loaded, reusing for generation.")
    except Exception as e:
        app_logger.error(f"Failed to load SentenceTransformer model for generation: {e}")
        model = None 
        return

    app_logger.info(f"Generating embeddings for {len(texts_for_embedding)} texts. This may take a while...")
    try:
        corpus_embeddings = model.encode(texts_for_embedding, convert_to_tensor=True, show_progress_bar=True)
        corpus_texts = texts_for_embedding
        corpus_urls = urls_for_embedding

        embeddings_data = {
            'embeddings': corpus_embeddings.tolist(),
            'texts': corpus_texts,
            'urls': corpus_urls
        }
        with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=4)
        app_logger.info(f"Embeddings generated and saved to {EMBEDDINGS_FILE}.")
    except Exception as e:
        app_logger.error(f"Error during embedding generation: {e}")
        corpus_embeddings = None
        corpus_texts = []
        corpus_urls = []

def _normalize_text_for_alias(text):
    """Normalizes text by stripping whitespace and extra spaces."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s/&-]', '', text)
    return text

def _infer_major_section_keywords():
    """Dynamically infers major section keywords from the scraped content."""
    global MAJOR_SECTION_KEYWORDS
    app_logger.info("Dynamically inferring major section keywords...")
    scraped_data_local = load_json_data(CONTENT_FILE)

    all_normalized_headings = []
    for url, page_content in scraped_data_local.items():
        for heading in page_content.get('headings', []):
            if heading and heading.strip():
                all_normalized_headings.append(_normalize_text_for_alias(heading))

    heading_counts = Counter(all_normalized_headings)

    inferred_keywords = set()
    frequency_threshold = max(1, len(scraped_data_local) // 3)

    baseline_keywords = [
        "contact us", "about us", "quick links", "our trusted clients",
        "request a call back", "services", "products", "home", "solutions",
        "faqs", "get in touch", "career", "careers","email"
    ]
    for kw in baseline_keywords:
        inferred_keywords.add(_normalize_text_for_alias(kw))

    for heading, count in heading_counts.items():
        if (count >= frequency_threshold and len(heading.split()) > 1) or \
           (len(heading.split()) > 2 and count > 0):
            inferred_keywords.add(heading)

    product_service_names = set(service_page_map.keys())

    final_inferred_keywords = []
    for kw in inferred_keywords:
        if kw not in product_service_names or \
           kw in ["services", "products", "solutions", "applications", "technologies"]:
            final_inferred_keywords.append(kw)

    MAJOR_SECTION_KEYWORDS = list(set(final_inferred_keywords))
    app_logger.info(f"Inferred {len(MAJOR_SECTION_KEYWORDS)} major section keywords.")

def _build_dynamic_service_map():
    """Dynamically builds the service_page_map from scraped content."""
    global service_page_map
    service_page_map = {}

    app_logger.info("Building dynamic service page map...")
    scraped_data_local = load_json_data(CONTENT_FILE)

    for url, page_content in scraped_data_local.items():
        potential_aliases = set()

        page_title = page_content.get('page_title', '')
        if page_title:
            potential_aliases.add(_normalize_text_for_alias(page_title))

        for h_text in page_content.get('headings', []):
            if h_text and h_text.strip():
                normalized_h = _normalize_text_for_alias(h_text)
                potential_aliases.add(normalized_h)
                
                parts = re.split(r'[/\-&]', normalized_h)
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 1:
                        potential_aliases.add(part)

        parsed_url = urlparse(url)
        url_path_segment = Path(parsed_url.path).stem.lower()
        if url_path_segment and url_path_segment != 'index':
            potential_aliases.add(url_path_segment)
            
            hyphen_to_space = url_path_segment.replace('-', ' ')
            if hyphen_to_space != url_path_segment:
                potential_aliases.add(hyphen_to_space)
            
        for list_key in ['services', 'products', 'use_cases']:
            for item in page_content.get(list_key, []):
                if item and item.strip():
                    potential_aliases.add(_normalize_text_for_alias(item))


        for alias in potential_aliases:
            if alias and alias not in service_page_map:
                service_page_map[alias] = url
    
    app_logger.info(f"Dynamic service map built with {len(service_page_map)} entries.")

def _get_section_and_sub_content(structured_content_list, start_index, parent_heading_level):
    """
    Linearly traverses structured_content_list from start_index to collect content
    (sub-headings, paragraphs, list items) belonging to the section defined by start_index,
    stopping when a new major section or a heading of equal/higher level is encountered.
    """
    collected_content = []
    
    current_idx = start_index + 1
    
    while current_idx < len(structured_content_list):
        element = structured_content_list[current_idx]
        element_heading_text = _normalize_text_for_alias(element.get("heading_text", ""))
        element_tag = element.get("tag", "")
        element_level = int(element_tag[1]) if element_tag.startswith('h') else 99
        
        if element_tag.startswith('h') and element_level <= parent_heading_level:
            if element_heading_text in MAJOR_SECTION_KEYWORDS:
                break
            if element_level <= parent_heading_level:
                break

        if element_tag.startswith('h') and element_level > parent_heading_level:
            if element_heading_text and element_heading_text not in collected_content:
                collected_content.append(element_heading_text)
            for p_text in element.get("content_paragraphs", []):
                if p_text.strip() and p_text.strip() not in collected_content:
                    collected_content.append(p_text.strip())
        elif element_tag == 'p' or element_tag == 'li':
            for p_text in element.get("content_paragraphs", []):
                if p_text.strip() and p_text.strip() not in collected_content:
                    collected_content.append(p_text.strip())
        
        current_idx += 1
    
    return list(filter(None, collected_content))

def get_specific_list_response(query):
    """
    Checks if the query is asking for a specific list (services, products, use cases, benefits, features).
    If so, retrieves the list from scraped_content.json (index.html or other relevant page) and formats it.
    """
    global model, corpus_embeddings, corpus_texts, corpus_urls, homepage_url

    query_lower = query.lower().strip()
    
    services_keywords = ["services", "list of services", "our services", "what services do you offer", "tell me about your services", "company services"]
    products_keywords = ["products", "list of products", "our products", "what products do you have", "tell me about your products", "company products"]
    use_cases_keywords = ["use case","use cases", "list of use cases", "our use cases", "examples of use cases", "tell me about use cases", "company use cases"]
    benefits_keywords = ["benefits of", "advantages of", "what are the benefits of", "tell me the benefits of", "benefits", "advantages", "perks", "value proposition"]
    features_keywords = ["features of", "functionalities of", "capabilities of", "what are the features of", "tell me the features of", "features", "functionalities", "capabilities"]

    target_category = None
    response_heading = ""
    target_topic = None 

    if any(keyword in query_lower for keyword in services_keywords):
        target_category = "services"
        response_heading = "Allow me to present the services offered by KTG Software and Consulting:"
    elif any(keyword in query_lower for keyword in products_keywords):
        target_category = "products"
        response_heading = "Discover the products provided by KTG Software and Consulting:"
    elif any(keyword in query_lower for keyword in use_cases_keywords):
        target_category = "use_cases"
        response_heading = "Explore key use cases for KTG Software and Consulting's solutions:"
    elif any(keyword in query_lower for keyword in benefits_keywords):
        target_category = "benefits"
        match = re.search(r'(?:benefits of|advantages of|what are the benefits of|tell me the benefits of)\s+(.+)', query_lower)
        if match:
            target_topic = match.group(1).strip()
            target_topic = re.sub(r'(?:your|the|a|an)$', '', target_topic).strip()
            response_heading = f"Certainly, here are the core benefits of {target_topic}:"
        else:
            response_heading = "Here are some key benefits you might find valuable:"
    elif any(keyword in query_lower for keyword in features_keywords):
        target_category = "features"
        match = re.search(r'(?:features of|functionalities of|capabilities of|what are the features of|tell me the features of)\s+(.+)', query_lower)
        if match:
            target_topic = match.group(1).strip()
            target_topic = re.sub(r'(?:your|the|a|an)$', '', target_topic).strip()
            response_heading = f"To elaborate on {target_topic}, here are its primary features:"
        else:
            response_heading = "Here are some of the distinctive features:"

    if target_category:
        try:
            if target_category in ["benefits", "features"] and target_topic:
                
                if model is None or corpus_embeddings is None or len(corpus_texts) == 0:
                    load_or_generate_embeddings()

                if model and corpus_embeddings is not None and len(corpus_texts) > 0:
                    topic_query_embedding = model.encode(target_topic, convert_to_tensor=True, show_progress_bar=False)
                    cosine_scores = util.cos_sim(topic_query_embedding, corpus_embeddings)[0]
                    
                    candidate_pages_for_topic = {}
                    
                    top_k_for_topic = min(len(corpus_texts), 200)
                    top_topic_results = torch.topk(cosine_scores, k=top_k_for_topic)
                    
                    for score, idx in zip(top_topic_results[0], top_topic_results[1]):
                        if score.item() < 0.50: 
                            continue
                        
                        current_url = corpus_urls[idx]
                        if current_url in scraped_data and scraped_data[current_url].get('structured_content'):
                            page_title_lower = scraped_data[current_url].get('page_title', '').lower()
                            headings_lower = [h.lower() for h in scraped_data[current_url].get('headings', [])]
                            
                            if target_topic in page_title_lower or any(target_topic in h for h in headings_lower):
                                candidate_pages_for_topic[current_url] = max(candidate_pages_for_topic.get(current_url, 0), score.item())

                    best_url_for_topic_specific_list = None
                    if candidate_pages_for_topic:
                        best_url_for_topic_specific_list = max(candidate_pages_for_topic, key=candidate_pages_for_topic.get)

                    if best_url_for_topic_specific_list:
                        page_data = scraped_data.get(best_url_for_topic_specific_list)
                        if page_data and page_data.get('structured_content'):
                            collected_items = []
                            start_collecting = False
                            main_section_tag = "hX" 
                            
                            main_category_keywords = []
                            if target_category == "benefits":
                                main_category_keywords = ["benefits", "advantages"]
                            elif target_category == "features":
                                main_category_keywords = ["key features", "features", "functionalities", "capabilities"]

                            termination_keywords = ["other services", "contact us", "join our newsletter"]
                            
                            for i, section in enumerate(page_data['structured_content']):
                                section_heading_lower = section.get("heading_text", "").lower()
                                section_tag = section.get("tag", "hX")

                                if not start_collecting:
                                    is_main_category_heading = False
                                    if any(kw in section_heading_lower for kw in main_category_keywords):
                                        is_main_category_heading = True
                                    
                                    if is_main_category_heading and \
                                       (not target_topic or target_topic in section_heading_lower or target_topic in page_data.get('page_title', '').lower()):
                                        start_collecting = True
                                        main_section_tag = section_tag
                                        for p_text in section.get("content_paragraphs", []):
                                            if p_text.strip():
                                                collected_items.append(p_text.strip())
                                        continue 
                                    
                                    if start_collecting: # This block was incorrectly indented in previous versions
                                        if section_tag.startswith('h') and main_section_tag.startswith('h') and \
                                           int(section_tag[1]) <= int(main_section_tag[1]):
                                            break 
                                        
                                        if any(kw in section_heading_lower for kw in termination_keywords):
                                            break 

                                        item_text = section.get("heading_text", "").strip()
                                        item_paragraphs = [p.strip() for p in section.get("content_paragraphs", []) if p.strip()]

                                        if item_text and item_paragraphs:
                                            collected_items.append(f"{item_text}: {item_paragraphs[0]}")
                                            collected_items.extend(item_paragraphs[1:])
                                        elif item_text:
                                            collected_items.append(item_text)
                                        elif item_paragraphs:
                                            collected_items.extend(item_paragraphs)
                                        
                                        def deep_collect_nested(nested_sections):
                                            for nested_sec in nested_sections:
                                                nested_heading = nested_sec.get("heading_text", "").strip()
                                                nested_content = [p.strip() for p in nested_sec.get("content_paragraphs", []) if p.strip()]
                                                if nested_heading and nested_content:
                                                    collected_items.append(f"{nested_heading}: {nested_content[0]}")
                                                    collected_items.extend(nested_content[1:])
                                                elif nested_heading:
                                                    collected_items.append(nested_heading)
                                                elif nested_content:
                                                    collected_items.extend(nested_content)
                                                if nested_sec.get("sub_sections"):
                                                    deep_collect_nested(nested_sec["sub_sections"])

                                        if section.get("sub_sections"):
                                            deep_collect_nested(section["sub_sections"])

                                if collected_items:
                                    unique_items = []
                                    seen_display_items = set()
                                    for item in collected_items:
                                        cleaned_item = re.sub(r'^- ', '', str(item)).strip() 
                                        if cleaned_item and cleaned_item not in seen_display_items:
                                            unique_items.append(cleaned_item)
                                            seen_display_items.add(cleaned_item)

                                    if unique_items:
                                        list_items = "\n".join([f"- {item}" for item in unique_items])
                                        response_text = f"{response_heading}\n{list_items}" 
                                        return [('Answer', response_text, best_url_for_topic_specific_list)] 
                                    else:
                                        return None 
                                else:
                                    return None 
                            else:
                                return None 
                        else:
                            return None 
                    else:
                        return None 
                else:
                    app_logger.error("Model or embeddings not available for specific list retrieval. Falling back to semantic search.")
                    return [('Error', 'Our knowledge base is currently optimizing for performance. Please bear with me and try your query again shortly.', None)] 
            
            if target_category in ["services", "products", "use_cases"]: 
                if not homepage_url:
                    app_logger.error("homepage_url is not set. Cannot retrieve specific lists.")
                    return [('Answer', f"Our knowledge base is currently optimizing for performance. Please bear with me and try your query again shortly.", None)] 

                index_page_data = scraped_data.get(homepage_url) 
                if index_page_data:
                    item_list = index_page_data.get(target_category, []) 
                    if item_list and isinstance(item_list, list):
                        list_items = "\n".join([f"- {item}" for item in item_list])
                        response_text = f"{response_heading}\n{list_items}" 
                        return [('Answer', response_text, homepage_url)] 
                    else:
                        return [('Answer', f"I was unable to locate a comprehensive list of {target_category} on the homepage. You might consider refining your query or exploring our website for detailed information.", None)] 
                else: 
                    return [('Answer', f"Accessing the homepage data for {target_category} encountered an issue. Please verify the integrity of your scraped_content.json for '{homepage_url}'. For immediate support, kindly visit our official website. Is there anything else I can clarify?", None) ] 
            
            return None

        except Exception as e:
            app_logger.error(f"Error retrieving specific list for '{target_category}' with topic '{target_topic}': {e}")
            return [('Error', 'An internal error occurred while retrieving the requested list. We apologize for the inconvenience. Please try again in a moment, or consider browsing our website directly.', None)] 
    
    return None

def get_detailed_info_for_service(query):
    """
    Attempts to find specific details (features, benefits, applications, tools, overview) 
    for a named service/product by traversing the structured_content in scraped_content.json.
    """
    global model, corpus_embeddings, corpus_texts, corpus_urls, service_page_map, scraped_data
    
    if model is None or corpus_embeddings is None or len(corpus_texts) == 0:
        app_logger.warning("Model/Embeddings not fully initialized. Attempting to load/regenerate.")
        load_or_generate_embeddings() 
        if model is None or corpus_embeddings is None or len(corpus_texts) == 0:
            app_logger.error("Failed to load/regenerate model or embeddings. Cannot perform targeted search.")
            return [('Answer', 'The chatbot is still initializing or encountered a critical error loading its knowledge. Please try again in a moment.')]

    query_lower = query.lower()
    
    target_url = None
    matched_service_keyword = None
    for keyword, url in service_page_map.items():
        # Check for full keyword match first, then partial if query is longer
        if keyword == query_lower:
            target_url = url
            matched_service_keyword = keyword
            break
        elif len(query_lower) > 5 and keyword in query_lower: # Prioritize more specific matches for longer queries
            target_url = url
            matched_service_keyword = keyword
            break # Take the first reasonable match

    if target_url and target_url in scraped_data:
        app_logger.info(f"Detected specific query for {target_url}. Attempting structured content extraction.")
        page_content_data = scraped_data[target_url]
        structured_content = page_content_data.get('structured_content', []) 

        structured_query_keywords = {
            "features": ["key features", "features", "capabilities", "functionalities", "what it does"],
            "benefits": ["benefits", "advantages", "perks", "why choose", "value proposition"],
            "applications": ["applications", "use cases", "implementation", "scenarios", "where it applies"],
            "tools": ["tools", "technologies", "platforms", "stack used"],
            "overview": ["overview", "introduction", "what is", "about", "about this", "description"] 
        }

        extracted_details = []
        response_type_key = "" 
        main_section_heading_text = "" 

        # First, try to find a specific section (features, benefits, etc.)
        for key, keywords_list in structured_query_keywords.items():
            if any(kw in query_lower for kw in keywords_list):
                for i, section in enumerate(structured_content):
                    normalized_h = _normalize_text_for_alias(section.get("heading_text", ""))
                    if any(kw in normalized_h for kw in keywords_list) and section.get("tag", "").startswith('h'):
                        main_section_heading_text = section.get("heading_text", "")
                        parent_heading_level = int(section.get("tag")[1])
                        
                        extracted_details = _get_section_and_sub_content(structured_content, i, parent_heading_level)
                        response_type_key = key
                        break
                if extracted_details:
                    break
        
        # If no specific section (features, benefits) was found, try to extract a general overview
        if not extracted_details and matched_service_keyword:
            app_logger.info(f"No specific section found for '{query_lower}'. Attempting to find general overview.")
            # Find the main heading for the service/product on its page
            for i, section in enumerate(structured_content):
                normalized_h = _normalize_text_for_alias(section.get("heading_text", ""))
                
                # Broaden the match for overview to catch main page titles and primary H1/H2s
                is_main_page_heading = (normalized_h == _normalize_text_for_alias(page_content_data.get('page_title', '')) or \
                                        normalized_h == matched_service_keyword or \
                                        (section.get('tag') in ['h1', 'h2'] and matched_service_keyword in normalized_h))
                
                if is_main_page_heading:
                    # Try to extract initial paragraphs directly following this main service heading
                    initial_overview_paragraphs = []
                    current_idx = i + 1
                    while current_idx < len(structured_content):
                        element = structured_content[current_idx]
                        element_tag = element.get("tag", "")
                        element_heading_text = _normalize_text_for_alias(element.get("heading_text", ""))
                        element_level = int(element_tag[1]) if element_tag.startswith('h') else 99

                        # Stop if we hit another significant heading (H1/H2/H3) or a new major section
                        if element_tag.startswith('h') and (element_level <= 3 or element_heading_text in MAJOR_SECTION_KEYWORDS):
                            break
                        
                        if element_tag == 'p' or element_tag == 'li':
                            for p_text in element.get("content_paragraphs", []):
                                if p_text.strip():
                                    initial_overview_paragraphs.append(p_text.strip())
                        
                        # Collect maximum of 3-5 overview paragraphs to keep it concise
                        if len(initial_overview_paragraphs) >= 3: # Limit overview length
                            break
                        current_idx += 1
                    
                    if initial_overview_paragraphs:
                        extracted_details = initial_overview_paragraphs
                        response_type_key = "overview"
                        # Set main_section_heading_text for overview to be the service name/title
                        main_section_heading_text = page_content_data.get('page_title', matched_service_keyword)
                        if main_section_heading_text.lower().startswith('itech software group'):
                             main_section_heading_text = main_section_heading_text[len('itech software group'):].strip()
                        if not main_section_heading_text: # Fallback to matched service keyword if title is just "iTech Software Group"
                            main_section_heading_text = matched_service_keyword.title()
                        break


        if extracted_details:
            detail_list_text = "\n".join([f"- {item.strip()}" for item in extracted_details if item.strip()])
            
            # Special formatting for "Here's a key point:"
            if response_type_key in ["benefits", "features"]:
                response_text = f"Here's a key point: {main_section_heading_text}\n{detail_list_text}"
            elif response_type_key == "overview":
                 response_text = f"Here's an overview of {main_section_heading_text}:\n{detail_list_text}"
            else:
                response_text = f"Here are some {response_type_key} for {matched_service_keyword}:\n{detail_list_text}"
            
            return [('Answer', response_text, target_url)] # Return URL here
        
        app_logger.info(f"Structured extraction failed or was empty for '{query}' on {target_url}. Falling back to semantic search on the page.")
        
        # Fallback to semantic search on the specific page if structured extraction fails
        target_indices = [i for i, url in enumerate(corpus_urls) if url == target_url]
        
        if not target_indices:
            app_logger.warning(f"No embeddings found for specific page: {target_url} for semantic fallback.")
            return None 

        try:
            page_specific_embeddings = corpus_embeddings[target_indices]
            page_specific_texts = [corpus_texts[i] for i in target_indices]

            query_embedding = model.encode(query, convert_to_tensor=True)
            
            cosine_scores = util.cos_sim(query_embedding, page_specific_embeddings)[0]
            
            top_k_candidates_page = min(10, len(page_specific_texts)) 
            top_results_page = torch.topk(cosine_scores, k=top_k_candidates_page)

            best_snippet_info = None
            
            for score, idx in zip(top_results_page[0], top_results_page[1]):
                candidate_text = page_specific_texts[idx]
                is_substantial = len(candidate_text.split()) > 15 
                
                # Prioritize content that is not a generic heading if other options exist
                normalized_candidate = _normalize_text_for_alias(candidate_text)
                is_generic_heading = any(kw in normalized_candidate for kw in ["features", "benefits", "applications", "tools", "overview"]) and len(normalized_candidate.split()) < 4

                if score > 0.48 and not (best_snippet_info and not is_substantial and is_generic_heading): # Don't take a generic heading if we have something substantial
                    if best_snippet_info is None or \
                       (is_substantial and not best_snippet_info['is_substantial']) or \
                       (score > best_snippet_info['score'] and (is_substantial == best_snippet_info['is_substantial'])) or \
                       (score == best_snippet_info['score'] and is_substantial and len(candidate_text) > len(best_snippet_info['text'])):
                        best_snippet_info = {
                            'score': score,
                            'text': candidate_text,
                            'url': target_url,
                            'is_substantial': is_substantial
                        }
            
            if best_snippet_info and best_snippet_info['score'] > 0.45: 
                short_answer = best_snippet_info['text'][:500] + ('...' if len(best_snippet_info['text']) > 500 else '')
                return [('Answer', f"{short_answer} (Source: {best_snippet_info['url']})")]
            else: 
                app_logger.info(f"Targeted semantic search for '{query}' on {target_url} found no suitable answer. Returning None.")
                return None

        except Exception as e:
            app_logger.error(f"Error during targeted semantic search for '{query}': {e}")
            return [('Error', 'An error occurred during the search for specific details. Please try again later.', None)]

    return None 

def search_content(query, max_results=1, max_length=500): 
    """
    Performs a general semantic search on the entire corpus.
    It aims to find the single most relevant and substantial answer.
    """
    global model, corpus_embeddings, corpus_texts, corpus_urls, scraped_data

    if model is None or corpus_embeddings is None or len(corpus_texts) == 0:
        app_logger.warning("Model/Embeddings not fully initialized during search request. Attempting to load/regenerate.")
        load_or_generate_embeddings()

        if model is None or corpus_embeddings is None or len(corpus_texts) == 0:
            app_logger.error("Failed to load/regenerate model or embeddings. Cannot perform search.")
            return [('Error', 'My knowledge base is currently being loaded. This process may take a moment. Please try your query again shortly. Thank you for your patience.', None)] 

    try:
        query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        
        cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        
        top_k_candidates = min(len(corpus_texts), 200) 
        top_results = torch.topk(cosine_scores, k=top_k_candidates) 
        
        results_by_url = {} 
        
        sorted_indices = top_results[1][top_results[0].argsort(descending=True)]
        sorted_scores = top_results[0][top_results[0].argsort(descending=True)]

        score_threshold = 0.40 

        query_lower = query.lower()
        is_benefits_query = "benefits of" in query_lower or "advantages of" in query_lower or query_lower == "benefits" or query_lower == "advantages"
        is_features_query = "features of" in query_lower or "functionalities of" in query_lower or "capabilities of" in query_lower or query_lower == "features" or query_lower == "functionalities" or query_lower == "capabilities"
        
        is_definition_query = re.search(r'^(what is|what is mean by|define|explain)\s+', query_lower) is not None

        best_definitional_snippet = None
        best_definitional_score = 0.0
        
        definitional_term = None
        if is_definition_query:
            match = re.search(r'(?:what is|what is mean by|define|explain)\s+(.+)', query_lower)
            if match:
                definitional_term = match.group(1).strip()
                definitional_term = re.sub(r'(?:your|the|a|an)$', '', definitional_term).strip()
            app_logger.info(f"Definitional query detected. Term: '{definitional_term}'")


        for score, idx in zip(sorted_scores, sorted_indices):
            original_text = corpus_texts[idx]
            source_url = corpus_urls[idx]

            if is_definition_query and definitional_term: # CRITICAL FIX: Changed `best_definitional_snippet` to `definitional_term`
                if definitional_term in original_text.lower() and score > 0.75 and len(original_text.split()) < 75:
                    page_content = scraped_data.get(source_url, {})
                    is_paragraph_or_extracted_text = False
                    if original_text in page_content.get('paragraphs', []) or original_text == page_content.get('extracted_text', '').strip():
                        is_paragraph_or_extracted_text = True
                    
                    if is_paragraph_or_extracted_text:
                        if score > best_definitional_score:
                            best_definitional_score = score
                            best_definitional_snippet = {
                                'score': score,
                                'text': original_text,
                                'url': source_url,
                            }
                            app_logger.info(f"Found potential best definitional snippet: '{original_text[:50]}...' (Score: {score})")
            
            if score < score_threshold:
                continue 
            
            is_substantial_text = len(original_text.split()) > 8 

            current_best_for_url = results_by_url.get(source_url)

            text_starts_with_benefits_prefix = re.match(r'benefits of.*?:', original_text, re.IGNORECASE) or \
                                                re.match(r'advantages of.*?:', original_text, re.IGNORECASE)
            text_starts_with_features_prefix = re.match(r'features of.*?:', original_text, re.IGNORECASE) or \
                                                re.match(r'functionalities of.*?:', original_text, re.IGNORECASE) or \
                                                re.match(r'capabilities of.*?:', original_text, re.IGNORECASE)

            should_prioritize_this_snippet = False
            if is_benefits_query and text_starts_with_benefits_prefix and score > 0.65:
                should_prioritize_this_snippet = True
            elif is_features_query and text_starts_with_features_prefix and score > 0.65:
                should_prioritize_this_snippet = True

            if should_prioritize_this_snippet:
                results_by_url[source_url] = {
                    'score': score,
                    'text': original_text,
                    'url': source_url,
                    'is_substantial': True 
                }
            elif current_best_for_url is None:
                results_by_url[source_url] = {
                    'score': score,
                    'text': original_text,
                    'url': source_url,
                    'is_substantial': is_substantial_text
                }
            else:
                if is_substantial_text and not current_best_for_url['is_substantial']:
                    results_by_url[source_url] = {
                        'score': score,
                        'text': original_text,
                        'url': source_url,
                        'is_substantial': is_substantial_text
                    }
                elif score > current_best_for_url['score']:
                    results_by_url[source_url] = {
                        'score': score,
                        'text': original_text,
                        'url': source_url,
                        'is_substantial': is_substantial_text
                    }
                elif (score == current_best_for_url['score'] and
                      is_substantial_text == current_best_for_url['is_substantial'] and
                      len(original_text) > len(current_best_for_url['text'])):
                     results_by_url[source_url] = {
                        'score': score,
                        'text': original_text,
                        'url': source_url,
                        'is_substantial': is_substantial_text
                    }

        final_results = []
        
        sorted_best_snippets = []
        if results_by_url: 
            sorted_best_snippets = sorted(
                results_by_url.values(), 
                key=lambda x: (x['is_substantial'], x['score'], len(x['text'])),
                reverse=True 
            )
        
        if is_definition_query and best_definitional_snippet:
            app_logger.info(f"Prioritizing best definitional snippet for query. Score: {best_definitional_snippet['score']}")
            answer_text = best_definitional_snippet['text']
            if len(answer_text.split()) > 100: 
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', answer_text)
                answer_text = " ".join(sentences[:min(len(sentences), 3)]) 
                if len(sentences) > 3 and len(answer_text.split()) < len(best_definitional_snippet['text'].split()):
                    answer_text += '...'
            final_results.append(('Answer', answer_text, best_definitional_snippet['url']))
        elif sorted_best_snippets:
            best_snippet_info = sorted_best_snippets[0] 
            app_logger.info(f"Using top general snippet. Score: {best_snippet_info['score']}")

            if (is_benefits_query or is_features_query) and (best_snippet_info['text'].lower().startswith(("benefits of", "advantages of", "features of", "functionalities of", "capabilities of"))):
                prefix_removed_text = re.sub(r'^(benefits of|features of|advantages of|functionalities of|capabilities of).*?:', '', best_snippet_info['text'], flags=re.IGNORECASE).strip()
                final_results.append(('Answer', prefix_removed_text, best_snippet_info['url']))
            else:
                combined_texts = [best_snippet_info['text']]
                combined_urls = {best_snippet_info['url']}

                if len(best_snippet_info['text'].split()) < 30 or len(query.split()) < 3:
                    for i in range(1, min(len(sorted_best_snippets), 3)): 
                        current_snippet = sorted_best_snippets[i]
                        if len(current_snippet['text'].split()) > 5 and \
                           current_snippet['url'] != best_snippet_info['url'] and \
                           current_snippet['score'] > (best_snippet_info['score'] * 0.7):
                            combined_texts.append(current_snippet['text'])
                            combined_urls.add(current_snippet['url'])
                            if len(combined_texts) >= max_results: 
                                break
                        elif len(combined_texts) == 1 and current_snippet['url'] == best_snippet_info['url'] and \
                             len(current_snippet['text'].split()) > 5 and current_snippet['score'] > (best_snippet_info['score'] * 0.85):
                            combined_texts.append(current_snippet['text'])
                            
                final_combined_text = " ".join(combined_texts)
                
                short_answer = final_combined_text[:max_length] 
                if len(final_combined_text) > max_length:
                    last_period_idx = short_answer.rfind('.')
                    if last_period_idx > max_length * 0.8: 
                        short_answer = short_answer[:last_period_idx + 1]
                    else:
                        short_answer += '...'
                
                final_results.append(('Answer', short_answer, list(combined_urls)[0] if combined_urls else None))

        if not final_results:
            app_logger.info("No suitable answer found after all search attempts.")
            return [('Answer', 'I couldn\'t find information directly relevant to your request in our knowledge base. Please try rephrasing your question or explore our services in more detail on our website. We are always here to help!', None)]
        
        return final_results

    except Exception as e:
        app_logger.error(f"Error during semantic search for query '{query}': {e}")
        return [('Error', 'I apologize, an unexpected error occurred while processing your request. Our team is working to resolve this. Please try again in a few moments, or feel free to contact us directly for immediate assistance. Thank you for your understanding.', None)] 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_route(): 
    app_logger.info("Received request to /search endpoint.")
    
    data = request.get_json(silent=True, force=True) 

    app_logger.info(f"Request data received by Flask: {data}")

    if data is None:
        app_logger.error("Request body is not valid JSON or is empty, or Content-Type is incorrect.")
        return jsonify({'response_type': 'Error', 'response_content': 'Error: Invalid or empty request body. Please ensure your query is sent as valid JSON (check browser console for network errors).', 'source_url': None}), 400

    user_query = data.get('query')

    app_logger.info(f"Extracted query: '{user_query}'")

    if not user_query:
        app_logger.warning("User query is empty after extraction.")
        return jsonify({'response_type': 'Answer', 'response_content': 'Please enter a query.', 'source_url': None})

    # Check for intent match first (greetings, farewells)
    intent = match_intent(user_query)
    if intent:
        app_logger.info(f"Intent '{intent}' matched for query '{user_query}'. Getting intent response.")
        intent_response_text = get_intent_response(intent)
        return jsonify({'response_type': 'Answer', 'response_content': intent_response_text, 'source_url': None})


    list_response = get_specific_list_response(user_query)
    if list_response:
        app_logger.info(f"Sending specific list response: {list_response[0][1][:100]}...")
        # list_response is structured as [('Type', content, url)]
        response_type = list_response[0][0]
        response_content = list_response[0][1]
        response_url = list_response[0][2]
        return jsonify({'response_type': response_type, 'response_content': response_content, 'source_url': response_url})

    # The original get_detailed_info_for_service was not structured to return URL
    # So if it's still being called, it needs to be updated to match the new return format
    # For now, I'm commenting it out to rely on get_specific_list_response and search_content.
    # If it's intended to be used, it should also return [('Type', content, url)]
    # As per previous conversation, this function might have been a legacy from an earlier iteration.
    # For now, I'm commenting it out to rely on get_specific_list_response and search_content.
    # detailed_info_response = get_detailed_info_for_service(user_query)
    # if detailed_info_response:
    #     app_logger.info(f"Sending detailed info response: {detailed_info_response[0][1][:100]}...")
    #     return jsonify({'response': detailed_info_response[0][1]})

    search_results = search_content(user_query)
    
    # search_results is now a list of tuples like [('Answer', text, url)] or [('Error', text, None)]
    if search_results and len(search_results[0]) >= 2: # Check for at least type and content
        response_type = search_results[0][0]
        response_content = search_results[0][1]
        response_url = search_results[0][2] if len(search_results[0]) > 2 else None
        
        return jsonify({'response_type': response_type, 'response_content': response_content, 'source_url': response_url})
    else:
        # Fallback for unexpected format from search_content
        return jsonify({'response_type': 'Error', 'response_content': 'An unexpected response format was received. Please try again.', 'source_url': None})

# --- Main execution block ---
if __name__ == '__main__':
    app_logger.info("Starting Flask application...")
    load_scraped_content() # Ensure scraped_data and homepage_url are loaded
    load_intent_data() # Load intents and training data
    load_or_generate_embeddings() # Load/generate embeddings after content and intent data is ready
    
    app.run(debug=True)
