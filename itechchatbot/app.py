import json
import os
import re
import logging
from collections import Counter 
from pathlib import Path
from datetime import datetime # Import for get_current_time function
from urllib.parse import urlparse # Ensure urlparse is imported at the top
import random # Added import for the 'random' module

import torch 
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util

# --- Global variables for model and embeddings ---
# These will be loaded once when the application starts
model = None
corpus_embeddings = None
corpus_texts = []
corpus_urls = []

# --- Global variables for Intent Data ---
intents_data = {}
training_data = [] # List of {'text': '...', 'intent': '...'}
# Using a dictionary for faster lookup by normalized text
training_phrases_map = {} # {'normalized_text': 'intent_name'}

# Define file paths relative to the current script
BASE_DIR = Path(__file__).parent
# CRITICAL FIX: Adjusted paths to include 'modules' as per your file structure
STATIC_DIR = BASE_DIR / 'modules' / 'static' 
DATA_DIR = STATIC_DIR / 'data' # Data is still inside static

EMBEDDINGS_FILE = DATA_DIR / 'scraped_content_embeddings.json'
CONTENT_FILE = DATA_DIR / 'scraped_content.json'
INTENTS_FILE = BASE_DIR / 'intents.json' # Assuming intents.json is in the same directory as app.py
TRAINING_FILE = BASE_DIR / 'training.json' # Assuming training.json is in the same directory as app.py
CHATBOT_CONFIG_FILE = BASE_DIR / 'config.json' # Assuming config.json is in the base directory

# Ensure the data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Flask App Setup ---
# CRITICAL FIX: Adjusted template_folder path in Flask constructor
app = Flask(__name__,
            static_folder=str(STATIC_DIR),
            template_folder=str(BASE_DIR / 'modules' / 'templates')) 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app_logger = logging.getLogger(__name__)

# --- Global variables for scraped data and homepage URL ---
scraped_data = {}
homepage_url = "" # To store the normalized homepage URL from scraped data

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
    global intents_data, training_data, training_phrases_map

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

# CRITICAL FIX: Added this function for the "clients" intent
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
        return

    app_logger.info("Checking for existing embeddings to load or generate...")

    embeddings_exist_and_not_outdated = False
    if EMBEDDINGS_FILE.exists() and os.path.getsize(EMBEDDINGS_FILE) > 0:
        if CONTENT_FILE.exists() and CONTENT_FILE.stat().st_mtime > EMBEDDINGS_FILE.stat().st_mtime:
            app_logger.warning("Scraped content file is newer than embeddings file. Regenerating embeddings.")
            embeddings_exist_and_not_outdated = False # Force regeneration
        else:
            try:
                with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    corpus_embeddings = torch.tensor(data['embeddings'])
                    corpus_texts = data['texts']
                    corpus_urls = data['urls']
                app_logger.info(f"Loaded {len(corpus_embeddings)} existing embeddings from {EMBEDDINGS_FILE}.")
                embeddings_exist_and_not_outdated = True
            except (json.JSONDecodeError, KeyError, TypeError, FileNotFoundError) as e:
                app_logger.warning(f"Error loading embeddings file ({e}). Embeddings will be regenerated.")
                corpus_embeddings = None
                corpus_texts = []
                corpus_urls = []
            except Exception as e:
                app_logger.warning(f"An unexpected error occurred while loading embeddings ({e}). Embeddings will be regenerated.")
                corpus_embeddings = None
                corpus_texts = []
                corpus_urls = []

    if embeddings_exist_and_not_outdated:
        if model is None:
            try:
                app_logger.info("Model was None, loading SentenceTransformer model 'all-mpnet-base-v4' after loading embeddings...")
                # Load model from the bundled directory
                model = SentenceTransformer(str(BASE_DIR / 'all-mpnet-base-v4'))
                app_logger.info("SentenceTransformer model loaded successfully (after embeddings).")
            except Exception as e:
                app_logger.error(f"Failed to load SentenceTransformer model after embeddings: {e}")
                model = None
                corpus_embeddings = None
                corpus_texts = []
                corpus_urls = []
        if model is not None and corpus_embeddings is not None:
            return # Successfully loaded existing embeddings and model.

    app_logger.info("Embeddings not found/loaded or content updated. Generating new embeddings.")

    # Ensure scraped_data is loaded before generating embeddings
    if not scraped_data: # Check global scraped_data, assuming load_scraped_content() was called.
        if not load_scraped_content(): # Attempt to load if not already loaded
            app_logger.error("Cannot generate embeddings: Scraped data is empty or missing after attempting to load.")
            corpus_embeddings = None
            corpus_texts = []
            corpus_urls = []
            return

    texts_for_embedding = []
    urls_for_embedding = []
    seen_snippets = set() # For deduplication during embedding generation

    for url, page_content in scraped_data.items():
        # Process paragraphs
        paragraphs = page_content.get('paragraphs', [])
        for p_text in paragraphs:
            cleaned_p_text = p_text.strip()
            if cleaned_p_text and cleaned_p_text not in seen_snippets:
                texts_for_embedding.append(cleaned_p_text)
                urls_for_embedding.append(url)
                seen_snippets.add(cleaned_p_text)

        # Process headings
        headings = page_content.get('headings', [])
        for h_text in headings:
            cleaned_h_text = h_text.strip()
            if cleaned_h_text and cleaned_h_text not in seen_snippets:
                texts_for_embedding.append(cleaned_h_text)
                urls_for_embedding.append(url)
                seen_snippets.add(cleaned_h_text)
        
        # Process specific lists like 'services', 'products', 'use_cases', 'benefits', 'features'
        for category_key in ['services', 'products', 'use_cases', 'benefits', 'features']:
            items = page_content.get(category_key, [])
            if isinstance(items, list): 
                for item_text in items:
                    cleaned_item_text = str(item_text).strip()
                    if cleaned_item_text: 
                        page_title = page_content.get('page_title', url.split('/')[-1].replace('.html', '').replace('-', ' ').title())
                        # This forms the embedding text for structured items like "Service of Homepage: Digital Transformation"
                        text_to_embed = f"{category_key.replace('_', ' ').title()} of {page_title}: {cleaned_item_text}"
                        
                        if text_to_embed not in seen_snippets:
                            texts_for_embedding.append(text_to_embed)
                            urls_for_embedding.append(url)
                            seen_snippets.add(text_to_embed) 

        # Add 'extracted_text' as a last resort or for larger chunks of text
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
            app_logger.info("Loading SentenceTransformer model 'all-mpnet-base-v4' for generation...")
            # Load model from the bundled directory
            model = SentenceTransformer(str(BASE_DIR / 'all-mpnet-base-v4'))
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

def get_specific_list_response(query):
    """
    Checks if the query is asking for a specific list (services, products, use cases, benefits, features).
    If so, retrieves the list from scraped_content.json (index.html or other relevant page) and formats it.
    """
    global model, corpus_embeddings, corpus_texts, corpus_urls, homepage_url # Added homepage_url to global

    query_lower = query.lower().strip()
    
    services_keywords = ["services", "list of services", "our services", "what services do you offer", "tell me about your services", "company services"]
    products_keywords = ["products", "list of products", "our products", "what products do you have", "tell me about your products", "company products"]
    use_cases_keywords = ["use case","use cases", "list of use cases", "our use cases", "examples of use cases", "tell me about use cases", "company use cases"] # Added "applications"
    benefits_keywords = ["benefits of", "advantages of", "what are the benefits of", "tell me the benefits of", "benefits", "advantages", "perks", "value proposition"] # Added direct keywords
    features_keywords = ["features of", "functionalities of", "capabilities of", "what are the features of", "tell me the features of", "features", "functionalities", "capabilities"] # Added direct keywords

    target_category = None
    response_heading = ""
    target_topic = None 

    if any(keyword in query_lower for keyword in services_keywords):
        target_category = "services"
        # MODIFIED: More professional heading
        response_heading = "Allow me to present the services offered by KTG Software and Consulting:"
    elif any(keyword in query_lower for keyword in products_keywords):
        target_category = "products"
        # MODIFIED: More professional heading
        response_heading = "Discover the products provided by KTG Software and Consulting:"
    elif any(keyword in query_lower for keyword in use_cases_keywords):
        target_category = "use_cases"
        # MODIFIED: More professional heading
        response_heading = "Explore key use cases for KTG Software and Consulting's solutions:"
    elif any(keyword in query_lower for keyword in benefits_keywords):
        target_category = "benefits"
        match = re.search(r'(?:benefits of|advantages of|what are the benefits of|tell me the benefits of)\s+(.+)', query_lower)
        if match:
            target_topic = match.group(1).strip()
            target_topic = re.sub(r'(?:your|the|a|an)$', '', target_topic).strip()
            # MODIFIED: More professional heading
            response_heading = f"Certainly, here are the core benefits of {target_topic}:"
        else:
            # MODIFIED: More professional heading
            response_heading = "Here are some key benefits you might find valuable:"
    elif any(keyword in query_lower for keyword in features_keywords):
        target_category = "features"
        match = re.search(r'(?:features of|functionalities of|capabilities of|what are the features of|tell me the features of)\s+(.+)', query_lower)
        if match:
            target_topic = match.group(1).strip()
            target_topic = re.sub(r'(?:your|the|a|an)$', '', target_topic).strip()
            # MODIFIED: More professional heading
            response_heading = f"To elaborate on {target_topic}, here are its primary features:"
        else:
            # MODIFIED: More professional heading
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
                        # Only consider pages that have 'structured_content' for specific list extraction
                        if current_url in scraped_data and scraped_data[current_url].get('structured_content'):
                            page_title_lower = scraped_data[current_url].get('page_title', '').lower()
                            headings_lower = [h.lower() for h in scraped_data[current_url].get('headings', [])]
                            
                            # Check if the target topic is strongly present in the page title or headings
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
                                
                                if start_collecting:
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
                                    # Removed trailing sentence from response_text, frontend handles the closing phrase
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
                        # Removed trailing sentence from response_text, frontend handles the closing phrase
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

def search_content(query, max_results=1, max_length=500): 
    """
    Performs a general semantic search on the entire corpus.
    It aims to find the single most relevant and substantial answer.
    """
    global model, corpus_embeddings, corpus_texts, corpus_urls

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

        score_threshold = 0.40 # Adjusted to be slightly more lenient for general queries

        query_lower = query.lower()
        is_benefits_query = "benefits of" in query_lower or "advantages of" in query_lower or query_lower == "benefits" or query_lower == "advantages"
        is_features_query = "features of" in query_lower or "functionalities of" in query_lower or "capabilities of" in query_lower or query_lower == "features" or query_lower == "functionalities" or query_lower == "capabilities"
        
        # Modified the is_definition_query to be more flexible
        # It now explicitly checks for "what is", "what is mean by", "define", or "explain" at the beginning of the query
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

            # Prioritize direct definitions for definitional queries
            # Increased score threshold for direct definitional snippets
            if is_definition_query and definitional_term:
                # Check if the original text contains the definitional term AND has a good score
                # AND is relatively concise (like a definition). Also check if it's a paragraph or extracted text.
                if definitional_term in original_text.lower() and score > 0.75 and len(original_text.split()) < 75: # Increased word limit for definition
                    # Further check if this snippet comes from a paragraph or the main extracted text,
                    # to avoid picking up short headings as definitions.
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
            
            # General snippet selection logic
            if score < score_threshold:
                continue 
            
            is_substantial_text = len(original_text.split()) > 8 # Adjusted minimum words for substantiality

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
        
        # --- Final Answer Construction Logic ---
        if is_definition_query and best_definitional_snippet:
            app_logger.info(f"Prioritizing best definitional snippet for query. Score: {best_definitional_snippet['score']}")
            answer_text = best_definitional_snippet['text']
            # Trim if very long but still a definition
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

                # Attempt to combine more snippets if the initial best is short or query is broad
                if len(best_snippet_info['text'].split()) < 30 or len(query.split()) < 3:
                    for i in range(1, min(len(sorted_best_snippets), 3)): 
                        current_snippet = sorted_best_snippets[i]
                        # Only add if it's substantial, not from the same URL (unless it's the only other option), and highly relevant
                        if len(current_snippet['text'].split()) > 5 and \
                           current_snippet['url'] != best_snippet_info['url'] and \
                           current_snippet['score'] > (best_snippet_info['score'] * 0.7): # Score must be at least 70% of the top snippet
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
    # For now, I'm assuming search_content will handle most cases and this might be less critical.
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
