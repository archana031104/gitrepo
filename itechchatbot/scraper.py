import requests
from bs4 import BeautifulSoup
import logging
import time
import json
import configparser
from urllib.parse import urlparse, urljoin
from collections import deque
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
from pathlib import Path 
import re 

# Configure logging for the scraper
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
scraper_logger = logging.getLogger('WebScraper')

class WebScraper:
    def __init__(self, config_file='config.ini'):
        self.config = configparser.ConfigParser()
        # Resolve config_path relative to the script's directory
        config_path = Path(__file__).parent / config_file
        if not self.config.read(config_path):
            self.logger.warning(f"Config file '{config_path}' not found. Using default scraper settings.")
            # Default settings if config.ini is not found
            self.config['SCRAPER'] = {
                'timeout': '15',
                'max_retries': '3',
                'sleep_between': '2.0',
                'use_selenium': 'True', 
                'max_crawl_depth': '2',
                'max_pages_to_crawl': '100',
                'selenium_load_wait_time': '5.0', # Increased default for dynamic sites
                'root_url': 'https://www.itechsoftwaregroup.com/',
                'allowed_domains': 'www.itechsoftwaregroup.com,itechsoftwaregroup.com', 
                'service_keywords': 'digital transformation,ai / ml,extended warranty management,business intelligence,api services,webhooks',
                'product_keywords': 'asset tracking & management system,traceability,iot',
                'use_case_keywords': 'compressed air flow monitoring,rfid based asset tracking,iot gateway for plc'
            }
        
        self.logger = logging.getLogger('WebScraper')
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        ]
        
        # Load settings from config.ini
        self.timeout = self.config.getint('SCRAPER', 'timeout', fallback=15)
        self.max_retries = self.config.getint('SCRAPER', 'max_retries', fallback=3)
        self.sleep_between = self.config.getfloat('SCRAPER', 'sleep_between', fallback=2.0)
        self.use_selenium = self.config.getboolean('SCRAPER', 'use_selenium', fallback=True)
        self.max_crawl_depth = self.config.getint('SCRAPER', 'max_crawl_depth', fallback=2)
        self.max_pages_to_crawl = self.config.getint('SCRAPER', 'max_pages_to_crawl', fallback=100)
        self.selenium_load_wait_time = self.config.getfloat('SCRAPER', 'selenium_load_wait_time', fallback=5.0) # Load new setting
        self.root_url = self.config.get('SCRAPER', 'root_url', fallback='https://ktgsoftware.com/').strip()
        
        allowed_domains_str = self.config.get('SCRAPER', 'allowed_domains', fallback='').strip()
        self.logger.info(f"Raw allowed domains string from config: '{allowed_domains_str}'")
        self.allowed_domains = {self._normalize_domain(d) for d in allowed_domains_str.split(',') if d.strip()}
        self.logger.info(f"Processed allowed domains set: {self.allowed_domains}")

        self.service_keywords = self.config.get('SCRAPER', 'service_keywords', fallback='').split(',')
        self.service_keywords = [kw.strip().lower() for kw in self.service_keywords if kw.strip()]
        self.logger.info(f"Service keywords loaded: {self.service_keywords}")

        self.product_keywords = self.config.get('SCRAPER', 'product_keywords', fallback='').split(',')
        self.product_keywords = [kw.strip().lower() for kw in self.product_keywords if kw.strip()]
        self.logger.info(f"Product keywords loaded: {self.product_keywords}")


        self.use_case_keywords = self.config.get('SCRAPER', 'use_case_keywords', fallback='').split(',')
        self.use_case_keywords = [kw.strip().lower() for kw in self.use_case_keywords if kw.strip()]
        self.logger.info(f"Use case keywords loaded: {self.use_case_keywords}")
        
        self.visited = set()
        self.queued_urls = set()

    def _normalize_text(self, text):
        """Normalizes text by stripping whitespace and extra spaces."""
        if text is None:
            return ""
        return re.sub(r'\s+', ' ', text.replace('\xa0', ' ')).strip()

    def _normalize_domain(self, domain_or_url):
        """Normalizes a domain or URL's netloc by removing 'www.'"""
        if not domain_or_url:
            return ""
        if '://' in domain_or_url: 
            return urlparse(domain_or_url).netloc.replace('www.', '')
        return domain_or_url.replace('www.', '') 

    def _get_clean_url(self, url):
        """Standardizes URL format by removing fragments and ensuring consistent trailing slashes."""
        parsed_url = urlparse(url)
        clean_url = parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
        if clean_url != f"{parsed_url.scheme}://{parsed_url.netloc}" and clean_url.endswith('/'):
            clean_url = clean_url.rstrip('/')
        return clean_url

    def crawl_and_save(self, json_path='static/data/scraped_content.json'):
        """
        Starts the crawling process from the root_url and saves the results to a JSON file.
        """
        results = self.crawl(self.root_url, self.max_crawl_depth)
        
        final_data_to_save = {}
        for page_url, page_data in results.items():
            temp_page_data = page_data.copy()
            if 'images' in temp_page_data:
                del temp_page_data['images']
            final_data_to_save[page_url] = temp_page_data

        try:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(final_data_to_save, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Structured data saved to {json_path}")
        except Exception as e:
            self.logger.error(f"Failed to save structured data to JSON: {e}")

    def crawl(self, start_url, max_depth):
        """
        Performs a breadth-first crawl of the website.
        """
        results = {}
        url_queue = deque()
        self.visited = set()
        self.queued_urls = set()
        pages_processed = 0

        initial_clean_url = self._get_clean_url(start_url)
        if initial_clean_url not in self.queued_urls:
            url_queue.append((initial_clean_url, 1))
            self.queued_urls.add(initial_clean_url)

        self.logger.info(f"Starting crawl from {initial_clean_url} with max depth {max_depth} and max pages to crawl {self.max_pages_to_crawl}")

        while url_queue:
            if pages_processed >= self.max_pages_to_crawl:
                self.logger.warning(f"Maximum page limit of {self.max_pages_to_crawl} reached. Stopping crawl.")
                break

            current_url, current_depth = url_queue.popleft()

            if current_url in self.visited:
                self.logger.info(f"Skipping already processed (depth {current_depth}): {current_url}")
                continue

            parsed_current_url = urlparse(current_url)
            normalized_current_domain = self._normalize_domain(parsed_current_url.netloc)

            if self.allowed_domains and normalized_current_domain not in self.allowed_domains:
                self.logger.info(f"Skipping external domain: {current_url} (Normalized: {normalized_current_domain}, Allowed: {self.allowed_domains})")
                continue

            self.visited.add(current_url)
            pages_processed += 1
            self.logger.info(f"Processing page {pages_processed}/{self.max_pages_to_crawl} (Depth {current_depth}): {current_url}")

            try:
                page_data_container = self.extract(current_url)
                
                if not page_data_container:
                    self.logger.warning(f"No data extracted for {current_url}. Skipping processing of its links.")
                    time.sleep(self.sleep_between)
                    continue

                canonical_url_key = list(page_data_container.keys())[0] if page_data_container else current_url
                page_data = page_data_container[canonical_url_key] if page_data_container else {}

                results[canonical_url_key] = page_data

                if current_depth < max_depth:
                    for link in page_data.get('links', []):
                        if not link or link.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                            self.logger.info(f"Skipping invalid/fragment link: {link} on {canonical_url_key}")
                            continue
                        
                        absolute_link = urljoin(canonical_url_key, link)
                        
                        if not (absolute_link.startswith('http://') or absolute_link.startswith('https://')):
                            self.logger.info(f"Skipping non-HTTP/HTTPS relative link: {link} on {canonical_url_key}")
                            continue

                        clean_absolute_link = self._get_clean_url(absolute_link)
                        parsed_absolute_link = urlparse(clean_absolute_link)
                        normalized_link_domain = self._normalize_domain(parsed_absolute_link.netloc)

                        if clean_absolute_link not in self.visited and \
                           clean_absolute_link not in self.queued_urls and \
                           normalized_link_domain in self.allowed_domains:
                            url_queue.append((clean_absolute_link, current_depth + 1))
                            self.queued_urls.add(clean_absolute_link)
                            self.logger.info(f"Discovered and queued: {clean_absolute_link} (from {canonical_url_key}, Depth {current_depth + 1})")
                        else:
                            skip_reason = []
                            if clean_absolute_link in self.visited:
                                skip_reason.append("already visited")
                            if clean_absolute_link in self.queued_urls:
                                skip_reason.append("already queued")
                            if normalized_link_domain not in self.allowed_domains:
                                skip_reason.append(f"external domain ({normalized_link_domain} not in {self.allowed_domains})")
                            
                            self.logger.info(f"Skipping {clean_absolute_link} (from {canonical_url_key}): {', '.join(skip_reason)}")

            except Exception as e:
                self.logger.error(f"Failed to process {current_url}: {e}")
            
            time.sleep(self.sleep_between) 

        self.logger.info(f"Crawl finished. Scraped {len(results)} pages.")
        return results

    def _get_user_agent(self):
        import random
        return random.choice(self.user_agents)

    def extract(self, url):
        """
        Fetches the raw HTML and determines if it's static or dynamic.
        Then calls the appropriate parsing method.
        Returns:
            dict: A dictionary where the key is the canonical URL and value is the page data.
        """
        for attempt in range(self.max_retries):
            self.logger.info(f"Attempt {attempt+1}/{self.max_retries} to extract content from {url}")
            try:
                headers = {'User-Agent': self._get_user_agent()}
                response = requests.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status() 

                final_url = self._get_clean_url(response.url)

                if self._is_dynamic(response.text) and self.use_selenium:
                    self.logger.info(f"Detected dynamic content for {url}. Using Selenium. Final URL: {final_url}")
                    return {final_url: self._extract_dynamic_content_and_parse(url)}
                else:
                    self.logger.info(f"Using BeautifulSoup for {url}. Final URL: {final_url}")
                    return {final_url: self._parse_html_content(response.text, final_url)}

            except requests.exceptions.Timeout:
                self.logger.warning(f"Request to {url} timed out.")
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"HTTP error for {url}: {e}")
            except Exception as e:
                self.logger.warning(f"An unexpected error occurred during extraction for {url}: {e}")
            
            time.sleep(self.sleep_between) 
        
        self.logger.error(f"Failed to extract content from {url} after {self.max_retries} attempts.")
        return {} 

    def _is_dynamic(self, html):
        """Heuristic to check if the page likely relies on JavaScript for content."""
        # More robust check for dynamic content
        soup = BeautifulSoup(html, 'html.parser')
        body_text_length = len(soup.body.get_text(strip=True)) if soup.body else 0
        num_scripts = len(soup.find_all('script'))
        
        # If body is empty or very small, and there are scripts, it's likely dynamic
        if body_text_length < 100 and num_scripts > 0: 
            return True
        # Check for common SPA root elements
        if soup.find(id='root') or soup.find(id='app') or soup.find(id='__next'):
            return True
        return False

    def _extract_dynamic_content_and_parse(self, url):
        """
        Extracts content using Selenium for JavaScript-rendered pages and then parses it.
        This helper is called by `extract` and returns the page data directly.
        """
        options = Options()
        options.headless = True
        options.add_argument('--no-sandbox') 
        options.add_argument('--disable-dev-shm-usage') 
        options.add_argument('--window-size=1920,1080')

        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(self.timeout) 
            driver.get(url)
            time.sleep(self.selenium_load_wait_time) # Use the configured wait time

            final_url_selenium = self._get_clean_url(driver.current_url)
            html = driver.page_source
            return self._parse_html_content(html, final_url_selenium)
        except TimeoutException:
            self.logger.error(f"Selenium page load timed out for {url}.")
            raise
        except WebDriverException as e:
            self.logger.error(f"Selenium WebDriver error for {url}: {e}")
            raise
        finally:
            if driver:
                driver.quit() 

    def _parse_html_content(self, html, url):
        """
        Parses the HTML content using BeautifulSoup, extracting hierarchical structure
        and other relevant data.
        """
        self.logger.info(f"Parsing HTML content for URL: {url}") 

        soup = BeautifulSoup(html, 'lxml')
        
        page_title = self._normalize_text(soup.title.string) if soup.title else ""
        extracted_text = self._normalize_text(soup.get_text(separator=' ', strip=True))
        
        all_links = [a['href'] for a in soup.find_all('a', href=True) if a.get('href')]
        images = [] # Images are not needed for chatbot knowledge
        
        structured_content_list = []
        
        main_content_area = soup.find('body') 
        if not main_content_area:
            self.logger.warning(f"Could not find a suitable main content area in {url}. Structured content might be incomplete.")
            main_content_area = soup 

        current_level_stack = [{'sections': structured_content_list, 'level': 0}] 

        for element in main_content_area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
            element_text = self._normalize_text(element.get_text())
            if not element_text:
                continue

            if element.name.startswith('h'):
                heading_level = int(element.name[1])
                
                while len(current_level_stack) > 1 and current_level_stack[-1]['level'] >= heading_level:
                    current_level_stack.pop()

                new_section = {
                    "heading_text": element_text,
                    "tag": element.name,
                    "content_paragraphs": [],
                    "sub_sections": []
                }
                current_level_stack[-1]['sections'].append(new_section)
                current_level_stack.append({'sections': new_section["sub_sections"], 'level': heading_level})

            elif element.name == 'p':
                if current_level_stack[-1]['sections'] and current_level_stack[-1]['sections'][-1].get('content_paragraphs') is not None:
                    current_level_stack[-1]['sections'][-1]['content_paragraphs'].append(element_text)
                else:
                    structured_content_list.append({
                        "heading_text": "", 
                        "tag": "p",
                        "content_paragraphs": [element_text],
                        "sub_sections": []
                    })
            elif element.name == 'li':
                if current_level_stack[-1]['sections'] and current_level_stack[-1]['sections'][-1].get('content_paragraphs') is not None:
                    current_level_stack[-1]['sections'][-1]['content_paragraphs'].append(f"- {element_text}")
                else:
                    structured_content_list.append({
                        "heading_text": "", 
                        "tag": "li",
                        "content_paragraphs": [f"- {element_text}"],
                        "sub_sections": []
                    })
        
        services_set = set()
        products_set = set()
        use_cases_set = set()
        page_features_set = set()
        page_benefits_set = set()

        service_keywords_to_exclude = {"services", "solutions", "offerings", "our services", "our solutions", "service", "solution", "offering"} 
        product_keywords_to_exclude = {"products", "our products", "product"} 
        use_case_keywords_to_exclude = {"use cases", "applications", "use case", "our use cases", "key applications", "solutions for", "application", "case study"}

        def _add_filtered_item(item_text, target_set, exclusion_set, min_words=1, max_words=15):
            item_text = self._normalize_text(item_text)
            if not item_text:
                self.logger.debug(f"  _add_filtered_item: Skipping empty item_text.")
                return
            normalized_item_text = item_text.lower()
            
            words = item_text.split()
            if len(words) < min_words or len(words) > max_words:
                self.logger.debug(f"  _add_filtered_item: Skipping '{item_text}' (word count {len(words)}) outside [{min_words}, {max_words}].")
                return
            
            if normalized_item_text in exclusion_set:
                self.logger.debug(f"  _add_filtered_item: Skipping '{item_text}' due to exclusion.")
                return
            
            target_set.add(item_text)
            self.logger.info(f"  _add_filtered_item: Successfully added '{item_text}' to target set. Current target set: {target_set}")


        # --- Extract from MAIN navigation ---
        main_nav = soup.find('nav', class_='navbar')
        self.logger.info(f"Main nav found: {bool(main_nav)}") 
        if main_nav:
            navbar_nav_ul = main_nav.find('ul', class_='navbar-nav')
            self.logger.info(f"Navbar UL found: {bool(navbar_nav_ul)}") 
            if navbar_nav_ul:
                for li_index, li in enumerate(navbar_nav_ul.find_all('li', class_='nav-item', recursive=False)):
                    top_level_link_a = li.find('a', class_='nav-link', recursive=False)
                    if top_level_link_a:
                        top_level_text = self._normalize_text(top_level_link_a.get_text())
                        top_level_href = top_level_link_a.get('href', 'N/A')
                        self.logger.info(f"Processing top-level nav item (Index {li_index}): '{top_level_text}' (Link: {top_level_href})") 
                        dropdown_ul = li.find('ul', class_='dropdown-menu')
                        
                        if dropdown_ul:
                            self.logger.info(f"Dropdown UL found for '{top_level_text}'.")
                            dropdown_links = dropdown_ul.find_all('a', class_='nav-link', href=True)
                            for link_index, a_item in enumerate(dropdown_links):
                                item_text = self._normalize_text(a_item.get_text())
                                item_href = a_item.get('href', 'N/A')
                                self.logger.info(f"  Processing dropdown item (Index {link_index}): '{item_text}' (Link: {item_href}) under '{top_level_text}'") 
                                
                                # Check if top-level text indicates category, then try to add item_text
                                if re.search(r'products', top_level_text, re.IGNORECASE): # Changed to check literal "products"
                                    self.logger.debug(f"    '{top_level_text}' contains 'products'. Attempting to add '{item_text}' to products_set.")
                                    _add_filtered_item(item_text, products_set, product_keywords_to_exclude, 1, 15)
                                elif re.search(r'services', top_level_text, re.IGNORECASE): # Changed to check literal "services"
                                    self.logger.debug(f"    '{top_level_text}' contains 'services'. Attempting to add '{item_text}' to services_set.")
                                    _add_filtered_item(item_text, services_set, service_keywords_to_exclude, 1, 15)
                                elif re.search(r'use cases?', top_level_text, re.IGNORECASE): # Changed to check literal "use case(s)"
                                    self.logger.debug(f"    '{top_level_text}' contains 'use case(s)'. Attempting to add '{item_text}' to use_cases_set.")
                                    _add_filtered_item(item_text, use_cases_set, use_case_keywords_to_exclude, 1, 15)
                                else:
                                    self.logger.debug(f"    '{top_level_text}' did not indicate a clear category for dropdown item '{item_text}'.")
                        else: 
                            if top_level_link_a.get('href') and top_level_link_a.get('href') != '#':
                                self.logger.info(f"  No dropdown for '{top_level_text}', considering top-level link itself for categorization.")
                                # Check top-level link text against all keywords if it's a direct link
                                if any(re.search(re.escape(kw), top_level_text, re.IGNORECASE) for kw in self.service_keywords):
                                    self.logger.debug(f"    '{top_level_text}' matched a service keyword. Attempting to add '{top_level_text}' to services_set.")
                                    _add_filtered_item(top_level_text, services_set, service_keywords_to_exclude, 1, 15) 
                                elif any(re.search(re.escape(kw), top_level_text, re.IGNORECASE) for kw in self.product_keywords):
                                    self.logger.debug(f"    '{top_level_text}' matched a product keyword. Attempting to add '{top_level_text}' to products_set.")
                                    _add_filtered_item(top_level_text, products_set, product_keywords_to_exclude, 1, 15) 
                                elif any(re.search(re.escape(kw), top_level_text, re.IGNORECASE) for kw in self.use_case_keywords):
                                    self.logger.debug(f"    '{top_level_text}' matched a use case keyword. Attempting to add '{top_level_text}' to use_cases_set.")
                                    _add_filtered_item(top_level_text, use_cases_set, use_case_keywords_to_exclude, 1, 15) 
                                else:
                                    self.logger.debug(f"    '{top_level_text}' did not match any category keywords for top-level link.")
            else:
                self.logger.info(f"Could not find ul.navbar-nav in main nav on {url}")
        else:
            self.logger.info(f"Could not find main nav on {url}")

        # --- Fallback to 'Quick Links' section (e.g., in footer) ---
        quick_links_h4 = soup.find('h4', string=re.compile(r'Quick Links', re.IGNORECASE))
        self.logger.info(f"Quick Links h4 found: {bool(quick_links_h4)}") 
        if quick_links_h4:
            parent_of_quick_links_content = quick_links_h4.find_next_sibling(['ul', 'div']) or \
                                            quick_links_h4.find_parent(['div', 'footer'], class_=re.compile(r'footer|quick-links|navbar', re.IGNORECASE))
            self.logger.info(f"Parent of Quick Links content found: {bool(parent_of_quick_links_content)}") 
            if parent_of_quick_links_content:
                for a_tag_index, a_tag in enumerate(parent_of_quick_links_content.find_all('a', href=True)):
                    item_text = self._normalize_text(a_tag.get_text())
                    item_href = a_tag.get('href', 'N/A')
                    self.logger.info(f"  Processing Quick Link item (Index {a_tag_index}): '{item_text}' (Link: {item_href})") 
                    if not item_text:
                        continue
                    
                    # For quick links, match the item_text directly against the specific category keywords
                    if any(re.search(re.escape(kw), item_text, re.IGNORECASE) for kw in self.service_keywords):
                        self.logger.debug(f"    Quick Link '{item_text}' matched a service keyword. Attempting to add to services_set.")
                        _add_filtered_item(item_text, services_set, service_keywords_to_exclude, 1, 15)
                    elif any(re.search(re.escape(kw), item_text, re.IGNORECASE) for kw in self.product_keywords):
                        self.logger.debug(f"    Quick Link '{item_text}' matched a product keyword. Attempting to add to products_set.")
                        _add_filtered_item(item_text, products_set, product_keywords_to_exclude, 1, 15)
                    elif any(re.search(re.escape(kw), item_text, re.IGNORECASE) for kw in self.use_case_keywords):
                        self.logger.debug(f"    Quick Link '{item_text}' matched a use case keyword. Attempting to add to use_cases_set.")
                        _add_filtered_item(item_text, use_cases_set, use_case_keywords_to_exclude, 1, 15)
                    else:
                        self.logger.debug(f"    Quick Link '{item_text}' did not match any category keywords.")
        
        def extract_sub_content(section, target_keywords_for_check, target_set, min_words_for_item=3, max_words_for_item=25): # Broadened max_words
            self.logger.info(f"  Entering extract_sub_content for section with heading: '{section.get('heading_text', '')}'") 
            for p_text in section.get("content_paragraphs", []):
                if p_text.strip():
                    words = p_text.split()
                    if len(words) >= min_words_for_item and len(words) <= max_words_for_item:
                         target_set.add(p_text.strip())
                         self.logger.info(f"    Added content paragraph to target set: '{p_text.strip()}'. Current target set size: {len(target_set)}") 
                    else:
                        self.logger.debug(f"    Skipping content paragraph '{p_text.strip()}' (word count {len(words)}) outside [{min_words_for_item}, {max_words_for_item}].")
            
            for sub_sec in section.get("sub_sections", []):
                sub_heading_text = sub_sec.get("heading_text", "").strip()
                self.logger.info(f"    Processing sub-section heading: '{sub_heading_text}'") 
                if sub_heading_text:
                    words = sub_heading_text.split()
                    if len(words) >= min_words_for_item and len(words) <= max_words_for_item and \
                       not any(kw == sub_heading_text.lower() for kw in target_keywords_for_check):
                        target_set.add(sub_heading_text)
                        self.logger.info(f"      Added sub-section heading to target set: '{sub_heading_text}'. Current target set size: {len(target_set)}") 
                    else:
                        self.logger.debug(f"      Skipping sub-section heading '{sub_heading_text}' (word count {len(words)}) outside [{min_words_for_item}, {max_words_for_item}] or matched exclusion.")
                
                for sub_p_text in sub_sec.get("content_paragraphs", []):
                    if sub_p_text.strip():
                        words = sub_p_text.split()
                        if len(words) >= min_words_for_item and len(words) <= max_words_for_item:
                            target_set.add(sub_p_text.strip())
                            self.logger.info(f"      Added sub-section content paragraph to target set: '{sub_p_text.strip()}'. Current target set size: {len(target_set)}") 
                        else:
                            self.logger.debug(f"      Skipping sub-section content paragraph '{sub_p_text.strip()}' (word count {len(words)}) outside [{min_words_for_item}, {max_words_for_item}].")
                
                if sub_sec.get("sub_sections"):
                    extract_sub_content(sub_sec, target_keywords_for_check, target_set, min_words_for_item, max_words_for_item)

        
        feature_keywords_main = {"key features", "features", "capabilities", "functionalities"}
        benefit_keywords_main = {"benefits", "advantages", "perks", "why choose", "value proposition"}
        
        for section in structured_content_list:
            heading_lower = section.get("heading_text", "").lower()
            self.logger.info(f"Checking structured content section with heading: '{section.get('heading_text', '')}'. Lowercased: '{heading_lower}'") 
            if any(kw in heading_lower for kw in feature_keywords_main):
                self.logger.info(f"  Found feature-related heading. Extracting sub-content for features.")
                extract_sub_content(section, feature_keywords_main, page_features_set, 2, 25) 
            elif any(kw in heading_lower for kw in benefit_keywords_main):
                self.logger.info(f"  Found benefit-related heading. Extracting sub-content for benefits.")
                extract_sub_content(section, benefit_keywords_main, page_benefits_set, 2, 25) 
            elif "use case" in heading_lower or "applications" in heading_lower or "solutions for" in heading_lower:
                self.logger.info(f"  Found use case-related heading. Extracting sub-content.")
                extract_sub_content(section, use_case_keywords_to_exclude, use_cases_set, 2, 25)


        return {
            'page_title': page_title,
            'headings': [self._normalize_text(h.get_text()) for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])], 
            'paragraphs': [self._normalize_text(p.get_text()) for p in soup.find_all('p')], 
            'links': all_links,
            'images': images, 
            'extracted_text': extracted_text,
            'structured_content': structured_content_list,
            'services': sorted(list(services_set)),
            'products': sorted(list(products_set)),
            'use_cases': sorted(list(use_cases_set)),
            'features': sorted(list(page_features_set)),
            'benefits': sorted(list(page_benefits_set))
        }


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'static' / 'data'
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    scraper = WebScraper(str(BASE_DIR / 'config.ini')) # Pass the full path to config.ini
    
    CONTENT_FILE_PATH = DATA_DIR / 'scraped_content.json'

    scraper_logger.info("Starting web scraping process...")
    scraper.crawl_and_save(json_path=CONTENT_FILE_PATH)
    scraper_logger.info("Web scraping process finished.")
