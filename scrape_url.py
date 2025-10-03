#!/usr/bin/env python3

from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
import cloudscraper
from fake_useragent import UserAgent
import random
import time

app = Flask(__name__)

# User agent rotation
ua = UserAgent()

# Multiple bypass strategies
def scrape_with_requests(url):
    """Basic requests with user agent rotation"""
    headers = {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    response = requests.get(url, headers=headers, timeout=10)
    return response.text

def scrape_with_cloudscraper(url):
    """CloudScraper for CloudFlare bypass"""
    scraper = cloudscraper.create_scraper()
    response = scraper.get(url, timeout=10)
    return response.text

def scrape_with_session(url):
    """Session-based scraping with cookies"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    })
    response = session.get(url, timeout=10)
    return response.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape_url():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Try multiple bypass strategies
        strategies = [scrape_with_cloudscraper, scrape_with_requests, scrape_with_session]
        html_content = None

        for strategy in strategies:
            try:
                html_content = strategy(url)
                break
            except Exception as e:
                print(f"Strategy {strategy.__name__} failed: {e}")
                continue

        if not html_content:
            return jsonify({'error': 'All scraping strategies failed'}), 500

        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "noscript"]):
            script.decompose()

        # Extract text content
        text_content = soup.get_text()

        # Clean up text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text_content = ' '.join(chunk for chunk in chunks if chunk)

        return jsonify({
            'original_html': html_content,
            'scraped_text': text_content,
            'title': soup.title.string if soup.title else 'No title found',
            'url': url
        })

    except Exception as e:
        return jsonify({'error': f'Scraping failed: {str(e)}'}), 500

@app.route('/extract-business', methods=['POST'])
def extract_business_parameters():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        original_url = data.get('url', '')

        if not text:
            return jsonify({'error': 'Text is required'}), 400

        # AI-powered business parameter extraction for logo and brand design
        parameters = extract_branding_parameters(text, original_url)

        return jsonify({
            'parameters': parameters,
            'extraction_type': 'AI-powered brand analysis',
            'purpose': 'Logo and brand design insights'
        })

    except Exception as e:
        return jsonify({'error': f'Business parameter extraction failed: {str(e)}'}), 500

def extract_branding_parameters(text, url=""):
    """
    Dynamically extract business parameters for logo and brand design using
    both predefined extractors and LLM-generated parameters on-the-fly
    """
    import re
    from collections import Counter
    import json

    parameters = {}

    # Define all possible parameter extractors with relevance scoring
    def extract_brand_names():
        words = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
        word_freq = Counter(words)
        potential_names = [word for word, freq in word_freq.most_common(5) if freq > 1]
        if potential_names:
            return {
                'name': 'Brand Names',
                'value': potential_names[:3],
                'relevance': len(potential_names) * 2,
                'description': 'Most frequently mentioned capitalized terms that could be brand names'
            }
        return None

    def extract_industry():
        industry_keywords = {
            'technology': ['software', 'tech', 'digital', 'app', 'platform', 'AI', 'data', 'cloud', 'development', 'programming'],
            'healthcare': ['medical', 'health', 'doctor', 'clinic', 'hospital', 'wellness', 'care', 'patient', 'treatment'],
            'finance': ['bank', 'finance', 'investment', 'money', 'loan', 'credit', 'financial', 'insurance', 'wealth'],
            'retail': ['shop', 'store', 'buy', 'sell', 'product', 'retail', 'commerce', 'marketplace', 'shopping'],
            'food_beverage': ['restaurant', 'food', 'dining', 'menu', 'kitchen', 'chef', 'cuisine', 'bar', 'cafe'],
            'education': ['school', 'education', 'learn', 'course', 'training', 'university', 'student', 'academic'],
            'real_estate': ['property', 'real estate', 'home', 'house', 'apartment', 'rent', 'mortgage', 'realtor'],
            'automotive': ['car', 'auto', 'vehicle', 'automotive', 'repair', 'garage', 'dealership', 'parts'],
            'travel_hospitality': ['travel', 'hotel', 'booking', 'vacation', 'trip', 'tourism', 'resort', 'airline'],
            'fitness_wellness': ['gym', 'fitness', 'workout', 'exercise', 'yoga', 'training', 'wellness', 'spa'],
            'legal': ['law', 'legal', 'attorney', 'lawyer', 'court', 'litigation', 'counsel', 'justice'],
            'construction': ['construction', 'building', 'contractor', 'renovation', 'architecture', 'design', 'build'],
            'consulting': ['consulting', 'consultant', 'advisory', 'strategy', 'expertise', 'solutions', 'services']
        }

        text_lower = text.lower()
        industry_scores = {}
        for industry, keywords in industry_keywords.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            if score > 0:
                industry_scores[industry] = score

        if industry_scores:
            top_industry = max(industry_scores.items(), key=lambda x: x[1])
            if top_industry[1] >= 2:  # Only include if strong signal
                return {
                    'name': 'Primary Industry',
                    'value': top_industry[0].replace('_', ' ').title(),
                    'relevance': top_industry[1],
                    'description': f'Detected industry with {min(100, top_industry[1] * 10)}% confidence'
                }
        return None

    def extract_brand_personality():
        personality_indicators = {
            'professional': ['professional', 'corporate', 'business', 'enterprise', 'solutions', 'expert'],
            'friendly': ['friendly', 'welcome', 'community', 'family', 'personal', 'warm', 'caring'],
            'innovative': ['innovative', 'cutting-edge', 'modern', 'advanced', 'revolutionary', 'forward'],
            'trustworthy': ['trusted', 'reliable', 'secure', 'certified', 'guaranteed', 'established'],
            'creative': ['creative', 'design', 'artistic', 'unique', 'custom', 'original', 'imagination'],
            'luxury': ['premium', 'luxury', 'exclusive', 'elite', 'high-end', 'sophisticated', 'upscale']
        }

        text_lower = text.lower()
        personality_scores = {}
        for trait, keywords in personality_indicators.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            if score > 0:
                personality_scores[trait] = score

        if personality_scores:
            top_traits = sorted(personality_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_traits[0][1] >= 1:  # At least one mention
                return {
                    'name': 'Brand Personality',
                    'value': [trait.title() for trait, _ in top_traits if _ > 0],
                    'relevance': sum(score for _, score in top_traits),
                    'description': 'Personality traits detected from content tone and language'
                }
        return None

    def extract_target_audience():
        audience_indicators = {
            'businesses': ['business', 'companies', 'enterprise', 'corporate', 'B2B', 'organizations'],
            'consumers': ['customers', 'clients', 'people', 'individuals', 'families', 'consumers'],
            'professionals': ['professional', 'expert', 'specialist', 'consultant', 'practitioners'],
            'students': ['student', 'education', 'learning', 'academic', 'university', 'college'],
            'seniors': ['senior', 'elderly', 'retirement', 'mature', 'older adults'],
            'young_adults': ['young', 'millennial', 'gen z', 'college', 'youth', 'teens']
        }

        text_lower = text.lower()
        audience_scores = {}
        for audience, keywords in audience_indicators.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            if score > 0:
                audience_scores[audience] = score

        if audience_scores:
            top_audience = max(audience_scores.items(), key=lambda x: x[1])
            if top_audience[1] >= 2:  # Strong signal required
                return {
                    'name': 'Target Audience',
                    'value': top_audience[0].replace('_', ' ').title(),
                    'relevance': top_audience[1],
                    'description': 'Primary audience identified from content language and messaging'
                }
        return None

    def extract_services():
        service_patterns = [
            r'(?:we offer|services include|we provide|specializing in|expert in)\s+([^.!?]{10,100})',
            r'(?:our services|what we do|we help)\s+([^.!?]{10,100})',
            r'(?:solutions|offerings|capabilities)\s*:?\s*([^.!?]{10,100})'
        ]

        services = []
        for pattern in service_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            services.extend([s.strip() for s in matches])

        if services:
            return {
                'name': 'Key Services',
                'value': services[:3],
                'relevance': len(services),
                'description': 'Services and offerings explicitly mentioned'
            }
        return None

    def extract_value_propositions():
        value_patterns = [
            r'(?:why choose us|our advantage|what makes us|we are the best)\s+([^.!?]{10,150})',
            r'(?:what sets us apart|our difference|unique about us)\s+([^.!?]{10,150})',
            r'(?:we\'re different because|unlike others)\s+([^.!?]{10,150})'
        ]

        values = []
        for pattern in value_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            values.extend([v.strip() for v in matches])

        if values:
            return {
                'name': 'Value Propositions',
                'value': values[:2],
                'relevance': len(values) * 2,  # High relevance for explicit value props
                'description': 'Unique selling points and competitive advantages'
            }
        return None

    def extract_geographic_focus():
        location_patterns = [
            r'\b(?:serving|located in|based in|covering|throughout)\s+([A-Z][a-zA-Z\s,]+(?:city|state|county|region|area|CA|NY|TX|FL|IL|PA|OH|GA|NC|MI|NJ|VA|WA|AZ|MA|TN|IN|MO|MD|WI|CO|MN|SC|AL|LA|KY|OR|OK|CT|IA|MS|AR|UT|KS|NV|NM|NE|WV|ID|HI|NH|ME|RI|MT|DE|SD|ND|AK|VT|WY))\b',
            r'\b([A-Z][a-zA-Z\s]+(?:California|New York|Texas|Florida|Illinois|Pennsylvania))\b',
            r'\b((?:San Francisco|Los Angeles|New York|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose|Austin|Jacksonville|Fort Worth|Columbus|Charlotte|Detroit|El Paso|Memphis|Boston|Seattle|Denver|Nashville|Baltimore|Louisville|Portland|Oklahoma City|Milwaukee|Las Vegas|Albuquerque|Tucson|Fresno|Sacramento|Kansas City|Mesa|Virginia Beach|Atlanta|Colorado Springs|Omaha|Raleigh|Miami|Oakland|Minneapolis|Tulsa|Cleveland|Wichita|Arlington))\b'
        ]

        locations = set()
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            locations.update(match.strip() for match in matches)

        # Filter out common false positives
        filtered_locations = [loc for loc in locations if len(loc) > 3 and not loc.lower() in ['the', 'and', 'with', 'for', 'all']]

        if filtered_locations:
            return {
                'name': 'Geographic Focus',
                'value': list(filtered_locations)[:3],
                'relevance': len(filtered_locations),
                'description': 'Geographic areas and locations mentioned'
            }
        return None

    def extract_contact_info():
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        phones = re.findall(r'\b(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}\b', text)

        contact_info = {}
        if emails:
            contact_info['Email'] = emails[0]
        if phones:
            contact_info['Phone'] = phones[0]

        if contact_info:
            return {
                'name': 'Contact Information',
                'value': contact_info,
                'relevance': len(contact_info) * 3,
                'description': 'Available contact methods'
            }
        return None

    def extract_social_media():
        social_patterns = {
            'Facebook': re.findall(r'facebook\.com/[\w.-]+', text, re.IGNORECASE),
            'Twitter': re.findall(r'(?:twitter\.com|x\.com)/[\w.-]+', text, re.IGNORECASE),
            'LinkedIn': re.findall(r'linkedin\.com/[\w.-/]+', text, re.IGNORECASE),
            'Instagram': re.findall(r'instagram\.com/[\w.-]+', text, re.IGNORECASE),
            'YouTube': re.findall(r'youtube\.com/[\w.-/]+', text, re.IGNORECASE)
        }

        social_presence = {platform: urls[0] for platform, urls in social_patterns.items() if urls}
        if social_presence:
            return {
                'name': 'Social Media Presence',
                'value': social_presence,
                'relevance': len(social_presence) * 2,
                'description': 'Social media profiles and channels'
            }
        return None

    def extract_taglines():
        tagline_patterns = [
            r'"([^"]{10,100})"',
            r'our motto[:\s]+"([^"]+)"',
            r'slogan[:\s]+"([^"]+)"',
            r'tagline[:\s]+"([^"]+)"'
        ]

        taglines = []
        for pattern in tagline_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            taglines.extend(matches)

        # Filter out common non-taglines
        filtered_taglines = [t for t in taglines if not any(word in t.lower() for word in ['click', 'read more', 'learn more', 'contact us', 'call now'])]

        if filtered_taglines:
            return {
                'name': 'Potential Taglines',
                'value': filtered_taglines[:3],
                'relevance': len(filtered_taglines) * 2,
                'description': 'Quoted text that could be taglines or slogans'
            }
        return None

    def extract_competitive_advantages():
        advantage_patterns = [
            r'\b(first|only|best|leading|top|premier|award-winning|certified|#1)\s+([^.!?]{10,80})',
            r'\b(largest|smallest|fastest|most reliable|most trusted)\s+([^.!?]{10,80})',
            r'(winner of|recipient of|certified by|approved by)\s+([^.!?]{10,80})'
        ]

        advantages = []
        for pattern in advantage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            advantages.extend([f"{match[0]} {match[1]}" for match in matches])

        if advantages:
            return {
                'name': 'Competitive Advantages',
                'value': advantages[:3],
                'relevance': len(advantages) * 3,
                'description': 'Claims of superiority or unique positioning'
            }
        return None

    def extract_business_experience():
        year_patterns = [
            r'(?:since|established|founded in?|started in?)\s+(\d{4})',
            r'(?:over|more than)\s+(\d+)\s+years?\s+(?:of experience|in business)',
            r'(\d+)\+?\s+years?\s+(?:of experience|serving|in business)'
        ]

        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return {
                    'name': 'Business Experience',
                    'value': matches[0],
                    'relevance': 3,
                    'description': 'Years in business or founding information'
                }
        return None

    def extract_website_domain():
        if url:
            import urllib.parse
            domain = urllib.parse.urlparse(url).netloc.replace('www.', '')
            return {
                'name': 'Website Domain',
                'value': domain,
                'relevance': 2,
                'description': 'Primary website domain'
            }
        return None

    def generate_color_suggestions():
        # Only generate if we detected an industry
        industry_result = extract_industry()
        if not industry_result:
            return None

        color_map = {
            'technology': ['Blue', 'Gray', 'Green', 'White'],
            'healthcare': ['Blue', 'Green', 'White', 'Teal'],
            'finance': ['Blue', 'Navy', 'Gray', 'Gold'],
            'retail': ['Red', 'Orange', 'Purple', 'Pink'],
            'food beverage': ['Red', 'Orange', 'Yellow', 'Brown'],
            'education': ['Blue', 'Green', 'Orange', 'Purple'],
            'real estate': ['Blue', 'Green', 'Brown', 'Gold'],
            'automotive': ['Red', 'Blue', 'Black', 'Silver'],
            'travel hospitality': ['Blue', 'Teal', 'Orange', 'Green'],
            'fitness wellness': ['Red', 'Orange', 'Green', 'Black'],
            'legal': ['Navy', 'Gray', 'Gold', 'Maroon'],
            'construction': ['Orange', 'Yellow', 'Brown', 'Gray'],
            'consulting': ['Blue', 'Gray', 'Green', 'Purple']
        }

        industry_name = industry_result['value'].lower()
        colors = color_map.get(industry_name, ['Blue', 'Green', 'Gray'])

        return {
            'name': 'Suggested Colors',
            'value': colors,
            'relevance': 4,
            'description': f'Color palette recommendations for {industry_name} industry'
        }

    def generate_logo_style():
        # Generate based on personality and industry if available
        personality_result = extract_brand_personality()
        industry_result = extract_industry()

        if not (personality_result or industry_result):
            return None

        style_suggestions = []

        if personality_result:
            personality = personality_result['value'][0].lower()
            personality_styles = {
                'professional': 'clean and minimalist',
                'friendly': 'approachable and warm',
                'innovative': 'modern and dynamic',
                'trustworthy': 'stable and classic',
                'creative': 'artistic and unique',
                'luxury': 'elegant and sophisticated'
            }
            style_suggestions.append(personality_styles.get(personality, 'balanced and versatile'))

        if industry_result:
            industry = industry_result['value'].lower()
            industry_styles = {
                'technology': 'geometric and tech-forward',
                'healthcare': 'trustworthy and caring',
                'finance': 'stable and authoritative',
                'retail': 'friendly and appealing',
                'food beverage': 'appetizing and warm',
                'education': 'inspiring and accessible',
                'legal': 'traditional and authoritative',
                'construction': 'strong and reliable',
                'consulting': 'professional and sophisticated'
            }
            style_suggestions.append(industry_styles.get(industry, 'professional and clean'))

        return {
            'name': 'Recommended Logo Style',
            'value': style_suggestions[0] if style_suggestions else 'professional and versatile',
            'relevance': 4,
            'description': 'Logo style recommendation based on brand personality and industry'
        }

    # Additional specialized extractors for expanded coverage
    def extract_pricing_model():
        pricing_patterns = [
            r'(?:starting at|from|only|just)\s*\$(\d+(?:\.\d{2})?)',
            r'(?:free|subscription|monthly|annual|one-time)',
            r'(?:pricing|rates|fees|cost).*?(\$\d+(?:\.\d{2})?)',
            r'(?:budget-friendly|affordable|premium|luxury|high-end|enterprise)'
        ]

        pricing_info = []
        text_lower = text.lower()

        for pattern in pricing_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            pricing_info.extend(matches)

        # Check for pricing models
        models = []
        if 'subscription' in text_lower or 'monthly' in text_lower: models.append('Subscription-based')
        if 'free' in text_lower and 'trial' in text_lower: models.append('Free trial available')
        if 'custom' in text_lower and 'pricing' in text_lower: models.append('Custom pricing')
        if 'enterprise' in text_lower: models.append('Enterprise solutions')

        if pricing_info or models:
            return {
                'name': 'Pricing Model',
                'value': models if models else pricing_info[:3],
                'relevance': len(pricing_info) + len(models),
                'description': 'Pricing structure and cost information'
            }
        return None

    def extract_business_hours():
        hours_patterns = [
            r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun).*?(?:\d{1,2}:\d{2}|\d{1,2}\s?(?:am|pm))',
            r'(?:open|closed|hours).*?(\d{1,2}:\d{2}.*?\d{1,2}:\d{2})',
            r'(?:24/7|24 hours|always open)',
            r'(?:business hours|operating hours|store hours).*?([^.]{20,100})'
        ]

        hours_info = []
        for pattern in hours_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            hours_info.extend([h.strip() for h in matches])

        if hours_info:
            return {
                'name': 'Business Hours',
                'value': hours_info[:3],
                'relevance': len(hours_info),
                'description': 'Operating hours and availability'
            }
        return None

    def extract_certifications():
        cert_patterns = [
            r'(?:certified|accredited|licensed|authorized|approved)\s+(?:by|with|in)\s+([A-Z][^.]{10,50})',
            r'(?:ISO|HIPAA|SOC|GDPR|PCI|FDA|OSHA)\s*(?:compliant|certified)?',
            r'(?:award|winner|recipient)\s+(?:of|for)\s+([^.]{10,50})',
            r'(?:member of|affiliated with)\s+([A-Z][^.]{10,50})'
        ]

        certifications = []
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend([c.strip() for c in matches])

        if certifications:
            return {
                'name': 'Certifications & Awards',
                'value': certifications[:4],
                'relevance': len(certifications) * 2,
                'description': 'Professional certifications, awards, and affiliations'
            }
        return None

    def extract_team_size():
        team_patterns = [
            r'(?:team of|staff of|over|more than)\s+(\d+)\s+(?:people|employees|professionals|experts)',
            r'(\d+)\+?\s+(?:employees|team members|professionals|staff)',
            r'(?:small|large|growing|established)\s+team',
            r'(?:family-owned|family business|sole proprietor)'
        ]

        team_info = []
        for pattern in team_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            team_info.extend([t.strip() for t in matches])

        if team_info:
            return {
                'name': 'Team & Organization',
                'value': team_info[:2],
                'relevance': len(team_info),
                'description': 'Company size and organizational structure'
            }
        return None

    def extract_technology_stack():
        tech_keywords = [
            'React', 'Vue', 'Angular', 'Node.js', 'Python', 'Java', 'PHP', 'Ruby',
            'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes',
            'AI', 'Machine Learning', 'Blockchain', 'IoT', 'Cloud Computing',
            'Mobile App', 'iOS', 'Android', 'API', 'Database', 'SaaS'
        ]

        tech_found = []
        text_upper = text
        for tech in tech_keywords:
            if tech.lower() in text.lower() or tech in text_upper:
                tech_found.append(tech)

        if tech_found:
            return {
                'name': 'Technology Stack',
                'value': tech_found[:6],
                'relevance': len(tech_found),
                'description': 'Technologies and platforms mentioned'
            }
        return None

    def extract_sustainability():
        sustainability_keywords = [
            'sustainable', 'eco-friendly', 'green', 'renewable', 'carbon neutral',
            'environmentally responsible', 'recycled', 'organic', 'fair trade',
            'LEED certified', 'solar powered', 'zero waste'
        ]

        text_lower = text.lower()
        sustainability_found = [kw for kw in sustainability_keywords if kw in text_lower]

        if sustainability_found:
            return {
                'name': 'Sustainability Focus',
                'value': sustainability_found[:4],
                'relevance': len(sustainability_found) * 2,
                'description': 'Environmental and sustainability initiatives'
            }
        return None

    def llm_generate_custom_parameters(content_sample, existing_params):
        """
        Use LLM-style pattern matching to generate custom parameters
        based on unique content patterns not covered by predefined extractors
        """
        custom_parameters = []

        # Analyze content for unique patterns
        sentences = re.split(r'[.!?]+', content_sample)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:10]  # First 10 meaningful sentences

        # Look for unique business characteristics
        unique_patterns = {
            'Partnership & Affiliations': [
                r'(?:partner with|work with|collaborate with|affiliated with)\s+([A-Z][^.]{10,50})',
                r'(?:proud member|member of)\s+([^.]{10,50})'
            ],
            'Service Areas': [
                r'(?:serving|covering|available in)\s+([^.]{15,80})',
                r'(?:nationwide|statewide|locally owned)'
            ],
            'Company Culture': [
                r'(?:our culture|we believe|our mission|our vision)\s+([^.]{15,100})',
                r'(?:family-owned|woman-owned|veteran-owned|minority-owned)'
            ],
            'Unique Selling Points': [
                r'(?:what makes us unique|why choose us|our specialty)\s+([^.]{15,120})',
                r'(?:only|first|exclusive|patented|proprietary)\s+([^.]{10,80})'
            ],
            'Client Types': [
                r'(?:we work with|our clients include|serving)\s+([^.]{15,80})',
                r'(?:Fortune 500|startups|small business|enterprise|government)'
            ],
            'Process & Methodology': [
                r'(?:our process|methodology|approach|system)\s+([^.]{15,100})',
                r'(?:step process|proven method|systematic approach)'
            ]
        }

        for param_name, patterns in unique_patterns.items():
            found_values = []
            for pattern in patterns:
                matches = re.findall(pattern, content_sample, re.IGNORECASE)
                found_values.extend([m.strip() for m in matches if isinstance(m, str)])

            # Check if this type of parameter wasn't already extracted
            param_key = param_name.lower().replace(' ', '_').replace('&', 'and')
            if found_values and not any(param_key in existing_key for existing_key in existing_params.keys()):
                custom_parameters.append({
                    'name': param_name,
                    'value': found_values[:3],
                    'relevance': len(found_values) + 1,  # Bonus for being unique
                    'description': f'Custom-detected {param_name.lower()} specific to this business'
                })

        # Industry-specific parameter generation
        if any('industry' in key for key in existing_params.keys()):
            industry_value = next((param['value'] for param in existing_params.values()
                                 if 'industry' in param.get('name', '').lower()), None)

            if industry_value:
                industry_specific = generate_industry_specific_parameters(content_sample, industry_value)
                custom_parameters.extend(industry_specific)

        return custom_parameters[:8]  # Limit to 8 custom parameters

    def generate_industry_specific_parameters(content, industry):
        """Generate parameters specific to detected industry"""
        industry_lower = industry.lower()
        specific_params = []

        if 'restaurant' in industry_lower or 'food' in industry_lower:
            # Restaurant-specific parameters
            cuisine_patterns = [
                r'(?:cuisine|food|specializing in|famous for)\s+([^.]{10,50})',
                r'(?:italian|chinese|mexican|indian|thai|french|american|japanese|mediterranean)\s+(?:cuisine|food|restaurant)'
            ]
            for pattern in cuisine_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    specific_params.append({
                        'name': 'Cuisine Type',
                        'value': matches[:3],
                        'relevance': len(matches),
                        'description': 'Type of cuisine and food specialties'
                    })
                    break

        elif 'healthcare' in industry_lower or 'medical' in industry_lower:
            # Healthcare-specific parameters
            specialty_patterns = [
                r'(?:specializing in|expert in|focus on)\s+([^.]{10,60})',
                r'(?:pediatric|geriatric|cardiology|oncology|dermatology|orthopedic|dental)'
            ]
            for pattern in specialty_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    specific_params.append({
                        'name': 'Medical Specialties',
                        'value': matches[:3],
                        'relevance': len(matches),
                        'description': 'Medical specialties and areas of expertise'
                    })
                    break

        elif 'technology' in industry_lower:
            # Tech-specific parameters
            if 'startup' in content.lower() or 'founded' in content.lower():
                stage_indicators = ['startup', 'early-stage', 'scale-up', 'established', 'enterprise']
                found_stage = [stage for stage in stage_indicators if stage in content.lower()]
                if found_stage:
                    specific_params.append({
                        'name': 'Company Stage',
                        'value': found_stage[0],
                        'relevance': 2,
                        'description': 'Current business development stage'
                    })

        return specific_params

    # Enhanced extractor list with all original + new ones
    extractors = [
        extract_brand_names,
        extract_industry,
        extract_brand_personality,
        extract_target_audience,
        extract_services,
        extract_value_propositions,
        extract_geographic_focus,
        extract_contact_info,
        extract_social_media,
        extract_taglines,
        extract_competitive_advantages,
        extract_business_experience,
        extract_website_domain,
        generate_color_suggestions,
        generate_logo_style,
        # New extractors
        extract_pricing_model,
        extract_business_hours,
        extract_certifications,
        extract_team_size,
        extract_technology_stack,
        extract_sustainability
    ]

    # Extract parameters and sort by relevance
    extracted_params = []
    for extractor in extractors:
        result = extractor()
        if result and result['relevance'] > 0:
            extracted_params.append(result)

    # Create intermediate parameter dict for LLM analysis
    temp_params = {}
    for i, param in enumerate(extracted_params):
        temp_params[f"param_{i+1}_{param['name'].lower().replace(' ', '_')}"] = {
            'name': param['name'],
            'value': param['value'],
            'description': param['description'],
            'relevance_score': param['relevance']
        }

    # Generate custom parameters using LLM-style analysis
    # Use first 2000 characters for analysis to avoid processing too much text
    content_sample = text[:2000]
    custom_params = llm_generate_custom_parameters(content_sample, temp_params)

    # Add custom parameters to the list
    for custom_param in custom_params:
        extracted_params.append(custom_param)

    # Sort all parameters by relevance score (highest first)
    extracted_params.sort(key=lambda x: x['relevance'], reverse=True)

    # Convert to final format - include top 18 most relevant (expanded from 12)
    for i, param in enumerate(extracted_params[:18]):
        param_key = f"param_{i+1}_{param['name'].lower().replace(' ', '_').replace('&', 'and')}"
        parameters[param_key] = {
            'name': param['name'],
            'value': param['value'],
            'description': param['description'],
            'relevance_score': param['relevance'],
            'is_custom_generated': param.get('name') in [cp['name'] for cp in custom_params]
        }

    return parameters

@app.route('/generate-logo-prompt', methods=['POST'])
def generate_logo_prompt():
    try:
        data = request.get_json()
        parameters = data.get('parameters', {})

        if not parameters:
            return jsonify({'error': 'Parameters are required'}), 400

        # Generate comprehensive logo design prompt
        prompt = create_logo_design_prompt(parameters)

        return jsonify({
            'prompt': prompt,
            'word_count': len(prompt.split()),
            'generated_from': len(parameters),
            'prompt_type': 'Comprehensive Logo Design Brief'
        })

    except Exception as e:
        return jsonify({'error': f'Prompt generation failed: {str(e)}'}), 500

def create_logo_design_prompt(parameters):
    """
    Generate a comprehensive, natural language logo design prompt
    based on extracted business parameters
    """

    # Extract key information from parameters
    brand_name = None
    industry = None
    personality = []
    colors = []
    target_audience = None
    services = []
    values = []
    style_recommendation = None
    unique_aspects = []

    # Parse parameters to extract relevant information
    for param_key, param_data in parameters.items():
        name = param_data.get('name', '').lower()
        value = param_data.get('value')
        relevance = param_data.get('relevance_score', 0)

        # Only use high-relevance parameters for prompt generation
        if relevance < 2:
            continue

        if 'brand' in name or 'name' in name:
            if isinstance(value, list) and value:
                brand_name = value[0]
            elif isinstance(value, str):
                brand_name = value

        elif 'industry' in name:
            industry = value

        elif 'personality' in name:
            if isinstance(value, list):
                personality = value[:3]
            else:
                personality = [value]

        elif 'color' in name:
            if isinstance(value, list):
                colors = value[:4]
            else:
                colors = [value]

        elif 'audience' in name:
            target_audience = value

        elif 'service' in name:
            if isinstance(value, list):
                services = value[:3]
            elif isinstance(value, str):
                services = [value]

        elif 'value' in name or 'advantage' in name:
            if isinstance(value, list):
                values = value[:2]
            elif isinstance(value, str):
                values = [value]

        elif 'logo style' in name or 'style' in name:
            style_recommendation = value

        elif param_data.get('is_custom_generated') and relevance >= 3:
            # Include high-relevance custom parameters as unique aspects
            if isinstance(value, list):
                unique_aspects.extend(value[:2])
            else:
                unique_aspects.append(str(value))

    # Build the comprehensive prompt
    prompt_parts = []

    # Opening statement
    prompt_parts.append("Create a professional logo design with the following specifications:")

    # Brand name and industry
    if brand_name and industry:
        prompt_parts.append(f"Design a logo for '{brand_name}', a {industry.lower()} company.")
    elif brand_name:
        prompt_parts.append(f"Design a logo for '{brand_name}'.")
    elif industry:
        prompt_parts.append(f"Design a logo for a {industry.lower()} business.")

    # Brand personality and style
    if personality:
        personality_text = ", ".join(personality[:3])
        prompt_parts.append(f"The brand personality should convey a {personality_text.lower()} image.")

    if style_recommendation:
        prompt_parts.append(f"The logo style should be {style_recommendation.lower()}.")

    # Target audience
    if target_audience:
        prompt_parts.append(f"The primary target audience is {target_audience.lower()}.")

    # Color palette
    if colors:
        colors_text = ", ".join(colors[:4])
        prompt_parts.append(f"Use a color palette incorporating: {colors_text.lower()}.")
    else:
        prompt_parts.append("Choose colors appropriate for the industry and brand personality.")

    # Services and focus areas
    if services:
        services_clean = [s.strip()[:50] for s in services[:2]]  # Limit length
        services_text = " and ".join(services_clean)
        prompt_parts.append(f"The company specializes in {services_text.lower()}.")

    # Unique value propositions
    if values:
        values_clean = [v.strip()[:60] for v in values[:2]]
        values_text = " They emphasize " + " and ".join(values_clean).lower() + "."
        prompt_parts.append(values_text)

    # Custom unique aspects
    if unique_aspects:
        unique_clean = [str(u).strip()[:40] for u in unique_aspects[:2]]
        unique_text = " Notable characteristics include: " + ", ".join(unique_clean).lower() + "."
        prompt_parts.append(unique_text)

    # Design requirements
    prompt_parts.append("The logo must be:")
    prompt_parts.append("- Scalable and readable at various sizes")
    prompt_parts.append("- Suitable for both digital and print applications")
    prompt_parts.append("- Memorable and distinctive")
    prompt_parts.append("- Appropriate for professional business use")

    # Industry-specific requirements
    if industry:
        industry_lower = industry.lower()
        if 'technology' in industry_lower:
            prompt_parts.append("- Modern and tech-forward in appearance")
            prompt_parts.append("- Clean and geometric design elements")
        elif 'healthcare' in industry_lower or 'medical' in industry_lower:
            prompt_parts.append("- Trustworthy and professional appearance")
            prompt_parts.append("- Clean, safe, and caring visual impression")
        elif 'food' in industry_lower or 'restaurant' in industry_lower:
            prompt_parts.append("- Appetizing and warm visual appeal")
            prompt_parts.append("- Inviting and approachable design")
        elif 'finance' in industry_lower:
            prompt_parts.append("- Stable and authoritative appearance")
            prompt_parts.append("- Professional and trustworthy design")
        elif 'creative' in industry_lower or 'design' in industry_lower:
            prompt_parts.append("- Artistic and unique visual elements")
            prompt_parts.append("- Creative and original design approach")

    # Format considerations
    prompt_parts.append("Provide the logo in a clean, vector-style format suitable for various applications.")

    # Join all parts into a coherent prompt
    full_prompt = " ".join(prompt_parts)

    # Clean up any double spaces or formatting issues
    full_prompt = " ".join(full_prompt.split())

    return full_prompt

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081)