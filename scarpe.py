import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time
import re # Added for more flexible element finding and ID extraction

def scrape_tds_course_content(url):
    """
    Scrapes content from the TDS course website.
    Args:
        url (str): The URL of the course content page.
    Returns:
        list: A list of dictionaries, each containing 'url' and 'content'.
    """
    scraped_data = []
    print(f"Attempting to scrape TDS course content from: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Target the main content area based on common HTML structures
        # On inspection, content seems to be within <main> and then <div class="section-content">
        main_content = soup.find('main')
        if main_content:
            content_elements = main_content.find_all('div', class_='section-content')
            if not content_elements: # Fallback if specific class not found, get general text elements
                content_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'li', 'code'])

            page_content_parts = []
            for element in content_elements:
                text = element.get_text(separator='\n', strip=True)
                if text:
                    page_content_parts.append(text)

            full_content = "\n\n".join(page_content_parts)
            if full_content:
                scraped_data.append({
                    "url": url,
                    "content": full_content
                })
            else:
                print(f"No significant content found within main_content on {url}")
        else:
            print(f"Warning: Could not find <main> content area on {url}")

    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while scraping {url}: {e}")
    return scraped_data

def scrape_discourse_posts(base_url, category_url, num_topics_to_scrape=5):
    """
    Scrapes recent topics and their posts from a Discourse category page,
    attempting to extract detailed metadata for each post.
    Args:
        base_url (str): The base URL of the Discourse forum.
        category_url (str): The URL of the category page.
        num_topics_to_scrape (int): The maximum number of topics to scrape for demonstration.
    Returns:
        list: A list of dictionaries, each representing a post with 'url', 'content', and metadata.
    """
    scraped_data = []
    print(f"Attempting to scrape Discourse posts from category: {category_url}")
    try:
        # Step 1: Get topic links from the category page
        response = requests.get(category_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        topic_links = []
        topic_list_table = soup.find('table', class_='topic-list') # Find the table listing topics
        if topic_list_table:
            # Find all topic title links
            for link_tag in topic_list_table.find_all('a', class_='title raw-link raw-topic-link'):
                href = link_tag.get('href')
                if href and href.startswith('/t/'): # Ensure it's a topic link
                    full_url = urljoin(base_url, href)
                    topic_links.append(full_url)
        else:
            print(f"Warning: Could not find topic list table on {category_url}")

        print(f"Found {len(topic_links)} topics. Scraping content from up to {num_topics_to_scrape} for demonstration.")

        # Limit scraping to a few topics to avoid excessive requests for demonstration
        for i, topic_url in enumerate(topic_links[:num_topics_to_scrape]):
            print(f"  Scraping topic ({i+1}/{num_topics_to_scrape}): {topic_url}")
            try:
                topic_response = requests.get(topic_url)
                topic_response.raise_for_status()
                topic_soup = BeautifulSoup(topic_response.text, 'html.parser')

                # Extract topic_id from the body tag or a known element
                topic_id = None
                body_tag = topic_soup.find('body')
                if body_tag and 'data-topic-id' in body_tag.attrs:
                    topic_id = int(body_tag['data-topic-id'])
                elif topic_url:
                    # Fallback: try to extract from URL if not found in data-attributes
                    match = re.search(r'/t/[^/]+/(\d+)', topic_url)
                    if match:
                        topic_id = int(match.group(1))

                # Iterate through individual posts within the topic
                # Each post is typically within an <article> tag or <div> with classes like 'topic-post'
                post_elements = topic_soup.find_all('div', class_=re.compile(r'topic-post|post-stream'))

                for post_elem in post_elements:
                    post_data = {}
                    
                    # Extract post ID (data-post-id attribute)
                    if 'data-post-id' in post_elem.attrs:
                        post_data['id'] = int(post_elem['data-post-id'])
                    
                    # Assign topic_id to each post
                    post_data['topic_id'] = topic_id
                    
                    # Construct URL for this specific post if post ID is available
                    post_data['url'] = f"{topic_url}/{post_data.get('id', '')}" if 'id' in post_data else topic_url
                    
                    # Extract username
                    username_tag = post_elem.find('span', class_='username')
                    if username_tag:
                        post_data['username'] = username_tag.get_text(strip=True)
                    
                    # Extract post content from the 'cooked' div
                    cooked_content = post_elem.find('div', class_='cooked')
                    if cooked_content:
                        post_data['content'] = cooked_content.get_text(separator='\n', strip=True)
                    
                    # Extract created_at timestamp
                    time_tag = post_elem.find('time', class_='post-time')
                    if time_tag and 'datetime' in time_tag.attrs:
                        post_data['created_at'] = time_tag['datetime']
                    
                    # Add to scraped data if content is found
                    if post_data.get('content'):
                        scraped_data.append(post_data)

            except requests.exceptions.RequestException as e:
                print(f"  Error scraping topic {topic_url}: {e}")
            except Exception as e:
                print(f"  An unexpected error occurred while scraping topic {topic_url}: {e}")
            
            time.sleep(1) # Be polite and avoid overwhelming the server

    except requests.exceptions.RequestException as e:
        print(f"Error scraping Discourse category {category_url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while scraping Discourse {category_url}: {e}")
    return scraped_data

if __name__ == "__main__":
    tds_course_url = "https://tds.s-anand.net/#/2025-01/"
    discourse_base_url = "https://discourse.onlinedegree.iitm.ac.in/"
    discourse_category_url = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34"

    # Scrape TDS Course Content
    course_data = scrape_tds_course_content(tds_course_url)
    print("\n--- Sample Scraped TDS Course Content ---")
    for item in course_data:
        print(f"URL: {item['url']}")
        print(f"Content snippet (first 500 chars): {item['content'][:500]}...\n")
    if not course_data:
        print("No course data was scraped.")


    # Scrape Discourse Posts
    # Set num_topics_to_scrape to a small number for quick testing (e.g., 3 topics)
    discourse_data = scrape_discourse_posts(discourse_base_url, discourse_category_url, num_topics_to_scrape=3)
    print("\n--- Sample Scraped Discourse Posts ---")
    for item in discourse_data:
        print(f"URL: {item.get('url', 'N/A')}")
        print(f"ID: {item.get('id', 'N/A')}, Topic ID: {item.get('topic_id', 'N/A')}")
        print(f"Username: {item.get('username', 'N/A')}, Created At: {item.get('created_at', 'N/A')}")
        print(f"Content snippet (first 500 chars): {item.get('content', '')[:500]}...\n")
    if not discourse_data:
        print("No Discourse data was scraped.")

    # Combine all scraped data
    all_scraped_data = course_data + discourse_data

    # Save to a JSON file for further processing
    # This JSON file would contain the raw text content from the scraped pages.
    output_filename = "raw_scraped_data_with_metadata.json"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(all_scraped_data, f, ensure_ascii=False, indent=2)
        print(f"\nTotal {len(all_scraped_data)} items scraped and saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving scraped data to JSON: {e}")