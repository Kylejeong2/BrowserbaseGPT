import os
import json
import time
from scrapegraphai.graphs import SmartScraperGraph
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found in environment variables.")

    # Define the scraping configuration
    graph_config = {
        "llm": {
            "api_key": OPENAI_API_KEY,
            "model": "openai/gpt-4o-mini",
            "temperature": 0.7,
        },
        "verbose": True,
        "headless": True,
    }

    # Read URLs from docs_urls.txt
    urls_file = "scraping_browserbase_scripts/docs_urls.txt"
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    # Initialize a list to hold scraped data
    scraped_data = []

    # Iterate through each URL and scrape content
    for url in urls:
        print(f"Scraping URL: {url}")
        prompt = "Extract all relevant content from this documentation page."
        
        try:
            # Initialize and run the SmartScraperGraph
            scraper = SmartScraperGraph(
                prompt=prompt,
                source=url,
                config=graph_config
            )
            
            result = scraper.run()
            
            scraped_data.append({
                "url": url,
                "content": result
            })
            
            print(f"Successfully scraped: {url}")
        
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            continue
        
        # Optional: Add a delay between requests
        time.sleep(1)

    # Save the scraped data
    output_file = "scraping_browserbase_scripts/scraped_docs.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scraped_data, f, ensure_ascii=False, indent=2)
    
    print(f"Scraped data has been saved to {output_file}")

if __name__ == "__main__":
    main()
