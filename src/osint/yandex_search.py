"""
OSINT Module - Automated Yandex Reverse Image Search
Searches for person identity using facial image
"""

import os
import time
import logging
from typing import Optional, Dict, List
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YandexImageSearch:
    """Automated reverse image search using Yandex"""

    def __init__(self, headless: bool = True):
        """
        Initialize the search engine

        Args:
            headless: If True, browser runs invisibly in background
        """
        self.headless = headless
        self.driver = None

    def _setup_driver(self):
        """Setup Chrome driver with options"""
        options = Options()

        if self.headless:
            options.add_argument('--headless=new')
            options.add_argument('--disable-gpu')

        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        options.add_argument(
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        options.add_argument('--lang=en-US')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)

    def _close_driver(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def search_by_image(self, image_path: str) -> Dict:
        """
        Search for person information using reverse image search

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with found information
        """
        result = {
            'success': False,
            'name': None,
            'description': None,
            'links': [],
            'social_profiles': [],
            'similar_images': 0,
            'raw_text': None,
            'error': None
        }

        if not os.path.exists(image_path):
            result['error'] = f"Image not found: {image_path}"
            return result

        try:
            logger.info(f"🔍 Starting OSINT search for: {image_path}")
            self._setup_driver()

            # Go to Yandex Images
            logger.info("📡 Connecting to Yandex Images...")
            self.driver.get("https://yandex.com/images/")
            time.sleep(2)

            # Find and click the camera icon (search by image)
            logger.info("📷 Looking for image upload button...")

            try:
                # Try to find the camera/image search button
                camera_btn = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, "button.input__cbir-button, .input__icon_cbir, [class*='cbir']"))
                )
                camera_btn.click()
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Camera button not found, trying alternative: {e}")
                # Alternative: go directly to reverse image search URL
                self.driver.get("https://yandex.com/images/search?rpt=imageview")
                time.sleep(2)

            # Upload image
            logger.info("📤 Uploading image...")

            try:
                # Find file input
                file_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
                )

                # Send absolute path
                abs_path = os.path.abspath(image_path)
                file_input.send_keys(abs_path)

                logger.info("⏳ Waiting for results...")
                time.sleep(5)

            except Exception as e:
                logger.error(f"Failed to upload image: {e}")
                result['error'] = f"Upload failed: {str(e)}"
                return result

            # Extract results
            logger.info("📊 Extracting results...")

            # Get page text for analysis
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            result['raw_text'] = page_text[:2000]  # First 2000 chars

            # Try to find person name/description
            try:
                # Look for "similar images" section title or description
                selectors = [
                    ".CbirObjectResponse-Title",
                    ".CbirItem-Title",
                    ".cbir-object-title",
                    ".Tags-Title",
                    "[class*='title']",
                    ".MMViewerButtons-TextContainer"
                ]

                for selector in selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for elem in elements:
                            text = elem.text.strip()
                            if text and len(text) > 2 and len(text) < 100:
                                if not result['name']:
                                    result['name'] = text
                                else:
                                    result['description'] = text
                                break
                    except:
                        continue

            except Exception as e:
                logger.warning(f"Could not extract name: {e}")

            # Try to find links
            try:
                links = self.driver.find_elements(By.CSS_SELECTOR,
                                                  "a[href*='instagram'], a[href*='twitter'], a[href*='facebook'], a[href*='linkedin'], a[href*='wikipedia']")

                social_domains = ['instagram.com', 'twitter.com', 'facebook.com', 'linkedin.com', 'tiktok.com']

                for link in links[:10]:  # Limit to 10 links
                    href = link.get_attribute('href')
                    if href:
                        result['links'].append(href)

                        # Identify social profiles
                        for domain in social_domains:
                            if domain in href:
                                result['social_profiles'].append({
                                    'platform': domain.split('.')[0].title(),
                                    'url': href
                                })
                                break

            except Exception as e:
                logger.warning(f"Could not extract links: {e}")

            # Count similar images
            try:
                similar_section = self.driver.find_elements(By.CSS_SELECTOR,
                                                            ".CbirSites-Item, .serp-item, [class*='similar']")
                result['similar_images'] = len(similar_section)
            except:
                pass

            # If we found anything, mark as success
            if result['name'] or result['links'] or result['similar_images'] > 0:
                result['success'] = True
                logger.info(f"✅ Search completed! Found: {result['name'] or 'Unknown'}")
            else:
                logger.warning("⚠️ No significant results found")
                result['success'] = True  # Search worked, just no results

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            result['error'] = str(e)

        finally:
            self._close_driver()

        return result

    def search_multiple(self, image_paths: List[str]) -> List[Dict]:
        """Search for multiple images"""
        results = []
        for path in image_paths:
            result = self.search_by_image(path)
            results.append({
                'image': path,
                'result': result
            })
        return results


# Standalone function for easy import
def search_person(image_path: str, headless: bool = True) -> Dict:
    """
    Quick function to search for a person by image

    Args:
        image_path: Path to image file
        headless: Run browser invisibly

    Returns:
        Dictionary with search results
    """
    searcher = YandexImageSearch(headless=headless)
    return searcher.search_by_image(image_path)


# Test function
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image = sys.argv[1]
    else:
        image = "data/input/test.jpg"

    print(f"\n🔍 Testing OSINT search on: {image}\n")

    result = search_person(image, headless=False)  # Show browser for testing

    print("\n" + "=" * 50)
    print("📊 RESULTS:")
    print("=" * 50)
    print(f"Success: {result['success']}")
    print(f"Name: {result['name']}")
    print(f"Description: {result['description']}")
    print(f"Social Profiles: {result['social_profiles']}")
    print(f"Links found: {len(result['links'])}")
    print(f"Similar images: {result['similar_images']}")
    if result['error']:
        print(f"Error: {result['error']}")