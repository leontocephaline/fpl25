from __future__ import annotations

import json
import logging
import time
from typing import Optional, Dict, Any

import keyring
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)

SERVICE_NAME = "fpl-weekly-updater"
LOGIN_URL = (
    "https://users.premierleague.com/accounts/login/"
    "?app=plfpl&redirect_uri=https%3A%2F%2Ffantasy.premierleague.com%2F"
)
MY_TEAM_URL = "https://fantasy.premierleague.com/my-team"
AUTHORIZE_URL = (
    "https://account.premierleague.com/as/authorize"
    "?response_type=code&client_id=1f243d70-a140-4035-8c41-341f5af5aa12"
    "&redirect_uri=https%3A%2F%2Ffantasy.premierleague.com%2F"
    "&scope=openid%20profile%20offline_access%20p1:update:user%20p1:read:device%20p1:reset:userPassword&onboarding="
)


def _extract_team_data(driver: webdriver.Chrome) -> Optional[Dict[str, Any]]:
    """Extract team data from the my-team page HTML."""
    try:
        # Wait for the team data to be present in the page source
        WebDriverWait(driver, 15).until(
            lambda d: 'elementSummary' in d.page_source or 'picks' in d.page_source
        )
        
        # Extract the JSON data from the page source
        script_content = driver.execute_script(
            """
            for (const script of document.scripts) {
                const content = script.textContent || '';
                if (content.includes('elementSummary') || content.includes('picks')) {
                    return content;
                }
            }
            return '';
            """
        )
        
        if not script_content:
            logger.error("Could not find team data in page source")
            return None
            
        # Find the JSON data in the script content
        start = script_content.find('{')
        end = script_content.rfind('}') + 1
        
        if start == -1 or end == 0:
            logger.error("Could not parse team data from script")
            return None
            
        json_str = script_content[start:end]
        team_data = json.loads(json_str)
        
        # Save the raw data for debugging
        with open('team_data.json', 'w') as f:
            json.dump(team_data, f, indent=2)
            
        logger.info("Successfully extracted team data from page")
        return team_data
        
    except Exception as e:
        logger.error(f"Error extracting team data: {e}")
        # Save page source for debugging
        try:
            with open('error_page.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            logger.info("Saved error page source to error_page.html")
        except Exception as save_error:
            logger.error(f"Failed to save error page: {save_error}")
        return None


def login_and_get_cookie(email: str, headless: bool = True, browser: str = "edge", team_id: Optional[int] = None) -> Optional[str]:
    """
    Perform a headless browser login and return a Cookie header string
    suitable for authenticated API requests. Password is read from keyring.
    """
    password = None
    try:
        password = keyring.get_password(SERVICE_NAME, email)
    except Exception:
        logger.exception("Failed to read password from keyring")
        return None
    if not password:
        logger.error("No password in keyring for %s", email)
        return None

    driver = None
    b = (browser or "edge").lower()
    if b in ("edge",):
        options = EdgeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1200,800")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--lang=en-GB")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
        try:
            driver = webdriver.Edge(options=options)
        except Exception:
            logger.exception("Failed to start Edge WebDriver")
            return None
    elif b in ("chrome", "chromium"):
        options = ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1200,800")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--lang=en-GB")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
        options.add_experimental_option("excludeSwitches", ["enable-automation"]) 
        options.add_experimental_option('useAutomationExtension', False)
        try:
            driver = webdriver.Chrome(options=options)
        except Exception:
            logger.exception("Failed to start Chrome/Chromium WebDriver")
            return None
    elif b in ("firefox",):
        options = FirefoxOptions()
        options.headless = bool(headless)
        try:
            driver = webdriver.Firefox(options=options)
        except Exception:
            logger.exception("Failed to start Firefox WebDriver")
            return None
    else:
        logger.error("Unsupported browser: %s", browser)
        return None

    driver.set_page_load_timeout(60)
    try:
        driver.get(LOGIN_URL)
        wait = WebDriverWait(driver, 30)
        # If we hit a holding page, route via fantasy homepage and click Sign in
        try:
            cur = driver.current_url
            if cur and ("holding" in cur or "holding.html" in cur):
                logger.info("Detected holding page; routing via fantasy homepage")
                driver.get("https://fantasy.premierleague.com/")
                try:
                    driver.set_window_size(1400, 1000)
                except Exception:
                    try:
                        driver.maximize_window()
                    except Exception:
                        pass
                # Accept cookies on fantasy site if needed
                try:
                    consent_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept')]")))
                    consent_btn.click()
                except Exception:
                    pass
                # Ensure header rendered
                try:
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, "header")))
                except Exception:
                    pass
                # Aggressive cookie/consent clear on fantasy homepage before clicking
                try:
                    for selector in [
                        (By.CSS_SELECTOR, "button[aria-label='Accept All']"),
                        (By.XPATH, "//button[contains(., 'Accept all') or contains(., 'Accept All') or contains(., 'Accept all cookies') or contains(., 'Accept all Cookies') or contains(., 'Accept')]"),
                        (By.CSS_SELECTOR, "button#onetrust-accept-btn-handler"),
                    ]:
                        try:
                            btn = WebDriverWait(driver, 3).until(EC.element_to_be_clickable(selector))
                            if btn:
                                btn.click()
                                time.sleep(0.2)
                        except Exception:
                            continue
                except Exception:
                    pass
                # Retry loop to click Log in / Sign in as the app hydrates
                sign_in = None
                sign_in_selectors = [
                    (By.XPATH, "//a[normalize-space(text())='Log in' or normalize-space(text())='Sign in' or contains(., 'Log in') or contains(., 'Sign in')]") ,
                    (By.CSS_SELECTOR, "a[href*='account.premierleague.com'][href*='/as/authorize']"),
                    (By.CSS_SELECTOR, "a[href*='accounts.premierleague.com'][href*='/as/authorize']"),
                    (By.CSS_SELECTOR, "button[data-testid='login-button']"),
                    (By.XPATH, "//button[normalize-space(text())='Log in' or normalize-space(text())='Sign in' or contains(., 'Log in') or contains(., 'Sign in')]") ,
                    (By.LINK_TEXT, "Log in"),
                    (By.PARTIAL_LINK_TEXT, "Log in"),
                    (By.LINK_TEXT, "Sign in"),
                    (By.PARTIAL_LINK_TEXT, "Sign in"),
                ]
                for _ in range(6):
                    clicked_any = False
                    for sel in sign_in_selectors:
                        try:
                            sign_in = WebDriverWait(driver, 3).until(EC.element_to_be_clickable(sel))
                            if sign_in:
                                try:
                                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", sign_in)
                                except Exception:
                                    pass
                                try:
                                    sign_in.click()
                                    clicked_any = True
                                except Exception:
                                    try:
                                        driver.execute_script("arguments[0].click();", sign_in)
                                        clicked_any = True
                                    except Exception:
                                        continue
                                try:
                                    WebDriverWait(driver, 4).until(EC.url_contains("account.premierleague.com/as/authorize"))
                                    clicked_any = True
                                    break
                                except Exception:
                                    pass
                        except Exception:
                            continue
                    # JS fallback if not yet navigated
                    if "account.premierleague.com/as/authorize" not in driver.current_url:
                        try:
                            js = r"""
                            function visible(el) {
                              const rect = el.getBoundingClientRect();
                              const style = window.getComputedStyle(el);
                              return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
                            }
                            function closestClickable(el) {
                              let cur = el;
                              while (cur && cur !== document.body) {
                                const role = cur.getAttribute && cur.getAttribute('role');
                                if ((cur.tagName === 'A') || (cur.tagName === 'BUTTON') || (role && role.toLowerCase() === 'button') || typeof cur.onclick === 'function') return cur;
                                cur = cur.parentElement;
                              }
                              return el;
                            }
                            function collectNodes(root, out) {
                              const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
                              let n;
                              while ((n = walker.nextNode())) {
                                out.push(n);
                                if (n.shadowRoot) collectNodes(n.shadowRoot, out);
                              }
                            }
                            const all = [];
                            collectNodes(document.body, all);
                            const re = /(log\s*in|sign\s*in)/i;
                            const candidates = [];
                            for (const node of all) {
                              if (!visible(node)) continue;
                              const txt = (node.innerText || node.textContent || '').trim();
                              if (re.test(txt)) candidates.push(node);
                            }
                            if (!candidates.length) return false;
                            const target = closestClickable(candidates[0]);
                            target.scrollIntoView({block:'center'});
                            const evt = new MouseEvent('click', {bubbles: true, cancelable: true, view: window});
                            target.dispatchEvent(evt);
                            return true;
                            """
                            clicked = driver.execute_script(js)
                            if clicked:
                                try:
                                    WebDriverWait(driver, 8).until(EC.url_contains("account.premierleague.com/as/authorize"))
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    # Break if navigated
                    if "account.premierleague.com/as/authorize" in driver.current_url:
                        break
                    time.sleep(1)
                # Small wait for redirect
                time.sleep(2)
        except Exception:
            pass
        # Handle cookie/consent banner if present
        try:
            # Try common selectors/text for consent buttons
            consent_btn = None
            for selector in [
                (By.CSS_SELECTOR, "button[aria-label='Accept All']"),
                (By.XPATH, "//button[contains(., 'Accept all') or contains(., 'Accept All')]") ,
                (By.XPATH, "//button[contains(., 'Accept all cookies') or contains(., 'Accept all Cookies')]") ,
                (By.CSS_SELECTOR, "button#onetrust-accept-btn-handler"),
                (By.XPATH, "//button[contains(., 'Accept')]") ,
            ]:
                try:
                    consent_btn = wait.until(EC.element_to_be_clickable(selector))
                    if consent_btn:
                        consent_btn.click()
                        break
                except Exception:
                    pass
        except Exception:
            pass
        # Wait for login form to be present with a longer timeout
        logger.info("Waiting for login form to load...")
        try:
            # First try to navigate to the authorize URL directly if not already there
            if "account.premierleague.com/as/authorize" not in driver.current_url:
                logger.info("Navigating directly to authorize URL")
                driver.get(AUTHORIZE_URL)
                time.sleep(2)  # Small wait for redirect

            # Wait for either email or username field (some forms use different names)
            email_input = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 
                    "input[type='email'], input[name='email'], input[name='username'], #email-address, #username, [data-testid='email-input']"))
            )
            
            # Clear and fill email
            email_input.clear()
            email_input.send_keys(email)
            
            # Find and fill password field (allow for different selectors)
            password_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR,
                    "input[type='password'], #password, input[name='password'], [data-testid='password-input']"))
            )
            password_input.clear()
            password_input.send_keys(password)
            
            # Find and click submit button (try multiple selectors and methods)
            submit_clicked = False
            submit_selectors = [
                "button[type='submit']",
                "button[data-testid='login-submit']",
                "button:contains('Sign in')",
                "input[type='submit']",
                "//button[contains(., 'Sign in') or contains(., 'Log in')]"
            ]
            
            # Try direct click first
            for selector in submit_selectors:
                try:
                    submit = driver.find_element(By.CSS_SELECTOR if not selector.startswith('//') else By.XPATH, selector)
                    if submit.is_displayed() and submit.is_enabled():
                        driver.execute_script("arguments[0].scrollIntoView(true);", submit)
                        time.sleep(0.5)  # Small delay for any animations
                        submit.click()
                        submit_clicked = True
                        logger.info("Clicked submit button via Selenium")
                        break
                except Exception as e:
                    continue
            
            # If direct click didn't work, try JavaScript click
            if not submit_clicked:
                logger.info("Trying JavaScript click...")
                try:
                    driver.execute_script("document.querySelector('form').submit()")
                    submit_clicked = True
                    logger.info("Submitted form via JavaScript")
                except Exception as e:
                    logger.warning(f"Form submit via JS failed: {e}")
            
            # Wait for login to complete or redirect (with longer timeout)
            logger.info("Waiting for login to complete...")
            try:
                # Wait for either my-team page or pick team page
                WebDriverWait(driver, 20).until(
                    lambda d: 'my-team' in d.current_url 
                    or 'fantasy.premierleague.com' in d.current_url
                    or 'pick-team' in d.current_url
                )
                
                # Check if we're on the pick team page
                if 'pick-team' in driver.current_url:
                    logger.info("On pick team page, clicking to proceed...")
                    try:
                        # Try to find and click the 'Pick Team' button
                        pick_team_btn = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((
                                By.XPATH, 
                                "//button[contains(., 'Pick Team') or contains(., 'PICK TEAM')]"
                            ))
                        )
                        pick_team_btn.click()
                        logger.info("Clicked Pick Team button")
                        
                        # Wait for the my-team page to load
                        WebDriverWait(driver, 15).until(
                            lambda d: 'my-team' in d.current_url
                        )
                    except Exception as e:
                        logger.warning(f"Could not click Pick Team button: {e}")
                        # Try to navigate directly to my-team
                        driver.get(MY_TEAM_URL)
                
                logger.info("Successfully logged in and on my-team page")
            except Exception as e:
                logger.warning(f"Timed out waiting for login redirect: {e}")
                # Take screenshot for debugging
                try:
                    driver.save_screenshot("login_redirect_error.png")
                    logger.info("Saved screenshot as login_redirect_error.png")
                except Exception as screenshot_error:
                    logger.error(f"Failed to save screenshot: {screenshot_error}")
                
        except Exception as e:
            logger.error(f"Error during login form submission: {e}")
            # Take screenshot for debugging
            try:
                driver.save_screenshot("login_error.png")
                logger.info("Saved screenshot as login_error.png")
            except Exception as screenshot_error:
                logger.error(f"Failed to save screenshot: {screenshot_error}")
            return None

        # Wait for redirect to fantasy site or any indication of logged in state
        try:
            wait.until(EC.url_contains("fantasy.premierleague.com"))
        except Exception:
            pass

        # If the fantasy site shows another 'Log in' button in the header, click it
        try:
            second_login = None
            for sel in [
                (By.XPATH, "//a[normalize-space(.)='Log in' or contains(., 'Log in')]"),
                (By.XPATH, "//button[normalize-space(.)='Log in' or contains(., 'Log in')]"),
                (By.CSS_SELECTOR, "button[data-testid='login-button']"),
            ]:
                try:
                    second_login = WebDriverWait(driver, 4).until(EC.element_to_be_clickable(sel))
                    if second_login:
                        try:
                            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", second_login)
                        except Exception:
                            pass
                        try:
                            second_login.click()
                        except Exception:
                            try:
                                driver.execute_script("arguments[0].click();", second_login)
                            except Exception:
                                continue
                        # Give time to navigate to auth/return
                        WebDriverWait(driver, 8).until(EC.url_contains("account.premierleague.com"))
                        WebDriverWait(driver, 20).until(EC.url_contains("fantasy.premierleague.com"))
                        break
                except Exception:
                    continue
        except Exception:
            pass

        # Try to navigate via visible UI: click 'Pick Team' or a direct link to my-team
        try:
            # Common targets that lead to my-team
            ui_targets = [
                (By.XPATH, "//a[contains(@href, '/my-team')]|//a[contains(@href, 'my-team')]"),
                (By.XPATH, "//a[contains(@href, '/pick-team')]|//a[contains(@href, 'pick-team')]"),
                (By.XPATH, "//button[contains(., 'Pick Team') or contains(., 'My Team')]"),
                (By.XPATH, "//span[normalize-space(.)='Pick Team' or normalize-space(.)='My Team']/ancestor::a|//span[normalize-space(.)='Pick Team' or normalize-space(.)='My Team']/ancestor::button"),
            ]
            clicked_nav = False
            for sel in ui_targets:
                try:
                    el = WebDriverWait(driver, 4).until(EC.element_to_be_clickable(sel))
                    if el:
                        try:
                            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                        except Exception:
                            pass
                        try:
                            el.click()
                            clicked_nav = True
                        except Exception:
                            try:
                                driver.execute_script("arguments[0].click();", el)
                                clicked_nav = True
                            except Exception:
                                continue
                        if clicked_nav:
                            break
                except Exception:
                    continue
            if clicked_nav:
                # Wait for SPA navigation
                WebDriverWait(driver, 15).until(lambda d: 'my-team' in d.current_url or 'pick-team' in d.current_url)
        except Exception:
            pass

        # Navigate to My Team page and extract data directly from HTML
        if ('my-team' not in driver.current_url) and ('pick-team' not in driver.current_url):
            logger.info("Navigating to my-team page...")
            driver.get(MY_TEAM_URL)
        
        # Wait for the page URL to be correct (accept my-team or pick-team)
        try:
            WebDriverWait(driver, 20).until(
                lambda d: ('my-team' in d.current_url) or ('pick-team' in d.current_url)
            )
        except Exception as e:
            logger.error(f"Timed out waiting for my-team/pick-team URL: {e}")
            try:
                with open('my_team_error.html', 'w', encoding='utf-8') as f:
                    f.write(driver.page_source)
                logger.info("Saved error page source to my_team_error.html")
            except Exception as save_error:
                logger.error(f"Failed to save error page: {save_error}")
            return None

        # Fetch team data via in-page API call to ensure authenticated request
        if team_id:
            try:
                logger.info("Fetching team data via in-page API for team_id=%s", team_id)
                js = r"""
                const callback = arguments[arguments.length - 1];
                (async () => {
                  try {
                    const resp = await fetch(`/api/my-team/${%TEAM_ID%}/`, { credentials: 'include' });
                    if (!resp.ok) {
                      callback({ ok: false, status: resp.status, text: await resp.text() });
                      return;
                    }
                    const data = await resp.json();
                    callback({ ok: true, data });
                  } catch (e) {
                    callback({ ok: false, error: String(e) });
                  }
                })();
                """.replace('%TEAM_ID%', str(team_id))
                result = driver.execute_async_script(js)
                if result and result.get('ok') and result.get('data'):
                    data = result['data']
                    try:
                        with open('team_data.json', 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2)
                        logger.info("Saved team data to team_data.json")
                    except Exception:
                        logger.exception("Failed to save team_data.json")
                    return "DATA_EXTRACTED"
                else:
                    logger.warning(f"In-page API fetch failed: {result}")
            except Exception:
                logger.exception("Error executing in-page API fetch for team data")

        # Fallback: try old script-based extraction (may not be present on new SPA)
        team_data = _extract_team_data(driver)
        if not team_data:
            logger.error("Failed to extract team data from page")
            return None
        return "DATA_EXTRACTED"

        # Navigate explicitly to My Team to ensure session
        # driver.get(MY_TEAM_URL)
        # # Wait/poll for auth cookies to appear
        # for _ in range(10):
        #     time.sleep(1)
        #     names = [c.get("name") for c in (driver.get_cookies() or [])]
        #     if any(n and ("sess" in n.lower() or "csrf" in n.lower()) for n in names):
        #         break
        # try:
        #     logger.info("Browser at URL: %s", driver.current_url)
        # except Exception:
        #     pass

        # Collect cookies from multiple domains and paths
        cookies_map = {}
        
        # First, collect all cookies from the current domain
        for c in driver.get_cookies() or []:
            if c.get('name') and c.get('value'):
                cookies_map[c['name']] = c['value']
        
        # Try to collect from specific domains and paths
        domains_paths = [
            ("fantasy.premierleague.com", "/"),
            ("fantasy.premierleague.com", "/api"),
            ("www.premierleague.com", "/"),
            ("account.premierleague.com", "/"),
            ("users.premierleague.com", "/")
        ]
        
        for domain, path in domains_paths:
            try:
                # Use JavaScript to set a cookie that will help us identify the current domain
                driver.execute_script(f"document.cookie = 'test_cookie=1; domain=.{domain}; path={path}';")
                
                # Navigate to a URL that will set the domain cookies
                url = f"https://{domain}{path}"
                driver.get(url)
                time.sleep(1)  # Small delay for cookies to be set
                
                # Get cookies for this domain
                for c in driver.get_cookies() or []:
                    if c.get('name') and c.get('value'):
                        cookies_map[c['name']] = c['value']
                        
            except Exception as e:
                logger.debug(f"Could not collect cookies from {domain}{path}: {e}")
        
        # Log the names of all cookies we've collected
        logger.info(f"Collected {len(cookies_map)} cookies: {', '.join(sorted(cookies_map.keys()))}")
        
        # Ensure we have the necessary cookies for the API
        required_cookie_prefixes = ['pl_', 'session', 'csrftoken', 'fpl_']
        has_required = any(
            any(cookie_name.startswith(prefix) for prefix in required_cookie_prefixes)
            for cookie_name in cookies_map.keys()
        )
        
        if not has_required:
            logger.warning("No session cookies detected in collected cookies")
            
            # Try one last time to get cookies from the my-team page
            try:
                driver.get(MY_TEAM_URL)
                time.sleep(2)
                for c in driver.get_cookies() or []:
                    if c.get('name') and c.get('value'):
                        cookies_map[c['name']] = c['value']
                
                logger.info(f"After final attempt, have {len(cookies_map)} cookies: {', '.join(sorted(cookies_map.keys()))}")
            except Exception as e:
                logger.error(f"Final cookie collection failed: {e}")
        
        # If we still don't have the right cookies, log the page source for debugging
        if not has_required:
            try:
                page_source = driver.page_source
                with open("login_page.html", "w", encoding="utf-8") as f:
                    f.write(page_source)
                logger.info("Saved page source to login_page.html for debugging")
            except Exception as e:
                logger.error(f"Could not save page source: {e}")

        if not cookies_map:
            logger.error("No cookies captured after login.")
            return None
        cookie_header = "; ".join([f"{k}={v}" for k, v in cookies_map.items()])
        try:
            logger.info("Captured cookies: %s", ", ".join(sorted(cookies_map.keys())))
        except Exception:
            pass
        # Do not hard-require a specific cookie name; let API attempt with what we have
        return cookie_header
    finally:
        try:
            driver.quit()
        except Exception:
            pass
