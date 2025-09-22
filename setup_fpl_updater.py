#!/usr/bin/env python3
"""
FPL Weekly Updater Setup Script
Helps users configure their credentials and API keys for the FPL Weekly Updater executable
"""

import os
import sys
import json
from pathlib import Path
import keyring
import getpass

def create_env_file():
    """Create .env file with user configuration"""
    print("=" * 60)
    print("FPL Weekly Updater - First Time Setup")
    print("=" * 60)

    # Get project root (directory containing this script)
    script_dir = Path(__file__).parent
    env_file = script_dir / ".env"

    if env_file.exists():
        print("⚠️  .env file already exists!")
        overwrite = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("Setup cancelled. Your existing .env file was preserved.")
            return False

    print("\n📧 FANTASY PREMIER LEAGUE CONFIGURATION")
    print("-" * 40)

    fpl_email = input("Enter your FPL email address: ").strip()
    if not fpl_email:
        print("❌ FPL email is required")
        return False

    fpl_team_id = input("Enter your FPL team ID (optional, can be found in FPL URL): ").strip()
    fpl_team_id = int(fpl_team_id) if fpl_team_id.isdigit() else None

    print("\n🤖 PERPLEXITY API CONFIGURATION (for player news analysis)")
    print("-" * 40)
    print("Perplexity provides AI-powered player news analysis.")
    print("Get your API key from: https://www.perplexity.ai/settings/api")

    perplexity_key = input("Enter your Perplexity API key (optional, press Enter to skip): ").strip()
    if perplexity_key and not perplexity_key.startswith('pplx-'):
        print("⚠️  Warning: Perplexity API keys usually start with 'pplx-'")

    print("\n🔧 ADDITIONAL SETTINGS")
    print("-" * 40)

    report_dir = input("Where should reports be saved? (default: Desktop): ").strip()
    report_dir = report_dir if report_dir else str(Path.home() / "Desktop")

    browser = input("Which browser for FPL login? (edge/chrome/firefox, default: edge): ").strip().lower()
    browser = browser if browser in ['chrome', 'firefox'] else 'edge'

    headless = input("Run browser in headless mode? (y/N, default: y): ").lower().strip()
    headless = headless == 'y'

    # Create .env content
    env_content = f"""# Fantasy Premier League Weekly Updater Configuration
# This file contains your API keys and settings

# FPL Configuration
FPL_EMAIL={fpl_email}
FPL_TEAM_ID={fpl_team_id or ''}
FPL_BROWSER={browser}
FPL_BROWSER_HEADLESS={str(headless).lower()}

# Perplexity API (for player news analysis)
PERPLEXITY_API_KEY={perplexity_key}

# Output Settings
REPORT_OUTPUT_DIR={report_dir}

# Optional: Advanced Settings (usually not needed)
# FORCE_MODEL_RETRAIN=false
"""

    # Write .env file
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created .env file at: {env_file}")
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

    return True

def setup_password():
    """Set FPL password in Windows keyring"""
    print("\n🔐 FPL PASSWORD SETUP")
    print("-" * 40)
    print("Your FPL password will be stored securely in Windows Credential Manager.")
    print("This is much safer than storing it in plain text files.")

    email = input("Enter your FPL email address: ").strip()
    if not email:
        print("❌ FPL email is required")
        return False

    password = getpass.getpass("Enter your FPL password: ")
    if not password:
        print("❌ FPL password is required")
        return False

    confirm_password = getpass.getpass("Confirm your FPL password: ")
    if password != confirm_password:
        print("❌ Passwords don't match")
        return False

    service_name = "fpl-weekly-updater"

    try:
        keyring.set_password(service_name, email, password)
        print("✅ Password stored securely in Windows Credential Manager"        print("📋 You can view/manage it in: Control Panel → User Accounts → Credential Manager")

        # Verify it was stored
        retrieved = keyring.get_password(service_name, email)
        if retrieved == password:
            print("✅ Password verification successful")
        else:
            print("❌ Password verification failed")
            return False

    except Exception as e:
        print(f"❌ Error storing password: {e}")
        print("Make sure you have Windows Credential Manager access")
        return False

    return True

def test_configuration():
    """Test that the configuration works"""
    print("\n🧪 TESTING CONFIGURATION")
    print("-" * 40)

    try:
        # Test .env file loading
        from fpl_weekly_updater.config.settings import load_settings
        settings = load_settings()

        print("✅ Configuration loaded successfully"        print(f"   FPL Email: {'***' if settings.fpl_email else 'Not set'}")
        print(f"   FPL Team ID: {settings.fpl_team_id or 'Not set'}")
        print(f"   Perplexity API: {'***' if settings.perplexity_api_key else 'Not set'}")
        print(f"   Report Directory: {settings.report_output_dir}")

        # Test keyring
        if settings.fpl_email:
            try:
                password = keyring.get_password("fpl-weekly-updater", settings.fpl_email)
                print(f"✅ FPL Password: {'Stored' if password else 'Not found'}")
            except Exception as e:
                print(f"⚠️  FPL Password: Could not access ({e})")

        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Welcome to FPL Weekly Updater Setup!")
    print("This script will help you configure the application for first use.")

    # Check if we're in the right directory
    if not (Path(__file__).parent / "fpl_weekly_updater").exists():
        print("❌ Error: This script must be run from the FPL Weekly Updater directory")
        sys.exit(1)

    print("\nStep 1: Create .env configuration file")
    if not create_env_file():
        return

    print("\nStep 2: Set up FPL password in Windows keyring")
    if not setup_password():
        return

    print("\nStep 3: Test configuration")
    if test_configuration():
        print("\n🎉 SETUP COMPLETE!")
        print("You can now run the FPL Weekly Updater executable:")
        print("  .\\dist\\FPLWeeklyUpdater.exe")
    else:
        print("\n⚠️  Setup completed but configuration test failed.")
        print("Check the error messages above and try again.")

if __name__ == "__main__":
    main()
