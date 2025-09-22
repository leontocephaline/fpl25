#!/usr/bin/env python3
"""
Script to set FPL password in Windows keyring
"""

import keyring
import getpass

def set_fpl_password():
    """Set FPL password in Windows keyring"""
    print("FPL Password Setup for Windows Keyring")
    print("=" * 40)

    email = input("Enter your FPL email: ").strip()
    if not email:
        print("❌ Email is required")
        return

    password = getpass.getpass("Enter your FPL password: ")
    if not password:
        print("❌ Password is required")
        return

    service_name = "fpl-weekly-updater"

    try:
        # Set the password in keyring
        keyring.set_password(service_name, email, password)
        print(f"✅ Password successfully stored in Windows keyring for {email}")

        # Verify it was stored
        retrieved = keyring.get_password(service_name, email)
        if retrieved == password:
            print("✅ Password verification successful")
        else:
            print("❌ Password verification failed")

    except Exception as e:
        print(f"❌ Error storing password: {e}")
        print("Make sure you have Windows Credential Manager access")

if __name__ == "__main__":
    set_fpl_password()
