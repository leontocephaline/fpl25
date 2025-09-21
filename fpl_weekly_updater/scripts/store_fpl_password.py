from __future__ import annotations

import getpass
import sys

import keyring

SERVICE_NAME = "fpl-weekly-updater"

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python store_fpl_password.py <email>")
        return 2
    email = sys.argv[1]
    pw = getpass.getpass(f"FPL password for {email}: ")
    keyring.set_password(SERVICE_NAME, email, pw)
    print(f"Password stored for {email} in Windows Credential Manager.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
