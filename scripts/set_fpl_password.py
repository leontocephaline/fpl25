from __future__ import annotations

import getpass
import sys
from typing import Optional

import keyring

SERVICE_NAME = "fpl-weekly-updater"


def prompt_and_store(email: Optional[str] = None) -> None:
    """Prompt for FPL email and password and store the password in the OS keyring.

    This avoids committing secrets to Git while letting the app retrieve them at runtime.
    """
    if not email:
        email = input("FPL email: ").strip()
    if not email:
        print("No email provided; aborting.")
        sys.exit(1)

    pwd = getpass.getpass("FPL password (input hidden): ")
    if not pwd:
        print("Empty password; aborting.")
        sys.exit(1)

    keyring.set_password(SERVICE_NAME, email, pwd)
    print(f"Stored password for {email} in the OS keyring under service '{SERVICE_NAME}'.")


if __name__ == "__main__":
    e = sys.argv[1] if len(sys.argv) > 1 else None
    prompt_and_store(e)
