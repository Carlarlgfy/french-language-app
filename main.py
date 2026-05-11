#!/usr/bin/env python3
"""
Voice Chat AI — French Tutor
Main entry point

Run this file to start the application.
"""
import sys
import os

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Import and run the pygame version
from voice_chat_client_pygame import main

if __name__ == "__main__":
    main()
