#!/usr/bin/env python3
"""Unified Tunnel Launcher for OneVoice AI Dashboard.

Supports:
  1. ngrok: via pyngrok / ngrok CLI (requires authtoken from https://dashboard.ngrok.com)
  2. localtunnel: instant free public HTTPS tunnel without token required.
"""

import os
import sys
import time
import subprocess

def start_ngrok(token: str = None, port: int = 8080):
    try:
        from pyngrok import ngrok
        if token:
            print(f"[+] Configuring ngrok authtoken...")
            ngrok.set_auth_token(token)
        elif "NGROK_AUTHTOKEN" in os.environ:
            print(f"[+] Using NGROK_AUTHTOKEN from environment...")
            ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])
        
        print(f"[+] Connecting ngrok tunnel to port {port}...")
        tunnel = ngrok.connect(port)
        public_url = tunnel.public_url
        print("=" * 80)
        print(f" 🚀 NGROK PUBLIC DEMO URL: {public_url}")
        print("=" * 80)
        sys.stdout.flush()
        return public_url
    except Exception as e:
        print(f"[!] ngrok connection failed: {e}")
        return None

def start_localtunnel(port: int = 8080):
    print(f"[+] Starting localtunnel on port {port}...")
    try:
        proc = subprocess.Popen(
            ["npx", "-y", "localtunnel", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            if "your url is:" in line:
                url = line.split("your url is:")[1].strip()
                print("=" * 80)
                print(f" 🌐 PUBLIC DEMO URL (LOCALTUNNEL): {url}")
                print("=" * 80)
                sys.stdout.flush()
                break
        # Keep reading to prevent pipe buffer from blocking
        for _ in proc.stdout:
            pass
        return proc
    except Exception as e:
        print(f"[!] localtunnel error: {e}")
        return None

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8080
    token = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("NGROK_AUTHTOKEN")
    
    if token:
        url = start_ngrok(token, port)
        if url:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n[+] Closing ngrok...")
    else:
        print("[*] No NGROK_AUTHTOKEN provided. Starting free localtunnel...")
        print("[*] (Tip: To use ngrok, pass your token: python dashboard/tunnel.py 8080 <your_ngrok_token>)")
        proc = start_localtunnel(port)
        if proc:
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
