# Stage 1: GitHub Actions Trigger Test
# Future stages (RSS, AI, Telegram, Instagram) will be added here later.

from datetime import datetime

def run():
    print("Stage 1 OK — GitHub Actions trigger is working.")
    print(f"Current UTC time: {datetime.utcnow()}")

if __name__ == "__main__":
    run()
