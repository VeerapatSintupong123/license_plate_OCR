import requests
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Notifier:
    def __init__(self):
        """
        Retrieves credentials from the .env file for security.
        """
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        
        if not self.bot_token or not self.chat_id:
            logging.error("Notifier: Telegram credentials missing in .env!")

    def send_evidence(self, plate_text, province, image_path, duration):
        """
        Sends the plate details and the evidence photo to the Telegram group.
        """
        if not self.bot_token:
            return False

        # Professional formatting using Markdown
        message = (
            f"🚨 *ILLEGAL DUMPING DETECTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*Plate:* `{plate_text}`\n"
            f"*Province:* {province}\n"
            f"*Duration:* {int(duration)} seconds\n"
            f"*Status:* Evidence Archived ✅"
        )

        try:
            with open(image_path, 'rb') as photo:
                payload = {
                    'chat_id': self.chat_id,
                    'caption': message,
                    'parse_mode': 'Markdown'
                }
                files = {'photo': photo}
                
                response = requests.post(
                    self.api_url, 
                    data=payload, 
                    files=files, 
                    timeout=15
                )
            
            if response.status_code == 200:
                logging.info(f"Telegram: Alert sent for {plate_text}.")
                return True
            else:
                logging.error(f"Telegram: Error {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logging.error(f"Telegram: Connection failed: {e}")
            return False
