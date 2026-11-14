import os
import threading
import logging
from twilio.rest import Client

def _send_sms_task(emergency_type: str, location: str, to_phone_number: str):
    """
    The actual worker function that runs in the background to send the SMS.
    """
    try:
        # 1. Load credentials from environment variables
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        from_phone = os.environ.get("TWILIO_PHONE_NUMBER")
        
        # Verify credentials exist
        if not all([account_sid, auth_token, from_phone]):
            logging.error("Twilio credentials are missing from environment variables.")
            return

        # 2. Initialize the client
        client = Client(account_sid, auth_token)
        
        # 3. Format custom message
        message_body = f"EMERGENCY ALERT: {emergency_type.upper()}! Required at Location: {location}."
        
        # 4. Send the SMS
        message = client.messages.create(
            body=message_body,
            from_=from_phone,
            to=to_phone_number
        )
        
        logging.info(f"Emergency SMS sent successfully. Message SID: {message.sid}")
        
    except Exception as e:
        # 5. Log the error if the sending process fails
        logging.error(f"Failed to send emergency SMS: {e}")

def send_emergency_sms_async(emergency_type: str, location: str, to_phone_number: str):
    """
    Triggers the emergency SMS immediately without slowing down the main program.
    """
    # Create a background thread for the SMS task
    alert_thread = threading.Thread(
        target=_send_sms_task,
        args=(emergency_type, location, to_phone_number),
        daemon=True # Ensures the thread closes if the main program exits early
    )
    
    # Start the thread immediately (asynchronous execution)
    alert_thread.start()
