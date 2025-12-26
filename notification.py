from twilio.rest import Client
import keys # Placeholder keys file

def send_alert(msg):
    try:
        client = Client(keys.account_sid, keys.auth_token)
        message = client.messages.create(
            body=msg,
            from_= keys.twilio_number,
            to = keys.target_number
        )
        print("ALERT SENT SUCCESSFULLY")
    except Exception as e:
        print(f"ALERT FAILED: {e}")
