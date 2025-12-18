from adapters.zerodha_adapter import ZerodhaAdapter
from google.cloud import firestore
import os

# assumes GOOGLE_APPLICATION_CREDENTIALS is set like for main.py
db = firestore.Client()
adapter = ZerodhaAdapter(db)

UID = os.environ["DEBUG_UID"]  # set this to your Firebase uid

print("Fetching LTP for NSE:SILVERCASE for", UID)
print("LTP:", adapter.get_ltp(UID, "NSE", "SILVERCASE"))