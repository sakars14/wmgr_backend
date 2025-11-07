import os
from typing import Dict, Any, List, Optional
from kiteconnect import KiteConnect
from google.cloud import firestore

class ZerodhaAdapter:
    def __init__(self, db: firestore.Client):
        self.db = db
        self.api_key = os.getenv("ZERODHA_API_KEY", "")
        self.api_secret = os.getenv("ZERODHA_API_SECRET", "")

    def _get_access_token(self, uid: str) -> str:
        doc = self.db.collection("users").document(uid).collection("brokerLinks").document("zerodha").get()
        if not doc.exists:
            raise RuntimeError("Zerodha not connected for user")
        data = doc.to_dict()
        token = data.get("accessToken")
        if not token:
            raise RuntimeError("Missing access token")
        return token

    def _kite(self, uid: str) -> KiteConnect:
        k = KiteConnect(api_key=self.api_key)
        k.set_access_token(self._get_access_token(uid))
        return k

    def margins(self, uid: str) -> Dict[str, Any]:
        kite = self._kite(uid)
        return kite.margins()

    def place_order(self, uid: str, *, exchange: str, symbol: str, qty: int,
                    product: str, order_type: str, variety: str,
                    price: Optional[float], tag: str) -> str:
        kite = self._kite(uid)
        resp = kite.place_order(
            variety=variety,
            exchange=exchange,
            tradingsymbol=symbol,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=qty,
            product=product,
            order_type=order_type,
            price=None if order_type == kite.ORDER_TYPE_MARKET else price,
            validity=kite.VALIDITY_DAY,
            tag=tag
        )
        return resp["order_id"] if isinstance(resp, dict) else str(resp)

    def order_history(self, uid: str, order_id: str) -> List[Dict[str, Any]]:
        kite = self._kite(uid)
        return kite.order_history(order_id)

    def holdings(self, uid: str) -> List[Dict[str, Any]]:
        kite = self._kite(uid)
        return kite.holdings()
        
    def get_ltp(self, uid: str, exchange: str, symbol: str) -> float:
        """Return LTP for one instrument like NSE:SILVERCASE."""
        kite = self._kite(uid)  # your existing helper that sets access_token
        inst = f"{exchange}:{symbol}"
        q = kite.quote([inst])
        # Zerodha returns {'NSE:SILVERCASE': {'last_price': 18.52, ...}}
        return float(q[inst]["last_price"])