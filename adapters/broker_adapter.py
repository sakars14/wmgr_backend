from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BrokerAdapter(ABC):
    @abstractmethod
    def margins(self, uid: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def place_order(self, uid: str, *, exchange: str, symbol: str, qty: int,
                    product: str, order_type: str, variety: str,
                    price: Optional[float], tag: str) -> str:
        ...

    @abstractmethod
    def order_history(self, uid: str, order_id: str) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def holdings(self, uid: str) -> List[Dict[str, Any]]:
        ...
