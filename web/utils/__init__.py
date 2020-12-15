import json
from decimal import Decimal


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if str(obj).isdigit() else float(obj)
        return json.JSONEncoder.default(self, obj)
