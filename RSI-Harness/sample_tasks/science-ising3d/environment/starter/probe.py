"""Reproduce exactly what the starter implements; make no scientific pass claim."""
import json
from model import pressure

print(json.dumps({
    "scope": "local polynomial only; no controlled global or critical result",
    "pressure": {str(K): pressure(K) for K in [0, 0.01, 0.1, 0.2, 1, 8]},
    "scientific_completion_claimed": False,
}))
