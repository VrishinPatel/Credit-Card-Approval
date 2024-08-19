import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    'A1': 'a',          # Example value for A1
    'A2': 30.83,        # Example value for A2 (continuous)
    'A3': 0,            # Example value for A3 (continuous)
    'A4': 'u',          # Example value for A4
    'A5': 'g',          # Example value for A5
    'A6': 'w',          # Example value for A6
    'A7': 'v',          # Example value for A7
    'A8': 1.25,         # Example value for A8 (continuous)
    'A9': 't',          # Example value for A9
    'A10': 't',         # Example value for A10
    'A11': 1,           # Example value for A11 (continuous)
    'A12': 'f',         # Example value for A12
    'A13': 'g',         # Example value for A13
    'A14': 202,         # Example value for A14 (continuous)
    'A15': 0            # Example value for A15 (continuous)
}

response = requests.post(url, json=data)
print(response.json())
