import requests
url = "http://127.0.0.1:8000/ask?question=what%20is%20mindcoders"
response = requests.get(url)
t = response.json()
print(t)

requests.post()