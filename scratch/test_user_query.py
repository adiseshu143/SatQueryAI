import requests
import sys
sys.stdout.reconfigure(encoding='utf-8')

q1 = 'What\'s the difference between SAR and optical satellite imagery?"'
res = requests.post('http://127.0.0.1:8000/api/agent/chat', data={'query': q1})
print('STATUS:', res.status_code)
print('INTENT:', res.json().get('intent'))
print('ANSWER:\n', res.json().get('answer'))
