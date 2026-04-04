from app import app
c = app.test_client()
r = c.get('/dashboard')
html = r.get_data(as_text=True)
start = html.find('<head>')
end = html.find('</head>')
print(html[start:end])
print('\n--- Links ---')
import re
links = re.findall(r'<link[^>]+>', html[start:end])
for l in links:
    print(l)
