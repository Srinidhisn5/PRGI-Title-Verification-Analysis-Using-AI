from app import app

client = app.test_client()
with client.session_transaction() as sess:
    sess['admin_logged_in'] = True

print('No filters count:')
resp = client.get('/admin/dashboard')
print(resp.status_code, len(resp.get_data(as_text=True)))

print('Status=Approved:')
resp = client.get('/admin/dashboard?status=Approved')
print(resp.status_code, len(resp.get_data(as_text=True)))

print('State=Unknown:')
resp = client.get('/admin/dashboard?state=Unknown')
print(resp.status_code, len(resp.get_data(as_text=True)))

print('Status=Approved & State=Unknown:')
resp = client.get('/admin/dashboard?status=Approved&state=Unknown')
print(resp.status_code, len(resp.get_data(as_text=True)))
