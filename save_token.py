import os
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
CREDENTIALS_FILE = "client_secret_2_496304292584-focgmts10r0pc3cplngprpkiqshp5d2j.apps.googleusercontent.com.json"
TOKEN_FILE = "token.json"

# 1) Создаём flow
flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)

# 2) Запускаем локальный сервер без автоматического открытия браузера:
#    он напечатает URL, который нужно скопировать и открыть вручную.
creds = flow.run_local_server(port=0, open_browser=False)

# 3) Сохраняем токены в файл
with open(TOKEN_FILE, "w") as f:
    f.write(creds.to_json())

print("✅ token.json успешно создан. Теперь скопируйте его в ваш проект.")
