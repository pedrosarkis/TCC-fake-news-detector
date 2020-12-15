cd /app
pip install -r ./requirements.txt
gunicorn -w 4 --bind 0.0.0.0:5000 web.main:app