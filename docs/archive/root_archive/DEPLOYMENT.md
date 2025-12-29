# NeuroVest Deployment Guide

Complete guide for deploying the Streamlit dashboard locally and in production.

---

## 🖥️ Local Deployment

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run dashboard.py
```

Dashboard will be available at: **http://localhost:8501**

### Advanced Local Setup

**Custom Port:**
```bash
streamlit run dashboard.py --server.port 8080
```

**Custom Host (allow external access):**
```bash
streamlit run dashboard.py --server.address 0.0.0.0
```

**With Specific Configuration:**
```bash
streamlit run dashboard.py --server.headless true --server.port 8501
```

### Configuration File

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
headless = true
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

---

## ☁️ Production Deployment

### Option 1: Streamlit Community Cloud (Recommended for Quick Start)

**Best for:** Free hosting, quick deployment, personal projects

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub account
   - Select repository: `your-username/NeuroVest`
   - Select branch: `main`
   - Set main file: `dashboard.py`
   - Click "Deploy"

3. **Configure Secrets:**
   - In Streamlit Cloud dashboard, go to Settings → Secrets
   - Add environment variables:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   NEWS_API_KEY = "your-newsapi-key"
   ```

4. **Access:**
   - Your app will be at: `https://your-app-name.streamlit.app`

**Limitations:**
- 1 GB RAM
- Limited CPU
- Community tier is free but public

---

### Option 2: Heroku

**Best for:** Production apps, custom domains, more resources

1. **Install Heroku CLI:**
   ```bash
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Create Heroku Files:**

   **`Procfile`:**
   ```
   web: streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0
   ```

   **`setup.sh`:**
   ```bash
   mkdir -p ~/.streamlit/

   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

   **`runtime.txt`:**
   ```
   python-3.10.12
   ```

3. **Deploy:**
   ```bash
   heroku login
   heroku create neurovest-dashboard
   git push heroku main
   ```

4. **Set Environment Variables:**
   ```bash
   heroku config:set OPENAI_API_KEY="sk-your-key"
   heroku config:set NEWS_API_KEY="your-key"
   ```

5. **Scale Up (if needed):**
   ```bash
   heroku ps:scale web=1
   ```

**Cost:** $7/month (Hobby tier) to $25-50/month (Production)

---

### Option 3: AWS EC2

**Best for:** Full control, scalability, enterprise deployments

1. **Launch EC2 Instance:**
   - AMI: Ubuntu 22.04 LTS
   - Instance type: t3.medium or larger
   - Security group: Allow inbound on port 8501

2. **SSH into Instance:**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Install Dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv nginx -y

   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate

   # Clone repository
   git clone https://github.com/your-username/NeuroVest.git
   cd NeuroVest

   # Install requirements
   pip install -r requirements.txt
   ```

4. **Create Systemd Service:**

   **`/etc/systemd/system/neurovest.service`:**
   ```ini
   [Unit]
   Description=NeuroVest Streamlit Dashboard
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/NeuroVest
   Environment="PATH=/home/ubuntu/NeuroVest/venv/bin"
   ExecStart=/home/ubuntu/NeuroVest/venv/bin/streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

5. **Configure Nginx (optional, for domain/SSL):**

   **`/etc/nginx/sites-available/neurovest`:**
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;

       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
       }
   }
   ```

6. **Enable and Start:**
   ```bash
   sudo systemctl enable neurovest
   sudo systemctl start neurovest
   sudo systemctl enable nginx
   sudo systemctl restart nginx
   ```

7. **SSL Certificate (recommended):**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d yourdomain.com
   ```

**Cost:** $15-30/month (t3.medium) + domain/storage

---

### Option 4: Docker

**Best for:** Containerized deployments, Kubernetes, consistency

1. **Create Dockerfile:**

   **`Dockerfile`:**
   ```dockerfile
   FROM python:3.10-slim

   WORKDIR /app

   # Install dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application
   COPY . .

   # Expose port
   EXPOSE 8501

   # Health check
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

   # Run streamlit
   ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

   **`.dockerignore`:**
   ```
   __pycache__
   *.pyc
   .git
   .env
   venv/
   logs/
   models/
   data_cache/
   ```

2. **Build Image:**
   ```bash
   docker build -t neurovest-dashboard .
   ```

3. **Run Container:**
   ```bash
   docker run -p 8501:8501 \
     -e OPENAI_API_KEY="your-key" \
     -e NEWS_API_KEY="your-key" \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     neurovest-dashboard
   ```

4. **Docker Compose (recommended):**

   **`docker-compose.yml`:**
   ```yaml
   version: '3.8'

   services:
     dashboard:
       build: .
       ports:
         - "8501:8501"
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - NEWS_API_KEY=${NEWS_API_KEY}
         - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
       volumes:
         - ./data:/app/data
         - ./models:/app/models
         - ./logs:/app/logs
       restart: unless-stopped
   ```

   **Run:**
   ```bash
   docker-compose up -d
   ```

---

### Option 5: Google Cloud Run

**Best for:** Serverless, auto-scaling, pay-per-use

1. **Install Google Cloud SDK:**
   ```bash
   curl https://sdk.cloud.google.com | bash
   gcloud init
   ```

2. **Create `cloudbuild.yaml`:**
   ```yaml
   steps:
     - name: 'gcr.io/cloud-builders/docker'
       args: ['build', '-t', 'gcr.io/$PROJECT_ID/neurovest-dashboard', '.']
     - name: 'gcr.io/cloud-builders/docker'
       args: ['push', 'gcr.io/$PROJECT_ID/neurovest-dashboard']

   images:
     - 'gcr.io/$PROJECT_ID/neurovest-dashboard'
   ```

3. **Deploy:**
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   gcloud run deploy neurovest-dashboard \
     --image gcr.io/PROJECT_ID/neurovest-dashboard \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars OPENAI_API_KEY=your-key
   ```

**Cost:** Pay-per-request, ~$5-20/month for moderate traffic

---

## 📊 Performance Optimization

### For Production Deployments

**1. Enable Caching:**
```python
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_predictions():
    return pd.read_csv('logs/daily_predictions.csv')
```

**2. Optimize Data Loading:**
```python
# Use parquet instead of CSV
df.to_parquet('data.parquet')
df = pd.read_parquet('data.parquet')  # 10x faster
```

**3. Lazy Loading:**
```python
# Load data only when needed
if st.button('Show Analysis'):
    data = load_heavy_data()
```

**4. Resource Limits:**
```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200
maxMessageSize = 200

[runner]
magicEnabled = true
fastReruns = true
```

---

## 🔒 Security Best Practices

### Environment Variables

**Never commit:**
- API keys
- Database credentials
- SMTP passwords

**Use `.env` file locally:**
```bash
# .env
OPENAI_API_KEY=sk-your-key
NEWS_API_KEY=your-key
```

**Add to `.gitignore`:**
```
.env
.streamlit/secrets.toml
```

### Streamlit Secrets (Production)

**`.streamlit/secrets.toml`:**
```toml
OPENAI_API_KEY = "sk-your-key"
NEWS_API_KEY = "your-key"
```

**Access in code:**
```python
import streamlit as st
api_key = st.secrets["OPENAI_API_KEY"]
```

---

## 🔍 Monitoring & Logging

### Application Logs

```python
import logging

logging.basicConfig(
    filename='dashboard.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Dashboard started")
```

### Error Tracking (Production)

Use Sentry for error tracking:

```python
import sentry_sdk

sentry_sdk.init(
    dsn="your-sentry-dsn",
    traces_sample_rate=1.0
)
```

---

## 🚀 Quick Deployment Comparison

| Platform | Cost | Setup Time | Scalability | Control | Best For |
|----------|------|------------|-------------|---------|----------|
| **Streamlit Cloud** | Free | 5 min | Low | Low | Quick demos |
| **Heroku** | $7-50/mo | 15 min | Medium | Medium | Small apps |
| **AWS EC2** | $15-100/mo | 30 min | High | High | Production |
| **Docker** | Varies | 20 min | High | High | DevOps teams |
| **Google Cloud Run** | Pay-per-use | 20 min | Very High | Medium | Serverless |

---

## 📱 Mobile Access

All deployments are mobile-responsive by default. Access from:
- iOS Safari
- Android Chrome
- Any modern mobile browser

For better mobile UX, consider:
```python
# Adjust layout for mobile
if st.session_state.get('mobile_mode'):
    st.write("Mobile-optimized view")
```

---

## 🆘 Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Find and kill process on port 8501
lsof -ti:8501 | xargs kill -9
```

**Module not found:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Memory errors:**
```bash
# Increase Streamlit memory limit
streamlit run dashboard.py --server.maxUploadSize 1024
```

**Deployment failed:**
```bash
# Check logs
heroku logs --tail  # Heroku
docker logs container-id  # Docker
journalctl -u neurovest -f  # Systemd
```

---

## 📞 Support

For deployment issues:
1. Check [Streamlit documentation](https://docs.streamlit.io)
2. Review platform-specific docs (Heroku, AWS, etc.)
3. Open issue on GitHub

---

**Last Updated:** December 2024
