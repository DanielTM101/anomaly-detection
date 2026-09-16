# Anomaly Detection in Energy and Network Systems

A Flask-based web application for detecting anomalies in energy consumption and network traffic using Apache Spark ML pipelines with custom domain-aware models.

## Overview

This project implements a multi-domain anomaly detection system that combines machine learning with domain-specific rules to identify unusual patterns in:
- **Energy Systems**: Detects anomalous energy consumption, voltage, solar output, and frequency deviations
- **Network Traffic**: Identifies suspicious network packets, port activities, and protocol anomalies

The system uses a custom Spark ML pipeline with domain-aware transformers and logistic regression models to provide both statistical and rule-based anomaly detection.

## Key Features

- 🔐 **User Authentication**: Role-based access control (admin and regular users)
- ⚡ **Energy Anomaly Detection**: Batch processing of CSV files with real-time anomaly flagging
- 🌐 **Network Stream Processing**: Streaming anomaly detection on network traffic
- 📊 **Interactive Dashboard**: Web UI for uploading data and viewing predictions
- 🚀 **Spark Integration**: Distributed processing with HDFS support
- 🏷️ **Domain-Specific Rules**: Custom transformers encoding domain knowledge
- 📈 **Performance Logging**: Tracks processing metrics and model performance

## Technology Stack

### Backend
- **Framework**: Flask with SQLAlchemy ORM
- **Machine Learning**: Apache Spark (PySpark), MLlib
- **Data Processing**: Pandas, NumPy
- **Authentication**: Flask-Login 
- **Database**: SQLite

### Frontend
- HTML5/Jinja2 Templates
- Bootstrap CSS (via templates)
- Responsive design for mobile & desktop

### Infrastructure
- Apache Spark (Batch & Streaming)
- HDFS for distributed model storage
- Local file system for logs and predictions

## Project Structure

```
├── app.py                          # Main Flask application
├── custom_transformers.py          # Custom Spark ML transformers
├── auto_scaler.py                  # Auto-scaling utilities
├── enemodel.py                     # Energy model definitions
├── netmodel.py                     # Network model definitions
├── streaming_job.py                # Spark streaming job for network data
│
├── templates/                      # Jinja2 HTML templates
│   ├── index.html                  # Dashboard homepage
│   ├── login.html                  # User login
│   ├── register.html               # User registration
│   ├── profile.html                # User profile & settings
│   ├── upload_energy.html          # Energy data upload interface
│   ├── energy_auto_detect.html     # Automated energy anomaly detection
│   ├── admin_network.html          # Admin network anomaly viewer
│   └── layout.html                 # Base template
│
├── static/                         # Static assets
│   ├── dashboard_visual.svg        # Dashboard visualization
│   └── user_profile.png            # Profile assets
│
└── README.md                       # This file
```

## Core Components

### 1. **Custom Transformers** (`custom_transformers.py`)

#### MultiDomainRuleTransformer
Adds domain-specific feature flags for energy data:
- `timeFlag`: Detects unusual consumption during specific hours
- `voltageFlag`: Identifies voltage anomalies (< 180V or > 260V)
- `solarFlag`: Flags suspicious solar output patterns
- `freqFlag`: Detects frequency deviations from 50Hz

#### NetworkDomainRuleTransformer
Adds domain-specific flags for network traffic:
- `bigPacketFlag`: Large packet detection
- `synFlag`: TCP SYN flag monitoring
- `suspiciousPortFlag`: Low port number detection
- `protocolFlag`: Unusual protocol detection (ICMP, AMQP)

#### CustomLogisticRegression
Custom implementation of logistic regression using NumPy with:
- Gradient descent optimization
- Configurable learning rate and iterations
- Base64 serialization for Spark persistence

### 2. **Flask Application** (`app.py`)

**User Management**:
- Registration and authentication
- Password management
- Role-based access control

**Energy Anomaly Detection Routes**:
- `/upload_energy` - Upload and process CSV files
- `/energy_auto` - View automated detections
- `/api/trigger_energy_anomalies` - API endpoint for batch processing
- `/download_energy_anomalies/<filename>` - Download prediction results

**Network Monitoring Routes** (Admin Only):
- `/admin_network` - View real-time network anomalies
- `/api/energy_anomalies_all` - Aggregate API endpoint

### 3. **Streaming Jobs** (`streaming_job.py`)

Real-time Spark Streaming job that:
- Consumes network traffic data
- Applies custom transformers
- Generates predictions
- Logs results to HDFS

## Installation & Setup

### Prerequisites
- Python 3.8+
- Apache Spark 3.0+
- Java 8+
- HDFS (optional, for distributed deployment)

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/DanielTM101/anomaly-detection.git
cd anomaly-detection
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure paths** (in `app.py`)
Update these paths to match your environment:
```python
LOCAL_ENERGY_OUTPUT = "/path/to/predictions"
csv_folder = "/path/to/energy_input"
MODEL_PATH = "hdfs://localhost:9000/path/to/models"
```

5. **Initialize database and run**
```bash
python app.py
```

The application will start at `http://localhost:5000`

### Default Credentials
- Username: `admin`
- Password: `admin`

**⚠️ Important**: Change default credentials in production!

## Usage

### Energy Anomaly Detection

1. **Upload Data**:
   - Navigate to "Upload Energy"
   - Select a CSV file with columns: `timestamp`, `energy_consumption`, `voltage`, `current`, `power`, `solar_output`, `frequency`, `label`
   - System automatically detects anomalies

2. **Automated Batch Processing**:
   - Go to "Energy Auto-Detect"
   - System scans the input folder and processes latest files
   - Anomalies are displayed with explanations

3. **Download Results**:
   - Export prediction results as CSV
   - Includes columns: `predicted_label`, `demand_anomaly`, `reason`

### Network Anomaly Detection (Admin)

1. **Access Admin Panel** (admin users only):
   - Navigate to "Admin Network"
   - View streamed network anomalies in real-time
   - Network streaming job must be running separately

## Model Training & Configuration

### Energy Model Threshold
```python
THRESHOLD = 75  # Energy consumption threshold (kWh)
```

### Custom Logistic Regression Settings
```python
maxIter = 20      # Gradient descent iterations
lr = 0.01         # Learning rate
prediction_threshold = 0.3  # Probability threshold for classification
```

### Time-based Rules
Energy consumption is flagged as anomalous if:
- **Peak hours** (12:00-17:00): Consumption > 75 kWh
- **Off-peak** (before 05:00, after 23:00): Any high consumption

## Performance Logging

The system logs processing metrics to:
- **Energy logs**: `/logs/energy_processing_logs/{filename}_log.csv`
- **Network logs**: `/logs/network_stream_logs/`

Logged metrics include:
- Processing time (seconds)
- Number of anomalies detected
- Timestamp
- File processed

## API Endpoints

### Public (Authenticated Users)
```
GET  /api/energy_anomalies          # Latest energy anomalies
POST /api/trigger_energy_anomalies  # Trigger batch detection
GET  /download_energy_anomalies/<filename>  # Download results
```

### Admin Only
```
GET /api/energy_anomalies_all       # All energy anomalies
GET /admin_network                  # Network anomalies page
```

## Security Considerations

- User authentication implemented with Flask-Login
- Admin-only routes protected with role checks
- File uploads validated and sanitized
- HDFS paths configured for model storage

## Future Enhancements

- [ ] LSTM-based time-series anomaly detection
- [ ] Isolation Forest ensemble methods
- [ ] Real-time dashboard with WebSockets
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests
- [ ] Advanced authentication (OAuth2, LDAP)
- [ ] Model versioning and A/B testing
- [ ] Explainable AI (SHAP) integration
- [ ] Alert notifications (email, Slack)
- [ ] Multi-tenancy support

## Performance Benchmarks

- Energy file processing: ~2-5 seconds per 10K rows (on 4-core Spark cluster)
- Network stream throughput: ~1000 packets/second
- Model inference latency: <100ms per batch

## Troubleshooting

### "HDFS Connection Error"
- Ensure Hadoop namenode is running
- Verify HDFS URI in `app.py`
- Check network connectivity

### "No anomalies detected"
- Verify input CSV format matches schema
- Check if data meets threshold criteria
- Review domain rules in `custom_transformers.py`

### "Model not found"
- Ensure pre-trained model exists at `MODEL_PATH`
- Verify HDFS permissions
- Check Spark session configuration

## Testing

```bash
# Run with test data
python app.py

# Test energy detection
curl -X POST http://localhost:5000/api/trigger_energy_anomalies

# Test network anomalies
curl http://localhost:5000/api/energy_anomalies_all
```

## Contributing

This is an academic project, but contributions and improvements are welcome!

## Author

Daniel Adesanya  
BSc Natural Sciences (Mathematics and Computer Science), Durham University, 2025

**Last Updated**: April 2025  
**Status**: Complete (Academic Project)
