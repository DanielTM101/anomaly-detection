# Anomaly Detection in Energy and Network Systems

A Flask-based web application for detecting anomalies in energy consumption and network traffic using Apache Spark ML pipelines with custom domain-aware models.

## Overview

This project implements a multi-domain anomaly detection system that combines machine learning with domain-specific rules to identify unusual patterns in:

* **Energy Systems**: Detects anomalous energy consumption, voltage, solar output, and frequency deviations
* **Network Traffic**: Identifies suspicious network packets, port activities, and protocol anomalies

The system uses custom Spark ML pipelines, domain-aware transformers, and logistic regression models to provide both statistical and rule-based anomaly detection.

## Key Features

* 🔐 **User Authentication**: Role-based access control for administrators and regular users
* ⚡ **Energy Anomaly Detection**: Batch processing of CSV files with anomaly detection
* 🌐 **Network Stream Processing**: Streaming anomaly detection on network traffic
* 📊 **Interactive Dashboard**: Web interface for uploading data and viewing predictions
* 🚀 **Spark Integration**: Apache Spark processing with HDFS support
* 🏷️ **Domain-Specific Rules**: Custom transformers incorporating domain knowledge
* 📈 **Performance Logging**: Records processing and anomaly-detection metrics

## Technology Stack

### Backend

* **Framework**: Flask
* **Database**: SQLite with SQLAlchemy
* **Machine Learning**: Apache Spark, PySpark, MLlib
* **Data Processing**: Pandas, NumPy
* **Authentication**: Flask-Login

### Frontend

* HTML5
* Jinja2 Templates
* Bootstrap CSS
* Responsive web interface

### Infrastructure

* Apache Spark
* Spark MLlib
* Spark Streaming
* HDFS for distributed model and dataset storage
* Local filesystem for input files, predictions, logs, and checkpoints

---

## Project Structure

```text
├── app.py                          # Main Flask application
├── config.py                       # Application and environment configuration
├── custom_transformers.py          # Custom Spark ML transformers
├── auto_scaler.py                  # Auto-scaling utilities
├── enemodel.py                     # Energy model definitions
├── netmodel.py                     # Network model definitions
├── streaming_job.py                # Spark streaming job for network data
├── requirements.txt                # Python dependencies
│
├── energy_input/                   # Energy CSV input files
├── predictions_energy/             # Energy prediction outputs
├── downloads/                      # Downloadable anomaly reports
├── network_input/                  # Network streaming input files
├── checkpoint/                     # Spark Streaming checkpoint data
│
├── logs/
│   ├── energy_processing_logs/     # Energy processing logs
│   └── network_stream_logs/        # Network streaming logs
│
├── templates/                      # Jinja2 HTML templates
│   ├── index.html                  # Dashboard homepage
│   ├── login.html                  # User login
│   ├── register.html               # User registration
│   ├── profile.html                # User profile and settings
│   ├── upload_energy.html          # Energy data upload interface
│   ├── energy_auto_detect.html     # Automated energy anomaly detection
│   ├── admin_network.html          # Admin network anomaly viewer
│   └── layout.html                 # Base template
│
├── static/                         # Static assets
│   ├── dashboard_visual.svg        # Dashboard visualisation
│   └── user_profile.png            # Profile assets
│
└── README.md                       # Project documentation
```

---

# Core Components

## 1. Custom Transformers

### `custom_transformers.py`

Contains the custom Spark ML transformers used to incorporate domain-specific rules into the anomaly detection pipeline.

### MultiDomainRuleTransformer

Adds domain-specific feature flags for energy data:

* `timeFlag`: Detects unusual consumption during specific hours
* `voltageFlag`: Identifies voltage anomalies below 180V or above 260V
* `solarFlag`: Flags suspicious solar output patterns
* `freqFlag`: Detects frequency deviations from 50Hz

### NetworkDomainRuleTransformer

Adds domain-specific flags for network traffic:

* `bigPacketFlag`: Detects large packets
* `synFlag`: Monitors TCP SYN flags
* `suspiciousPortFlag`: Detects activity involving low-numbered ports
* `protocolFlag`: Identifies unusual protocols such as ICMP and AMQP

### CustomLogisticRegression

Custom implementation of logistic regression using NumPy, including:

* Gradient descent optimisation
* Configurable learning rate and iteration count
* Model persistence for Spark pipelines

---

## 2. Flask Application

### `app.py`

The main Flask application provides:

### User Management

* User registration
* User authentication
* Password management
* Role-based access control

### Energy Anomaly Detection

* `/upload_energy` — Upload and process energy CSV files
* `/energy_auto` — View automated energy detections
* `/api/trigger_energy_anomalies` — Trigger batch anomaly detection
* `/download_energy_anomalies/<filename>` — Download prediction results

### Network Monitoring

* `/admin_network` — View network anomaly results
* `/api/energy_anomalies_all` — Aggregate energy anomaly API endpoint

---

## 3. Configuration

### `config.py`

`config.py` centralises environment-specific configuration.

This includes:

* Energy input directory
* Energy prediction output directory
* Download directory
* Energy processing log directory
* Network log directory
* Network input directory
* Spark checkpoint directory
* HDFS host and port
* Energy model path
* Network model path
* Energy dataset path
* Network dataset path
* Flask secret key

This prevents machine-specific filesystem paths from being hardcoded throughout the application.

### Default Local Directories

The application can use directories relative to the project root:

```text
energy_input/
predictions_energy/
downloads/
network_input/
checkpoint/
logs/energy_processing_logs/
logs/network_stream_logs/
```

### Environment Variables

Custom locations can be supplied through environment variables.

For example:

```bash
export ENERGY_INPUT_DIR="/path/to/energy_input"
export ENERGY_OUTPUT_DIR="/path/to/predictions_energy"
export DOWNLOAD_DIR="/path/to/downloads"
export ENERGY_LOG_DIR="/path/to/energy_processing_logs"
export NETWORK_LOG_DIR="/path/to/network_stream_logs"
export NETWORK_INPUT_DIR="/path/to/network_input"
export CHECKPOINT_DIR="/path/to/checkpoint"
```

HDFS configuration can be supplied using:

```bash
export HDFS_HOST="localhost"
export HDFS_PORT="9000"
```

Model paths can also be overridden:

```bash
export ENERGY_MODEL_PATH="hdfs://<host>:<port>/<path>/oop_custom_energy_model"
export NETWORK_MODEL_PATH="hdfs://<host>:<port>/<path>/oop_custom_network_model"
```

Dataset paths can be configured similarly:

```bash
export ENERGY_DATASET_PATH="hdfs://<host>:<port>/<path>/energy_dataset.csv"
export NETWORK_DATASET_PATH="hdfs://<host>:<port>/<path>/train_network_anomaly_data.csv"
```

The Flask secret key should be supplied through:

```bash
export SECRET_KEY="your-secure-secret-key"
```

For production deployments, use a strong randomly generated secret and do not commit it to Git.

---

## 4. Streaming Jobs

### `streaming_job.py`

The network anomaly detection system uses Spark Streaming to:

* Consume network traffic data
* Apply custom domain-specific transformers
* Load the trained network anomaly model
* Generate anomaly predictions
* Record streaming results
* Use Spark checkpointing for streaming state

Network input is read from the configured:

```text
NETWORK_INPUT_DIR
```

Network processing logs are written to:

```text
NETWORK_LOG_DIR
```

Checkpoint data is stored using:

```text
CHECKPOINT_DIR
```

The trained network model is loaded using:

```text
NETWORK_MODEL_PATH
```

---

# Installation & Setup

## Prerequisites

* Python 3.8+
* Java 8+
* Apache Spark 3.0+
* PySpark
* HDFS — optional for local development, required when using distributed HDFS model storage
* Approximately 4 GB RAM recommended for local Spark processing

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/DanielTM101/anomaly-detection.git
cd anomaly-detection
```

### 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create required directories

```bash
mkdir -p energy_input
mkdir -p predictions_energy
mkdir -p downloads
mkdir -p network_input
mkdir -p checkpoint
mkdir -p logs/energy_processing_logs
mkdir -p logs/network_stream_logs
```

### 5. Configure the application

The application uses `config.py` for paths and environment-specific settings.

For a standard local installation, the default configuration should be sufficient.

If custom paths or HDFS settings are required, configure them using environment variables as described in the **Configuration** section above.

### 6. Initialise the database and run the application

```bash
python app.py
```

The Flask application will start using the configured Flask settings.

---

# Default Credentials

If the default development administrator account is enabled:

```text
Username: admin
Password: admin
```

**Important:** These credentials are intended for development/testing only. Change them before using the application in a production environment.

---

# Usage

## Energy Anomaly Detection

### Upload Data

1. Log in to the application.
2. Navigate to **Upload Energy**.
3. Select an energy CSV file.
4. Upload the file.
5. The application processes the data using the configured Spark model and domain-specific rules.
6. View the resulting anomaly predictions.

The expected CSV columns are:

```text
timestamp
energy_consumption
voltage
current
power
solar_output
frequency
label
```

Example:

```csv
timestamp,energy_consumption,voltage,current,power,solar_output,frequency,label
2025-01-01 12:00:00,45.2,230,5.1,1173,2.4,50.0,0
2025-01-01 13:00:00,82.7,235,6.4,1502,3.1,50.1,1
```

## Automated Batch Processing

The **Energy Auto-Detect** functionality scans the configured energy input directory and processes available files.

By default:

```text
energy_input/
```

Prediction results are stored in:

```text
predictions_energy/
```

Downloadable anomaly reports are stored in:

```text
downloads/
```

## Network Anomaly Detection

Network anomaly detection is available through the administrator interface.

The network streaming job must be running separately.

Network input files are placed in:

```text
network_input/
```

Run the streaming job with:

```bash
spark-submit streaming_job.py
```

Network processing results are recorded in the configured network log directory.

---

# Model Training & Configuration

## Energy Model Threshold

The current energy threshold is:

```python
THRESHOLD = 75
```

This threshold is used by the domain-specific energy anomaly rules.

## Custom Logistic Regression Settings

The current model configuration includes:

```python
maxIter = 20
lr = 0.01
prediction_threshold = 0.3
```

These parameters control the custom logistic regression training and classification behaviour.

## Time-Based Rules

The energy anomaly detection system includes time-based domain rules.

Current rules include:

* **Peak hours:** 12:00–17:00
* **Peak consumption threshold:** greater than 75 kWh
* **Off-peak periods:** before 05:00 and after 23:00
* High consumption during off-peak periods can be flagged as anomalous

The domain-specific rules are implemented in:

```text
custom_transformers.py
```

---

# Performance Logging

The application records processing metrics for energy and network workloads.

### Energy Logs

```text
logs/energy_processing_logs/
```

### Network Logs

```text
logs/network_stream_logs/
```

Logged information may include:

* Processing time
* Number of anomalies detected
* Timestamp
* File processed
* Streaming results

Performance measurements depend on the machine, Spark configuration, dataset size, and HDFS configuration.

---

# API Endpoints

## Authenticated Users

```text
GET  /api/energy_anomalies
POST /api/trigger_energy_anomalies
GET  /download_energy_anomalies/<filename>
```

## Admin Only

```text
GET /api/energy_anomalies_all
GET /admin_network
```

---

# Security Considerations

The application includes:

* Flask-Login authentication
* Role-based access control
* Admin-only routes
* File upload validation
* Configurable HDFS model paths
* Configurable Flask secret key

### Password Security

Passwords are stored using password hashing in the current implementation.

For production deployments, a dedicated password-hashing mechanism such as Werkzeug's password utilities should be used rather than relying on a plain SHA-256 hash.

### Secret Key

Set the Flask secret key through the environment:

```bash
export SECRET_KEY="your-secure-secret-key"
```

Do not commit production secret keys to the repository.

### Default Credentials

The default `admin/admin` credentials should only be used for development/testing and should be changed before deployment.

---

# Performance Benchmarks

Performance measurements are environment-dependent.

If benchmark figures are reported, they should be accompanied by the relevant test environment, including:

* CPU configuration
* RAM
* Spark version
* Python version
* Dataset size
* HDFS configuration

Example benchmark results should therefore be treated as measurements from a specific test environment rather than universal performance guarantees.

---

# Troubleshooting

## HDFS Connection Error

Check:

1. Hadoop NameNode is running.
2. `HDFS_HOST` is correct.
3. `HDFS_PORT` is correct.
4. The configured HDFS path exists.
5. Spark has permission to access the HDFS location.

For example:

```bash
hdfs dfs -ls /
```

If using the default model directory:

```bash
hdfs dfs -ls /models/
```

---

## No Anomalies Detected

Check:

1. The input CSV uses the expected column names.
2. The input values are valid.
3. The data meets the configured anomaly criteria.
4. The domain-specific rules are behaving as expected.
5. The correct trained model is being loaded.

---

## Model Not Found

Check:

1. The required model has been trained.
2. The configured model path is correct.
3. HDFS is available.
4. Spark can access HDFS.
5. HDFS permissions allow the application to read the model.

The relevant configuration variables are:

```text
ENERGY_MODEL_PATH
NETWORK_MODEL_PATH
```

---

## Directory Not Found

Check that the required directories exist:

```text
energy_input/
predictions_energy/
downloads/
network_input/
checkpoint/
logs/energy_processing_logs/
logs/network_stream_logs/
```

They can be recreated with:

```bash
mkdir -p energy_input
mkdir -p predictions_energy
mkdir -p downloads
mkdir -p network_input
mkdir -p checkpoint
mkdir -p logs/energy_processing_logs
mkdir -p logs/network_stream_logs
```

---

# Testing

Start the application:

```bash
python app.py
```

Test energy anomaly processing:

```bash
curl -X POST http://localhost:5000/api/trigger_energy_anomalies
```

For authenticated endpoints, an authenticated session may be required.

Test the network anomaly endpoint:

```bash
curl http://localhost:5000/api/energy_anomalies_all
```

Run network streaming separately:

```bash
spark-submit streaming_job.py
```

---

# Contributing

This is an undergraduate academic project, but contributions and improvements are welcome.

Before submitting changes:

1. Test the application locally.
2. Check that configuration does not contain machine-specific paths.
3. Do not commit passwords, API keys, or secret keys.
4. Update documentation when functionality changes.
5. Test energy and network functionality where applicable.

---

# Future Enhancements

Potential future improvements include:

* [ ] LSTM-based time-series anomaly detection
* [ ] Isolation Forest ensemble methods
* [ ] Real-time dashboard updates using WebSockets
* [ ] Docker containerisation
* [ ] Kubernetes deployment manifests
* [ ] Advanced authentication such as OAuth2 or LDAP
* [ ] Model versioning
* [ ] A/B model testing
* [ ] Explainable AI using SHAP
* [ ] Alert notifications
* [ ] Multi-tenancy support

---

# Author

**Daniel Adesanya**

BSc Natural Sciences (Mathematics and Computer Science)
Durham University, 2025

Undergraduate academic project.

---

**Last Updated:** April 2025
**Status:** Complete (Academic Project)
