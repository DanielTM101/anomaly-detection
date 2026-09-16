# Setup Guide — Anomaly Detection in Energy and Network Systems

This guide explains how to install, configure, and run the anomaly detection application.

The project is a Flask-based web application using Apache Spark and custom machine-learning components to detect anomalies in energy systems and network traffic.

---

## 1. Prerequisites

Before installing the application, make sure the following are available:

* Python 3.8+
* Java 8+
* Apache Spark 3.0+
* PySpark
* HDFS — optional for local development, required for the distributed model-storage configuration
* Approximately 4 GB RAM recommended for running Spark locally

Check your Python installation:

```bash
python3 --version
```

Check Java:

```bash
java -version
```

If Spark is installed separately, check:

```bash
spark-submit --version
```

---

## 2. Clone the Repository

Clone the repository and move into the project directory:

```bash
git clone <repository-url>
cd anomaly-detection
```

The repository is:

```text
DanielTM101/anomaly-detection
```

---

## 3. Create a Virtual Environment

Creating a virtual environment is recommended so that the project's Python dependencies are isolated from the rest of your system.

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

After activation, your terminal should indicate that the virtual environment is active.

---

## 4. Install Dependencies

Install the Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

If PySpark is not installed through the requirements file, install it with:

```bash
pip install pyspark
```

---

## 5. Project Directory Structure

The application uses several local directories for input files, predictions, logs, network streaming data, and Spark checkpoints.

Create the required directories from the project root:

```bash
mkdir -p energy_input
mkdir -p predictions_energy
mkdir -p downloads
mkdir -p logs/energy_processing_logs
mkdir -p logs/network_stream_logs
mkdir -p network_input
mkdir -p checkpoint
```

The resulting structure should look similar to:

```text
anomaly-detection/
│
├── app.py
├── config.py
├── auto_scaler.py
├── custom_transformers.py
├── enemodel.py
├── netmodel.py
├── streaming_job.py
├── requirements.txt
│
├── energy_input/
├── predictions_energy/
├── downloads/
├── network_input/
├── checkpoint/
│
├── logs/
│   ├── energy_processing_logs/
│   └── network_stream_logs/
│
├── templates/
└── static/
```

These directories are configured through `config.py`.

---

## 6. Application Configuration

The application now uses `config.py` to manage environment-specific paths and configuration values.

This means that paths should **not** need to be edited directly inside `app.py`, `enemodel.py`, `netmodel.py`, or `streaming_job.py`.

### Default Local Configuration

The default configuration uses directories relative to the project directory.

For a standard local installation, no path changes should be necessary.

The main local directories are:

```text
energy_input/
predictions_energy/
downloads/
network_input/
checkpoint/
logs/energy_processing_logs/
logs/network_stream_logs/
```

---

## 7. Customising Local Paths

If you want to store files somewhere other than the project directory, configure the locations using environment variables.

### Energy input

```bash
export ENERGY_INPUT_DIR="/path/to/energy_input"
```

### Energy prediction output

```bash
export ENERGY_OUTPUT_DIR="/path/to/predictions_energy"
```

### Download directory

```bash
export DOWNLOAD_DIR="/path/to/downloads"
```

### Energy processing logs

```bash
export ENERGY_LOG_DIR="/path/to/energy_processing_logs"
```

### Network streaming logs

```bash
export NETWORK_LOG_DIR="/path/to/network_stream_logs"
```

### Network streaming input

```bash
export NETWORK_INPUT_DIR="/path/to/network_input"
```

### Spark streaming checkpoint

```bash
export CHECKPOINT_DIR="/path/to/checkpoint"
```

These variables are optional. If they are not supplied, `config.py` should use the default directories within the project.

---

## 8. HDFS Configuration

HDFS is used for distributed storage of the trained Spark models and datasets.

The HDFS connection is configured through environment variables rather than being tied to a specific machine.

### HDFS Host

```bash
export HDFS_HOST="localhost"
```

### HDFS Port

```bash
export HDFS_PORT="9000"
```

The default model locations are:

```text
hdfs://localhost:9000/models/oop_custom_energy_model
hdfs://localhost:9000/models/oop_custom_network_model
```

If your HDFS installation uses a different host or port, change `HDFS_HOST` and `HDFS_PORT`.

---

## 9. Custom HDFS Model Paths

If the models are stored somewhere else in HDFS, configure them explicitly.

### Energy model

```bash
export ENERGY_MODEL_PATH="hdfs://<host>:<port>/<path>/oop_custom_energy_model"
```

### Network model

```bash
export NETWORK_MODEL_PATH="hdfs://<host>:<port>/<path>/oop_custom_network_model"
```

The corresponding dataset locations can also be configured.

### Energy dataset

```bash
export ENERGY_DATASET_PATH="hdfs://<host>:<port>/<path>/energy_dataset.csv"
```

### Network dataset

```bash
export NETWORK_DATASET_PATH="hdfs://<host>:<port>/<path>/train_network_anomaly_data.csv"
```

This allows the same application code to be used on different machines or Spark/HDFS configurations without changing the Python source files.

---

## 10. Flask Secret Key

The Flask application uses a secret key for session security.

Set the secret key using an environment variable:

```bash
export SECRET_KEY="replace-with-a-secure-random-key"
```

For example, a random key can be generated with Python:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Then set the generated value:

```bash
export SECRET_KEY="your-generated-key"
```

Do not commit a production secret key to the repository.

For local development, `config.py` may provide a development fallback.

---

## 11. Database Initialisation

The application uses SQLite for database storage.

Start the Flask application:

```bash
python app.py
```

On first startup, the application should create the required database structure if it does not already exist.

If the application creates a default administrator account, the current development credentials are:

```text
Username: admin
Password: admin
```

These credentials should not be used for a production deployment.

---

## 12. Running the Application

From the project directory, activate the virtual environment and run:

```bash
python app.py
```

The Flask application will run on the configured local Flask port.

Open the application in a web browser using the local address displayed in the Flask terminal output.

---

# 13. Testing the Energy Anomaly Detection

## 13.1 Energy CSV Format

Energy input files should contain the expected columns:

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

Place the CSV file inside:

```text
energy_input/
```

---

## 13.2 Upload Energy Data

1. Start the Flask application.
2. Log in.
3. Navigate to **Upload Energy**.
4. Select a CSV file.
5. Upload the file.
6. The application processes the data using the configured Spark model and domain rules.
7. View the resulting anomaly predictions.

---

## 13.3 Automated Energy Detection

The automated energy detection functionality scans the configured energy input directory.

By default:

```text
energy_input/
```

The application processes available input files and produces anomaly predictions.

Prediction outputs are stored in:

```text
predictions_energy/
```

Downloaded anomaly reports are stored in:

```text
downloads/
```

---

# 14. Network Anomaly Detection

Network anomaly detection uses Spark Streaming.

Network input data should be placed in:

```text
network_input/
```

The streaming job reads network data, applies the trained network anomaly model and domain-specific rules, and produces anomaly results.

Network processing logs are stored in:

```text
logs/network_stream_logs/
```

The network streaming job must be running separately from the Flask application.

---

## 15. Running the Network Streaming Job

The streaming job is implemented in:

```text
streaming_job.py
```

Run it using Spark:

```bash
spark-submit streaming_job.py
```

The job uses the paths configured in `config.py`, including:

```text
NETWORK_MODEL_PATH
NETWORK_INPUT_DIR
NETWORK_LOG_DIR
CHECKPOINT_DIR
```

This avoids hardcoding machine-specific filesystem paths inside the streaming job.

---

# 16. Spark Configuration

The application uses PySpark for machine-learning and data-processing tasks.

A local Spark session can use all available CPU cores:

```python
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("AnomalyDetection")
    .master("local[*]")
    .getOrCreate()
)
```

Do not place a comment after a line-continuation backslash. For example, avoid:

```python
.master("local[*]") \  # comment
```

Instead use:

```python
.master("local[*]")
```

or place the comment on a separate line.

---

## 17. HDFS Verification

If HDFS is being used, first verify that the Hadoop NameNode is running.

Check the configured HDFS location:

```bash
hdfs dfs -ls /
```

If your models are stored in the default model directory, check:

```bash
hdfs dfs -ls /models/
```

The exact HDFS location depends on the values configured in:

```text
config.py
```

or through the relevant environment variables.

---

# 18. Model Training

The project contains separate model implementations for the energy and network domains.

### Energy model

```text
enemodel.py
```

### Network model

```text
netmodel.py
```

The models use Spark ML components together with custom domain-specific transformers.

Before running inference, ensure that the required trained model exists at the configured HDFS model path.

---

# 19. Model Configuration

The energy anomaly detection model uses domain-specific thresholds and rules.

The energy threshold is currently:

```python
THRESHOLD = 75
```

The custom logistic regression configuration includes:

```python
maxIter = 20
lr = 0.01
prediction_threshold = 0.3
```

These values are model/application parameters rather than machine-specific filesystem configuration.

---

## 20. Time-Based Energy Rules

The system uses time-based domain rules when identifying unusual energy consumption.

The current rules include:

* Peak hours: `12:00–17:00`
* Peak consumption threshold: greater than `75 kWh`
* Off-peak periods: before `05:00` and after `23:00`
* High consumption during off-peak periods can be flagged as anomalous

The domain-specific implementation can be found in:

```text
custom_transformers.py
```

---

# 21. Performance Logging

The application records processing information for energy and network workloads.

### Energy logs

```text
logs/energy_processing_logs/
```

### Network logs

```text
logs/network_stream_logs/
```

Logged information may include:

* Processing time
* Number of anomalies detected
* Timestamp
* File processed
* Network streaming results

These logs can be used to evaluate the performance of the application during testing.

---

# 22. API Endpoints

The application provides several API endpoints.

## Authenticated User Endpoints

```text
GET  /api/energy_anomalies
POST /api/trigger_energy_anomalies
GET  /download_energy_anomalies/<filename>
```

## Admin Endpoints

```text
GET /api/energy_anomalies_all
GET /admin_network
```

Access to administrator functionality is controlled through the application's authentication and role system.

---

# 23. Database Management

The application uses SQLite for user and application data.

The database file should be treated as local application data and should not normally be committed to the repository.

If the database needs to be reset during development, stop the application and remove the local SQLite database file before restarting the application.

Only perform this operation if you understand that existing local database records will be removed.

---

# 24. Security Considerations

The application includes:

* Flask-Login authentication
* Role-based access control
* Admin-only routes
* File-upload validation
* Configurable HDFS model paths
* Configurable Flask secret key

### Passwords

The current project implementation uses password hashing for stored credentials.

For production use, a dedicated password-hashing mechanism such as Werkzeug's password utilities should be preferred over a plain cryptographic hash such as SHA-256.

### Secret Key

The Flask secret key should be supplied through the `SECRET_KEY` environment variable for production deployments.

Do not commit real production credentials or secret keys to Git.

### Default Credentials

If the default development administrator account is enabled:

```text
Username: admin
Password: admin
```

Change these credentials before any non-development deployment.

---

# 25. Troubleshooting

## "HDFS Connection Error"

Check the following:

1. Ensure the Hadoop NameNode is running.
2. Check `HDFS_HOST` and `HDFS_PORT`.
3. Verify that the configured HDFS model path exists.
4. Check network connectivity between Spark and HDFS.
5. Verify HDFS permissions.

For example:

```bash
hdfs dfs -ls /
```

---

## "Model Not Found"

Check:

1. The required model has been trained.
2. The model exists at the configured `ENERGY_MODEL_PATH` or `NETWORK_MODEL_PATH`.
3. The HDFS host and port are correct.
4. Spark can access HDFS.
5. The configured path matches the location used when the model was saved.

---

## "No Anomalies Detected"

Check:

1. The input CSV uses the expected column names.
2. The input values are valid.
3. The data meets the configured anomaly thresholds.
4. The domain-specific rules in `custom_transformers.py` are behaving as expected.
5. The correct trained model is being loaded.

---

## "Directory Not Found"

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

# 26. Testing

Start the Flask application:

```bash
python app.py
```

Then test the energy anomaly endpoint:

```bash
curl -X POST /api/trigger_energy_anomalies
```

For authenticated endpoints, ensure that an authenticated session is being used.

To test network anomaly processing, start the Spark streaming job separately:

```bash
spark-submit streaming_job.py
```

Then check the generated network logs and the corresponding admin interface.

---

# 27. Performance Considerations

Spark performance depends on:

* Number of available CPU cores
* Available RAM
* Dataset size
* Spark configuration
* HDFS performance
* Network configuration
* Model complexity

For local development, using:

```python
.master("local[*]")
```

allows Spark to use the available local CPU cores.

Performance measurements should be taken on the target environment rather than assumed to be identical across different machines.

---

# 28. Production Deployment

For a production deployment, consider:

* Using a production WSGI server rather than Flask's development server
* Setting a strong `SECRET_KEY`
* Replacing default administrator credentials
* Using a dedicated password-hashing mechanism
* Configuring HDFS appropriately
* Moving environment-specific values into environment variables
* Restricting file-upload permissions
* Enabling HTTPS
* Monitoring Spark and HDFS resources
* Keeping credentials and secrets outside the Git repository

The configuration approach in `config.py` allows deployment-specific paths and settings to be changed without modifying the main application source code.

---

# 29. Project Structure

The main application components are:

```text
app.py
```

Main Flask application, authentication, routes, uploads, and dashboard functionality.

```text
config.py
```

Central configuration for local directories, HDFS paths, model paths, datasets, and the Flask secret key.

```text
custom_transformers.py
```

Custom Spark ML transformers containing domain-specific anomaly rules.

```text
auto_scaler.py
```

Auto-scaling utilities used by the application.

```text
enemodel.py
```

Energy anomaly detection model definitions and training functionality.

```text
netmodel.py
```

Network anomaly detection model definitions and training functionality.

```text
streaming_job.py
```

Spark Streaming job for network anomaly detection.

```text
templates/
```

Jinja2 HTML templates used by the Flask application.

```text
static/
```

Static assets used by the web interface.

---

# 30. Before Committing to Git

Check that local and machine-specific files are not committed.

In particular, avoid committing:

```text
venv/
*.db
__pycache__/
logs/
downloads/
checkpoint/
predictions_energy/
energy_input/
network_input/
```

Also ensure that production secrets and credentials are not stored in the repository.

The `config.py` file should contain safe defaults only and should not contain personal machine paths or private credentials.

---

# 31. Future Enhancements

Potential future improvements include:

* LSTM-based time-series anomaly detection
* Isolation Forest ensemble methods
* Real-time dashboard updates using WebSockets
* Docker containerisation
* Kubernetes deployment
* Advanced authentication such as OAuth2 or LDAP
* Model versioning
* A/B model testing
* Explainable AI using SHAP
* Alert notifications
* Multi-tenancy support

---

# 32. Contributing

This project was developed as an academic project.

Improvements, bug fixes, and extensions are welcome.

Before submitting changes:

1. Test the application locally.
2. Check that configuration does not contain machine-specific paths.
3. Do not commit credentials or secrets.
4. Update documentation when functionality changes.
5. Test both energy and network functionality where applicable.

---

# 33. Author

**Daniel Adesanya**

BSc Natural Sciences (Mathematics and Computer Science)
Durham University, 2025

Undergraduate academic project.

---

## Status

**Complete — Academic Project**

**Last Updated:** 2025
