import os

BASE_DIR = os.getenv("PROJECT_BASE_DIR", os.getcwd())

ENERGY_INPUT_DIR = os.getenv(
    "ENERGY_INPUT_DIR",
    os.path.join(BASE_DIR, "energy_input")
)

ENERGY_OUTPUT_DIR = os.getenv(
    "ENERGY_OUTPUT_DIR",
    os.path.join(BASE_DIR, "predictions_energy")
)

DOWNLOAD_DIR = os.getenv(
    "DOWNLOAD_DIR",
    os.path.join(BASE_DIR, "downloads")
)

ENERGY_LOG_DIR = os.getenv(
    "ENERGY_LOG_DIR",
    os.path.join(BASE_DIR, "logs", "energy_processing_logs")
)

NETWORK_LOG_DIR = os.getenv(
    "NETWORK_LOG_DIR",
    os.path.join(BASE_DIR, "logs", "network_stream_logs")
)

MODEL_HOST = os.getenv("MODEL_HOST", "localhost")
MODEL_PORT = os.getenv("MODEL_PORT", "9000")

ENERGY_MODEL_PATH = os.getenv(
    "ENERGY_MODEL_PATH",
    f"hdfs://{MODEL_HOST}:{MODEL_PORT}/models/oop_custom_energy_model"
)

NETWORK_MODEL_PATH = os.getenv(
    "NETWORK_MODEL_PATH",
    f"hdfs://{MODEL_HOST}:{MODEL_PORT}/models/oop_custom_network_model"
)

NETWORK_INPUT_DIR = os.getenv(
    "NETWORK_INPUT_DIR",
    os.path.join(BASE_DIR, "network_input")
)

SECRET_KEY = os.getenv("SECRET_KEY", "development-secret-key")