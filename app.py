import os
import glob
import hashlib
import pandas as pd
import subprocess
import requests
import time
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, when, hour, to_timestamp, concat_ws, lit
from pyspark.ml.linalg import DenseVector, SparseVector
from custom_transformers import MultiDomainRuleTransformer, CustomLogisticRegressionModel


from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, send_file
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)

from datetime import timedelta

from config import (
    ENERGY_INPUT_DIR,
    ENERGY_OUTPUT_DIR,
    DOWNLOAD_DIR,
    ENERGY_LOG_DIR,
    NETWORK_LOG_DIR,
    ENERGY_MODEL_PATH,
    SECRET_KEY
)

app = Flask(__name__) 

app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)  # Adjust as needed


LOCAL_ENERGY_OUTPUT = ENERGY_OUTPUT_DIR
csv_folder = ENERGY_INPUT_DIR


# Configuration


app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///myapp.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


os.makedirs(csv_folder, exist_ok=True)

NETWORK_PREDICTIONS_FOLDER = NETWORK_LOG_DIR


# Database Model


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"<User {self.username}>"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Create Default Admin


def create_default_admin():
    existing_admin = User.query.filter_by(username="admin").first()
    if not existing_admin:
        pw_hash = hashlib.sha256("admin".encode("utf-8")).hexdigest()
        admin_user = User(
            username="admin",
            password_hash=pw_hash,
            is_admin=True
        )
        db.session.add(admin_user)
        db.session.commit()
        print("[INIT] Created default admin: admin / admin")
    else:
        print("[INIT] Admin already exists.")


# Routes


@app.route("/")
def index():
    if current_user.is_authenticated:
        return render_template("index.html")
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        is_admin = True if "is_admin" in request.form else False

        if User.query.filter_by(username=username).first():
            flash("Username already taken!")
            return redirect(url_for("register"))

        hashed_pw = hashlib.sha256(password.encode("utf-8")).hexdigest()
        new_user = User(username=username, password_hash=hashed_pw, is_admin=is_admin)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful. Please log in!")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_pw = hashlib.sha256(password.encode("utf-8")).hexdigest()

        user = User.query.filter_by(username=username, password_hash=hashed_pw).first()
        if user:
            login_user(user, remember=True)
            flash("Logged in successfully.")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials!")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.")
    return redirect(url_for("login"))

@app.route("/profile")
@login_required
def profile():
    folder = "/home/dan/project2/energy_input"
    try:
        all_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
        all_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
        recent_csvs = all_files[:5]
    except Exception as e:
        print("[Profile Page File Error]", e)
        recent_csvs = []

    return render_template("profile.html", recent_uploads=recent_csvs)

    
@app.route("/change_password", methods=["POST"])
@login_required
def change_password():
    current_pw = request.form["current_password"]
    new_pw = request.form["new_password"]
    current_hashed = hashlib.sha256(current_pw.encode("utf-8")).hexdigest()

    if current_hashed != current_user.password_hash:
        flash("Current password is incorrect.")
        return redirect(url_for("profile"))

    new_hashed = hashlib.sha256(new_pw.encode("utf-8")).hexdigest()
    current_user.password_hash = new_hashed
    db.session.commit()
    flash("Password changed successfully.")
    return redirect(url_for("profile"))



# Normal-user Energy Upload Page

@app.route("/upload_energy", methods=["GET", "POST"])
@login_required
def upload_energy():
    """
    This route displays a file-upload form (upload_energy.html).
    When the user uploads a CSV file (which has a label column),
    the app saves it, calls the detect_energy_anomalies function to load
    the pre-trained pipeline model and predict anomalies, and then renders
    the same template with the predicted anomaly rows.
    """
    anomalies = None  # Will hold anomaly data for display

    if request.method == "POST":
        if "csvfile" not in request.files:
            flash("No file part in request.")
            return redirect(request.url)

        file = request.files["csvfile"]
        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        # Save the file (use the original filename)
        save_path = os.path.join(csv_folder, file.filename)
        file.save(save_path)
        flash(f"File '{file.filename}' uploaded successfully.")

        # Call the function that loads the model, runs predictions,
        # and returns only the rows that the model predicts as anomalies.
        anomalies, _ = detect_energy_anomalies(save_path)

        if anomalies:
            flash(f"Detected {len(anomalies)} anomaly rows.")
        else:
            flash("No anomalies predicted in this file.")

    # Render the upload_energy.html template without renaming it.
    return render_template("upload_energy.html", anomalies=anomalies)


# Helper Function: Detect Energy Anomalies


MODEL_PATH = ENERGY_MODEL_PATH
THRESHOLD = 75


def detect_energy_anomalies(csv_path):
    """
    Detects energy anomalies using the custom Spark ML pipeline.
    Logs processing time and saves anomaly output and performance data.
    """
    MODEL_PATH = ENERGY_MODEL_PATH
    THRESHOLD = 75
    LOG_DIR = ENERGY_LOG_DIR
    os.makedirs(LOG_DIR, exist_ok=True)

    try:
        start_time = time.time()

        spark = SparkSession.builder.appName("EnergyBatchInference").getOrCreate()

        # Define schema
        energy_schema = StructType([
            StructField("timestamp", StringType(), True),
            StructField("energy_consumption", DoubleType(), True),
            StructField("voltage", DoubleType(), True),
            StructField("current", DoubleType(), True),
            StructField("power", DoubleType(), True),
            StructField("solar_output", DoubleType(), True),
            StructField("frequency", DoubleType(), True),
            StructField("label", StringType(), True)
        ])

        # Load data
        df = spark.read.csv(csv_path, header=True, schema=energy_schema).fillna(0)
        df = df.withColumn("ts", to_timestamp(col("timestamp"), "dd/MM/yyyy HH:mm:ss"))
        df = df.withColumn("hour", hour(col("ts")))

        # Load model and make predictions
        model = PipelineModel.load(MODEL_PATH)
        predictions = model.transform(df)

        # Convert prediction to label
        predictions = predictions.withColumn(
            "predicted_label",
            when(col("prediction") == 0, "Anomaly").otherwise("Normal")
        )

        # Add domain-based demand anomaly detection
        predictions = predictions.withColumn(
            "demand_anomaly",
            when(
                (
                    ((col("hour") >= 12) & (col("hour") < 17)) |
                    (col("hour") < 5) | (col("hour") >= 23)
                ) & (col("energy_consumption") > THRESHOLD),
                "Anomaly"
            ).otherwise("Normal")
        )

        # Add explanation reason for domain rule-based flags
        predictions = predictions.withColumn(
            "reason",
            concat_ws("; ",
                when(col("timeFlag") == 1, lit("High consumption during off-peak hours")).otherwise(lit("")),
                when(col("voltageFlag") == 1, lit("Abnormal voltage")).otherwise(lit("")),
                when(col("solarFlag") == 1, lit("Suspicious solar output")).otherwise(lit("")),
                when(col("freqFlag") == 1, lit("Frequency deviation")).otherwise(lit(""))
            )
        )

        # Filter to relevant anomalies
        anomalies_df = predictions.filter(
            (col("predicted_label") == "Anomaly") |
            (col("demand_anomaly") == "Anomaly")
        )

        # Convert DenseVector fields to serializable lists
        def convert_row(row):
            row_dict = row.asDict()
            for k, v in row_dict.items():
                if isinstance(v, (DenseVector, SparseVector)):
                    row_dict[k] = list(v)
            return row_dict

        anomaly_list = [convert_row(row) for row in anomalies_df.collect()]

        # Save anomaly output to CSV
        base = os.path.basename(csv_path)
        name, _ = os.path.splitext(base)
        output_path = os.path.join(DOWNLOAD_DIR, f"{name}_anomalies.csv")
        anomalies_df.toPandas().to_csv(output_path, index=False)

        # === METRICS LOGGING ===
        processing_time = time.time() - start_time
        log_data = {
            "timestamp_logged": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": os.path.basename(csv_path),
            "processing_time_sec": round(processing_time, 2)
        }
        log_df = pd.DataFrame([log_data])
        log_filename = os.path.join(LOG_DIR, f"{name}_log.csv")
        log_df.to_csv(log_filename, index=False)

        print(f"[METRIC] Energy file processed in {processing_time:.2f} seconds")
        print(f"[LOG] Saved processing log to {log_filename}")

        spark.stop()
        return anomaly_list, os.path.basename(output_path)

    except Exception as e:
        print("[Error in detect_energy_anomalies]:", e)
        return []



@app.route("/api/energy_anomalies", methods=["GET"])
@login_required
def api_energy_anomalies_auto():
    """
    Automatically scans the energy_input folder,
    picks the most recent CSV file, and returns anomaly results.
    """
    try:
        # Get list of CSV files
        files = glob.glob(os.path.join(csv_folder, "*.csv"))
        if not files:
            return jsonify({"error": "No CSV files found in input folder."}), 404

        # Sort by last modified time (latest file last)
        latest_file = max(files, key=os.path.getmtime)

        # Run detection on the latest file
        anomalies = detect_energy_anomalies(latest_file)
        return jsonify(anomalies), 200

    except Exception as e:
        print("[API Error]", e)
        return jsonify({"error": "An error occurred.", "details": str(e)}), 500


@app.route("/energy_auto")
@login_required
def energy_auto_page():
    try:
        response = requests.get("http://localhost:5000/api/energy_anomalies_all")
        if response.status_code == 200:
            #xtract the list correctly!
            anomalies = response.json().get("all_anomalies", [])
        else:
            anomalies = []
    except Exception as e:
        print("[Error fetching auto anomalies]", e)
        anomalies = []

    return render_template("energy_auto_detect.html", anomalies=anomalies)


@app.route("/api/trigger_energy_anomalies", methods=["POST"])
@login_required
def trigger_energy_anomalies():
    try:
        files = glob.glob(os.path.join(csv_folder, "*.csv"))
        if not files:
            return jsonify({"error": "No CSV files found."}), 404

        latest_file = max(files, key=os.path.getmtime)
        anomalies, download_path = detect_energy_anomalies(latest_file)
        # Only send filename (not path) to browser for security
        filename = os.path.basename(download_path)
        return jsonify({"anomalies": anomalies, "filename": filename}), 200
    except Exception as e:
        print("[Error triggering detection]", e)
        return jsonify({"error": "Something went wrong.", "details": str(e)}), 500


@app.route("/download_energy_anomalies/<filename>")
@login_required
def download_energy_anomalies(filename):
    download_dir = DOWNLOAD_DIR
    filepath = os.path.join(download_dir, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        flash("No anomaly file available for download.")
        return redirect(url_for("energy_auto_page"))

@app.route("/api/energy_anomalies_all", methods=["GET"])
@login_required
def api_energy_anomalies_all():
    try:
        files = glob.glob(os.path.join(csv_folder, "*.csv"))
        if not files:
            return jsonify({"error": "No CSV files found in input folder."}), 404

        all_anomalies = []
        for csv_file in sorted(files, key=os.path.getmtime, reverse=True):
            try:
                anomalies, _ = detect_energy_anomalies(csv_file)
                all_anomalies.extend(anomalies)
            except Exception as e:
                print(f"[File Skipped: {csv_file}]", e)
                continue

        return jsonify({"all_anomalies": all_anomalies}), 200
    except Exception as e:
        print("[API Error - all files]", e)
        return jsonify({"error": "Something went wrong", "details": str(e)}), 500




# Admin-only Network Anomalies Page

@app.route("/admin_network")
@login_required
def admin_network():
    if not current_user.is_admin:
        flash("Access denied: Admin only.")
        return redirect(url_for("index"))

    anomalies = get_streamed_network_anomalies()
    return render_template("admin_network.html", anomalies=anomalies)


def get_streamed_network_anomalies():
    """
    Loads anomaly rows from the predictions_network folder generated by the Spark streaming job.
    Only returns rows with 'predicted_label' == 'Anomaly'.
    """
    try:
        prediction_files = glob.glob(os.path.join(NETWORK_PREDICTIONS_FOLDER, "*.csv"))
        if not prediction_files:
            return []

        # Read and combine all CSV files
        df_list = [pd.read_csv(f) for f in prediction_files if os.path.isfile(f)]
        combined_df = pd.concat(df_list, ignore_index=True)

        # Filter for anomalies
        if "predicted_label" not in combined_df.columns:
            return []

        anomaly_df = combined_df[combined_df["predicted_label"] == "Anomaly"]

        return anomaly_df.to_dict(orient="records")
    except Exception as e:
        print("[Error loading streamed network anomalies]:", e)
        return []


# Main


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        create_default_admin()
    app.run(debug=True)
