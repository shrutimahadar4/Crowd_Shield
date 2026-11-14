import sqlite3

def connect_db():
    return sqlite3.connect("crowdshield.db")


def create_tables():
    conn = connect_db()
    cursor = conn.cursor()

    # Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            email TEXT
        )
    """)

    # Crowd Logs Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crowd_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            live_count INTEGER,
            capacity INTEGER,
            status TEXT
        )
    """)

    # Alerts Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            alert_type TEXT,
            message TEXT
        )
    """)

    # Emergency Snapshots Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emergency_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT UNIQUE,
            timestamp TEXT,
            location_id TEXT,
            severity TEXT,
            capacity_limit INTEGER,
            current_count INTEGER,
            snapshot_path TEXT,
            resolution_status TEXT
        )
    """)

    conn.commit()
    conn.close()


def add_user(username, password, email):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
            (username, password, email)
        )
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()


def check_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )
    user = cursor.fetchone()
    conn.close()
    return user is not None


def insert_crowd_log(timestamp, live_count, capacity, status):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO crowd_logs (timestamp, live_count, capacity, status) VALUES (?, ?, ?, ?)",
        (timestamp, live_count, capacity, status)
    )
    conn.commit()
    conn.close()


def insert_alert(timestamp, alert_type, message):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO alerts (timestamp, alert_type, message) VALUES (?, ?, ?)",
        (timestamp, alert_type, message)
    )
    conn.commit()
    conn.close()

def insert_emergency_snapshot(alert_id, timestamp, location_id, severity, capacity_limit, current_count, snapshot_path, resolution_status="Pending"):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO emergency_snapshots 
               (alert_id, timestamp, location_id, severity, capacity_limit, current_count, snapshot_path, resolution_status) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (alert_id, timestamp, location_id, severity, capacity_limit, current_count, snapshot_path, resolution_status)
        )
        conn.commit()
    except Exception as e:
        print(f"Error inserting snapshot: {e}")
    finally:
        conn.close()

def get_emergency_snapshots(limit=30):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT alert_id, timestamp, location_id, severity, capacity_limit, current_count, snapshot_path, resolution_status
           FROM emergency_snapshots 
           ORDER BY id DESC LIMIT ?""",
        (limit,)
    )
    records = cursor.fetchall()
    conn.close()
    
    # Standardize as list of dicts for easy Streamlit consumption
    results = []
    for row in records:
        results.append({
            "alert_id": row[0],
            "timestamp": row[1],
            "location_id": row[2],
            "severity": row[3],
            "capacity_limit": row[4],
            "current_count": row[5],
            "snapshot_path": row[6],
            "resolution_status": row[7]
        })
    return results

def delete_all_snapshots():
    conn = connect_db()
    cursor = conn.cursor()
    
    # First get all paths so we can delete the actual files
    cursor.execute("SELECT snapshot_path FROM emergency_snapshots")
    paths = [row[0] for row in cursor.fetchall()]
    
    # Then delete all records from database
    cursor.execute("DELETE FROM emergency_snapshots")
    conn.commit()
    conn.close()
    
    return paths