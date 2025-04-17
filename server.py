# file server.py

import json
import sqlite3
import logging
import threading
import queue
import time
from functools import wraps

import cv2
import numpy as np
import pytz
import torch
import os
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, jsonify, Response, session, flash, redirect, url_for, \
    send_from_directory
from auth import admin_required, login_required
from sort.sort import Sort
from ultralytics import YOLO
from util import read_license_plate, get_car, is_plate_inside_car, get_plate_center, get_car_center

app = Flask(__name__)

os.environ['TZ'] = 'Europe/Moscow'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app.secret_key = 'your-secret-key-here'
RECORDINGS_DIR = 'recordings'
SNAPSHOTS_DIR = 'snapshots'

# Создаем директории, если они не существуют
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# Configuration
CONFIG_FILE = 'config.json'
DEFAULT_CONFIG = {
    'cameras': [],
    'current_camera': None,
    'threshold': 0.85
}

camera_states = {}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()


# Global variables
config = load_config()
coco_model = None
license_plate_detector = None
mot_tracker = None
vehicles = [2, 3, 5, 7]
tracked_plates = {}
camera_instances = {}  # {camera_id: {'cap': cv2.VideoCapture, 'frame_queue': queue.Queue, 'thread': threading.Thread, 'stop_event': threading.Event}}
main_cap = None
main_processing_thread = None
stop_event = threading.Event()


def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)


def init_db():
    conn = sqlite3.connect('plates.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS recognized_plates
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  plate_text TEXT NOT NULL,
                  camera_id INTEGER,
                  camera_name TEXT,
                  score REAL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  image BLOB)''')

    # Индекс для быстрой проверки дубликатов
    c.execute('''CREATE INDEX IF NOT EXISTS idx_plate_text_time 
                 ON recognized_plates(plate_text, timestamp)''')

    conn.commit()
    conn.close()

    # Инициализация базы данных пользователей
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Проверяем существование таблицы
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = c.fetchone()

    if table_exists:
        # Проверяем существование колонки created_at
        c.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in c.fetchall()]

        if 'created_at' not in columns:
            # Создаем временную таблицу с новой структурой
            c.execute('''CREATE TABLE IF NOT EXISTS users_new
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT UNIQUE NOT NULL,
                          password TEXT NOT NULL,
                          role TEXT DEFAULT 'user',
                          created_at DATETIME DEFAULT (datetime('now')))''')

            # Копируем данные из старой таблицы
            c.execute(
                "INSERT INTO users_new (id, username, password, role) SELECT id, username, password, role FROM users")

            # Удаляем старую таблицу
            c.execute("DROP TABLE users")

            # Переименовываем новую таблицу
            c.execute("ALTER TABLE users_new RENAME TO users")
    else:
        # Создаем таблицу с нужными колонками
        c.execute('''CREATE TABLE users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      role TEXT DEFAULT 'user',
                      created_at DATETIME DEFAULT (datetime('now')))''')

    # Создаем администратора по умолчанию, если его нет
    c.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if c.fetchone()[0] == 0:
        hashed_password = generate_password_hash('admin')
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                  ('admin', hashed_password, 'admin'))

    conn.commit()
    conn.close()


def insert_plate_data(plate_text, image, camera_id, camera_name, score):
    """Вставляет данные о распознанном номере с проверкой на дубликаты за последние 5 минут"""
    moscow_tz = pytz.timezone('Europe/Moscow')
    current_time = datetime.now(moscow_tz)

    conn = sqlite3.connect('plates.db')
    c = conn.cursor()

    # Проверяем, был ли такой номер зарегистрирован в последние 5 минут
    five_minutes_ago = current_time - timedelta(minutes=5)
    c.execute("""
        SELECT id FROM recognized_plates 
        WHERE plate_text = ? AND timestamp >= ?
        LIMIT 1
    """, (plate_text, five_minutes_ago.strftime('%Y-%m-%d %H:%M:%S')))

    if c.fetchone():
        conn.close()
        logging.info(f"Номер {plate_text} уже был записан в последние 5 минут - пропускаем")
        return False

    # Удаляем записи старше 7 дней
    week_ago = current_time - timedelta(days=7)
    c.execute("DELETE FROM recognized_plates WHERE timestamp < ?",
              (week_ago.strftime('%Y-%m-%d %H:%M:%S'),))

    # Вставляем новую запись
    c.execute("""
        INSERT INTO recognized_plates 
        (plate_text, camera_id, camera_name, score, timestamp, image) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (plate_text, camera_id, camera_name, score,
          current_time.strftime('%Y-%m-%d %H:%M:%S'), image))

    conn.commit()
    conn.close()
    logging.info(f"Новый номер {plate_text} успешно записан")
    return True

def get_moscow_time():
    return datetime.now(pytz.timezone('Europe/Moscow'))


def get_recent_plates(limit=100):
    """Возвращает список последних распознанных номеров с московским временем"""
    conn = sqlite3.connect('plates.db')
    c = conn.cursor()
    c.execute("""
        SELECT id, plate_text, camera_id, camera_name, score, 
               datetime(timestamp, 'localtime') as local_timestamp 
        FROM recognized_plates 
        ORDER BY timestamp DESC 
        LIMIT ?""", (limit,))

    plates = []
    for row in c.fetchall():
        plates.append({
            'id': row[0],
            'text': row[1],
            'camera_id': row[2],
            'camera_name': row[3],
            'score': row[4],
            'timestamp': row[5]
        })
    conn.close()
    return plates

def get_plate_count_last_24h():
    """Возвращает количество распознанных номеров за последние 24 часа"""
    conn = sqlite3.connect('plates.db')
    c = conn.cursor()
    day_ago = datetime.now() - timedelta(days=1)
    c.execute("SELECT COUNT(*) FROM recognized_plates WHERE timestamp > ?", (day_ago.strftime('%Y-%m-%d %H:%M:%S'),))
    count = c.fetchone()[0]
    conn.close()
    return count


def get_all_users():
    """Возвращает список всех пользователей"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Проверяем существование колонки created_at
    c.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in c.fetchall()]

    if 'created_at' in columns:
        c.execute("SELECT id, username, role, created_at FROM users ORDER BY created_at DESC")
    else:
        c.execute("SELECT id, username, role FROM users")

    users = []
    for row in c.fetchall():
        user = {
            'id': row[0],
            'username': row[1],
            'role': row[2]
        }
        if len(row) > 3:  # Если есть created_at
            user['created_at'] = row[3]
        else:
            user['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        users.append(user)

    conn.close()
    return users


@app.route('/api/camera_status/<int:camera_id>')
@login_required
def get_camera_status(camera_id):
    """Возвращает текущий статус камеры"""
    status = camera_states.get(camera_id, {'connected': False})
    return jsonify(status)


@app.route('/api/plate_image/<int:plate_id>')
@login_required
def get_plate_image(plate_id):
    """Возвращает изображение с распознанным номером"""
    conn = sqlite3.connect('plates.db')
    c = conn.cursor()
    c.execute("SELECT image FROM recognized_plates WHERE id = ?", (plate_id,))
    image_data = c.fetchone()
    conn.close()

    if image_data:
        return Response(image_data[0], mimetype='image/jpeg')
    return jsonify({'status': 'error', 'message': 'Image not found'}), 404

def update_user_role(user_id, new_role):
    """Обновляет роль пользователя"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
    conn.commit()
    conn.close()


def delete_user(user_id):
    """Удаляет пользователя"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def update_user_password(user_id, new_password):
    """Обновляет пароль пользователя"""
    hashed_password = generate_password_hash(new_password)
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user_id))
    conn.commit()
    conn.close()


def initialize_models():
    global coco_model, license_plate_detector, mot_tracker
    coco_model = YOLO('models/yolo11n.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
    license_plate_detector = YOLO('models/license_plate_detector.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
    mot_tracker = Sort()


@app.route('/api/start_recording/<int:camera_id>', methods=['POST'])
@admin_required
def start_recording(camera_id):
    if camera_id not in camera_instances:
        return jsonify({'status': 'error', 'message': 'Camera not connected'}), 404

    camera_instances[camera_id]['recording'] = True
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"recording_{camera_id}_{timestamp}.avi"
    filepath = os.path.join(RECORDINGS_DIR, filename)

    # Создаем VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30  # Можно получить из камеры
    frame_size = (int(camera_instances[camera_id]['cap'].get(3)),
                  int(camera_instances[camera_id]['cap'].get(4)))
    out = cv2.VideoWriter(filepath, fourcc, fps, frame_size)

    camera_instances[camera_id]['video_writer'] = out
    camera_instances[camera_id]['recording_file'] = filepath

    return jsonify({
        'status': 'success',
        'message': 'Recording started',
        'filename': filename
    })


@app.route('/api/stop_recording/<int:camera_id>', methods=['POST'])
@admin_required
def stop_recording(camera_id):
    if camera_id not in camera_instances or not camera_instances[camera_id].get('recording'):
        return jsonify({'status': 'error', 'message': 'No active recording'}), 404

    camera_instances[camera_id]['recording'] = False
    if 'video_writer' in camera_instances[camera_id]:
        camera_instances[camera_id]['video_writer'].release()
        del camera_instances[camera_id]['video_writer']

    return jsonify({
        'status': 'success',
        'message': 'Recording stopped',
        'filename': camera_instances[camera_id].get('recording_file', '')
    })


@app.route('/api/take_snapshot/<int:camera_id>', methods=['POST'])
@login_required
def take_snapshot(camera_id):
    if camera_id not in camera_instances:
        return jsonify({'status': 'error', 'message': 'Camera not connected'}), 404

    try:
        # Получаем последний кадр из очереди
        frame_data = camera_instances[camera_id]['frame_queue'].get_nowait()
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"snapshot_{camera_id}_{timestamp}.jpg"
        filepath = os.path.join(SNAPSHOTS_DIR, filename)

        cv2.imwrite(filepath, frame)

        return jsonify({
            'status': 'success',
            'message': 'Snapshot saved',
            'filename': filename,
            'path': f"/snapshots/{filename}"
        })
    except queue.Empty:
        return jsonify({'status': 'error', 'message': 'No frames available'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/snapshots/<filename>')
@login_required
def serve_snapshot(filename):
    return send_from_directory(SNAPSHOTS_DIR, filename)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT id, password, role FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            session['role'] = user[2]
            flash('Logged in successfully')
            next_page = request.args.get('next') or url_for('index')
            return redirect(next_page)
        else:
            flash('Invalid username or password')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully')
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                      (username, hashed_password))
            conn.commit()
            conn.close()
            flash('Registration successful. Please login.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists')
            return redirect(url_for('register'))

    return render_template('register.html')


def process_frame(frame, camera_id=None):
    try:
        start_time = time.time()

        # Vehicle detection
        vehicle_detections = coco_model(frame)[0]
        vehicle_boxes = []
        for detection in vehicle_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                vehicle_boxes.append([x1, y1, x2, y2, score])

        # Vehicle tracking
        track_ids = mot_tracker.update(np.asarray(vehicle_boxes)) if vehicle_boxes else []
        current_time = time.time()

        # License plate detection
        license_detections = license_plate_detector(frame)[0]

        # Update existing tracks
        updated_vehicles = set()
        for track in track_ids:
            xcar1, ycar1, xcar2, ycar2, track_id = track
            car_bbox = (xcar1, ycar1, xcar2, ycar2)

            if track_id in tracked_plates:
                # Check for new plates for this vehicle
                new_plate = None
                for lp in license_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, _ = lp
                    if is_plate_inside_car((x1, y1, x2, y2), car_bbox):
                        plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        plate_text, plate_score = read_license_plate(plate_crop)
                        if plate_text and plate_score >= config['threshold']:
                            new_plate = (plate_text, plate_score)
                            break

                # Update or keep existing plate
                if new_plate:
                    tracked_plates[track_id] = {
                        'plate_text': new_plate[0],
                        'plate_score': new_plate[1],
                        'last_seen': current_time
                    }
                else:
                    tracked_plates[track_id]['last_seen'] = current_time
                updated_vehicles.add(track_id)

        # Process new plates for untracked vehicles
        for lp in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, _ = lp
            plate_bbox = (x1, y1, x2, y2)
            plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            plate_text, plate_score = read_license_plate(plate_crop)

            if not plate_text or plate_score < config['threshold']:
                continue

            # Find closest vehicle without plate
            best_match = None
            min_distance = float('inf')
            for track in track_ids:
                xcar1, ycar1, xcar2, ycar2, track_id = track
                if track_id in updated_vehicles:
                    continue
                car_bbox = (xcar1, ycar1, xcar2, ycar2)
                if is_plate_inside_car(plate_bbox, car_bbox):
                    distance = np.linalg.norm(
                        np.array(get_plate_center(plate_bbox)) -
                        np.array(get_car_center(car_bbox)))
                    if distance < min_distance:
                        min_distance = distance
                        best_match = track_id

            if best_match:
                tracked_plates[best_match] = {
                    'plate_text': plate_text,
                    'plate_score': plate_score,
                    'last_seen': current_time
                }
                updated_vehicles.add(best_match)

                # Сохраняем только если номер новый (без дубликатов за 5 минут)
                if plate_text != getattr(app, f'last_saved_plate_{camera_id}', None):
                    try:
                        _, buffer = cv2.imencode('.jpg', frame)
                        camera_name = next((cam['name'] for cam in config['cameras'] if cam['id'] == camera_id),
                                           'Unknown')

                        # Пытаемся вставить (функция сама проверит дубликаты)
                        if insert_plate_data(plate_text, buffer.tobytes(), camera_id, camera_name, plate_score):
                            setattr(app, f'last_saved_plate_{camera_id}', plate_text)
                    except Exception as e:
                        logging.error(f"Ошибка сохранения: {e}")

        # Clean old tracks (>5 seconds without update)
        to_delete = [tid for tid, plate in tracked_plates.items()
                     if current_time - plate['last_seen'] > 5.0]
        for tid in to_delete:
            del tracked_plates[tid]

        # Draw annotations
        for track in track_ids:
            x1, y1, x2, y2, track_id = track
            plate_info = tracked_plates.get(track_id)

            if plate_info:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                text = f"{plate_info['plate_text']} ({plate_info['plate_score']:.2f})"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame,
                              (int(x1), int(y1) - text_height - 10),
                              (int(x1) + text_width + 10, int(y1)),
                              (0, 0, 255), -1)
                cv2.putText(frame, text,
                            (int(x1) + 5, int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

        # Calculate and display FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame
    except Exception as e:
        logging.error(f"Error in process_frame: {e}")
        return frame


def camera_processing(camera_id, camera_url, stop_event):
    global camera_states
    camera_states[camera_id] = {'connected': False}

    try:

        cap = cv2.VideoCapture(camera_url)
        camera_states[camera_id]['connected'] = cap.isOpened()
        frame_queue = queue.Queue(maxsize=2)
        # Wait for camera to initialize
        retries = 0
        while not cap.isOpened() and retries < 10:
            time.sleep(0.1)
            retries += 1

        if not cap.isOpened():
            logging.error(f"Failed to open camera {camera_id} at {camera_url}")
            return

        camera_instances[camera_id] = {
            'cap': cap,
            'frame_queue': frame_queue,
            'stop_event': stop_event,
            'thread': threading.current_thread(),
            'recording': False,
            'video_writer': None
        }

        logging.info(f"Started processing for camera {camera_id}")

        while not stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                try:
                    processed_frame = process_frame(frame, camera_id)

                    # Если идет запись, сохраняем кадр
                    if camera_instances[camera_id].get('recording') and 'video_writer' in camera_instances[camera_id]:
                        camera_instances[camera_id]['video_writer'].write(processed_frame)

                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    frame_queue.put(buffer.tobytes())
                except queue.Full:
                    pass  # Skip frame if queue is full
                except Exception as e:
                    logging.error(f"Error processing frame for camera {camera_id}: {e}")
            else:
                logging.warning(f"Camera {camera_id} returned no frame")
                time.sleep(0.1)

        # Останавливаем запись при завершении
        if 'video_writer' in camera_instances[camera_id]:
            camera_instances[camera_id]['video_writer'].release()

        cap.release()
        if camera_id in camera_instances:
            del camera_instances[camera_id]
        logging.info(f"Stopped processing for camera {camera_id}")
    finally:
        camera_states[camera_id]['connected'] = False


def gen_camera_frames(camera_id):
    while True:
        if camera_id in camera_instances:
            try:
                frame_data = camera_instances[camera_id]['frame_queue'].get(timeout=1)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            except queue.Empty:
                continue
        else:
            time.sleep(0.1)


def generate_combined_feed():
    while True:
        frames = []
        for camera_id, instance in camera_instances.items():
            try:
                frame_data = instance['frame_queue'].get(timeout=0.1)
                frames.append((camera_id, frame_data))
            except queue.Empty:
                continue

        if frames:
            # Here you can implement logic to combine frames or select one to display
            # For simplicity, we'll just show the first available frame
            camera_id, frame_data = frames[0]
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        else:
            time.sleep(0.1)


@app.route('/')
def index():
    return render_template('index.html', config=config, user=session.get('username'), role=session.get('role'))


@app.route('/video_feed')
def video_feed():
    if config['current_camera'] is not None and config['current_camera'] in camera_instances:
        return Response(gen_camera_frames(config['current_camera']),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif camera_instances:
        # If no current camera but we have connected cameras, show the first one
        first_camera_id = next(iter(camera_instances))
        return Response(gen_camera_frames(first_camera_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Fallback to empty response
        return Response(b'',
                        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera/<int:camera_id>')
def camera_view(camera_id):
    camera = next((cam for cam in config['cameras'] if cam['id'] == camera_id), None)
    if not camera:
        return "Camera not found", 404
    return render_template('camera.html')


@app.route('/video_feed/<int:camera_id>')
def camera_video_feed(camera_id):
    return Response(gen_camera_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/camera_info/<int:camera_id>')
def get_camera_info(camera_id):
    camera = next((cam for cam in config['cameras'] if cam['id'] == camera_id), None)
    if not camera:
        return jsonify({'status': 'error', 'message': 'Camera not found'}), 404
    return jsonify({
        'status': 'success',
        'name': camera['name'],
        'url': camera['url']
    })


@app.route('/api/camera_urls', methods=['GET'])
def get_camera_urls():
    return jsonify({
        'cameras': [
            {'id': cam['id'], 'name': cam['name'], 'url': cam['url']}
            for cam in config['cameras']
        ]
    })


@app.route('/api/check_admin')
@login_required
def check_admin():
    return jsonify({'is_admin': session.get('role') == 'admin'})


@app.route('/api/recent_plates')
@login_required
def recent_plates():
    plates = get_recent_plates()
    count = get_plate_count_last_24h()

    # Добавляем информацию о дубликатах
    conn = sqlite3.connect('plates.db')
    c = conn.cursor()
    c.execute('''
        SELECT plate_text, COUNT(*) as dup_count 
        FROM recognized_plates 
        WHERE timestamp >= datetime('now', '-24 hours')
        GROUP BY plate_text
        HAVING dup_count > 1
    ''')
    duplicates = {row[0]: row[1] for row in c.fetchall()}
    conn.close()

    # Добавляем флаг дубликата в данные
    for plate in plates:
        plate['is_duplicate'] = duplicates.get(plate['text'], 0) > 1

    return jsonify({
        'plates': plates,
        'count': count,
        'duplicates_count': len(duplicates)
    })


@app.route('/api/users', methods=['GET', 'POST', 'PUT', 'DELETE'])
@admin_required
def manage_users():
    if request.method == 'GET':
        users = get_all_users()
        return jsonify({'users': users})

    elif request.method == 'POST':
        data = request.json
        username = data.get('username')
        password = data.get('password')
        role = data.get('role', 'user')

        if not username or not password:
            return jsonify({'status': 'error', 'message': 'Username and password are required'}), 400

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                      (username, hashed_password, role))
            conn.commit()
            user_id = c.lastrowid
            conn.close()

            return jsonify({
                'status': 'success',
                'user': {
                    'id': user_id,
                    'username': username,
                    'role': role
                }
            })
        except sqlite3.IntegrityError:
            return jsonify({'status': 'error', 'message': 'Username already exists'}), 400

    elif request.method == 'PUT':
        data = request.json
        user_id = data.get('id')
        new_role = data.get('role')
        new_password = data.get('password')

        if not user_id:
            return jsonify({'status': 'error', 'message': 'User ID is required'}), 400

        if new_role:
            update_user_role(user_id, new_role)

        if new_password:
            update_user_password(user_id, new_password)

        return jsonify({'status': 'success'})

    elif request.method == 'DELETE':
        data = request.json
        user_id = data.get('id')

        if not user_id:
            return jsonify({'status': 'error', 'message': 'User ID is required'}), 400

        if user_id == session.get('user_id'):
            return jsonify({'status': 'error', 'message': 'Cannot delete yourself'}), 400

        delete_user(user_id)
        return jsonify({'status': 'success'})


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html', config=config, user=session.get('username'), role=session.get('role'))


@app.route('/recognized_plates')
@login_required
def recognized_plates():
    return render_template('index.html', config=config, user=session.get('username'), role=session.get('role'))


@app.route('/settings')
@admin_required
def settings():
    return render_template('index.html', config=config, user=session.get('username'), role=session.get('role'))


@app.route('/user_management')
@admin_required
def user_management():
    return render_template('index.html', config=config, user=session.get('username'), role=session.get('role'))


@app.route('/statistics')
@login_required
def statistics():
    return render_template('index.html', config=config, user=session.get('username'), role=session.get('role'))


@app.route('/event_history')
@login_required
def event_history():
    return render_template('index.html', config=config, user=session.get('username'), role=session.get('role'))


@app.route('/api/cameras', methods=['GET', 'POST', 'DELETE'])
@admin_required
def manage_cameras():
    if request.method == 'GET':
        return jsonify({'cameras': config['cameras']})

    elif request.method == 'POST':
        data = request.json
        if 'url' not in data or 'name' not in data:
            return jsonify({'status': 'error', 'message': 'Missing url or name'}), 400

        if any(cam['url'] == data['url'] for cam in config['cameras']):
            return jsonify({'status': 'error', 'message': 'Camera with this URL already exists'}), 400

        camera_id = len(config['cameras'])
        new_camera = {
            'id': camera_id,
            'name': data['name'],
            'url': data['url']
        }
        config['cameras'].append(new_camera)
        save_config(config)

        return jsonify({'status': 'success', 'camera': new_camera})

    elif request.method == 'DELETE':
        data = request.json
        if 'id' not in data:
            return jsonify({'status': 'error', 'message': 'Missing camera id'}), 400

        config['cameras'] = [cam for cam in config['cameras'] if cam['id'] != data['id']]
        save_config(config)

        if config['current_camera'] == data['id']:
            disconnect_camera()
            config['current_camera'] = None
            save_config(config)

        return jsonify({'status': 'success'})


@app.route('/api/connect', methods=['POST'])
@admin_required
def connect_camera():
    global main_cap, main_processing_thread
    data = request.json

    if 'id' not in data:
        return jsonify({'status': 'error', 'message': 'Missing camera id'}), 400

    camera = next((cam for cam in config['cameras'] if cam['id'] == data['id']), None)
    if not camera:
        return jsonify({'status': 'error', 'message': 'Camera not found'}), 404

    # Check if this camera is already being processed
    if data['id'] in camera_instances:
        config['current_camera'] = data['id']
        save_config(config)
        return jsonify({'status': 'success', 'camera': camera})

    # Start processing this camera
    stop_event = threading.Event()
    thread = threading.Thread(
        target=camera_processing,
        args=(data['id'], camera['url'], stop_event))
    thread.start()

    # Wait a bit for the camera to initialize
    time.sleep(1)

    if data['id'] in camera_instances:
        config['current_camera'] = data['id']
        save_config(config)
        return jsonify({'status': 'success', 'camera': camera})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start camera processing'}), 500


@app.route('/api/disconnect', methods=['POST'])
@admin_required
def disconnect_camera():
    data = request.json
    if 'id' not in data:
        return jsonify({'status': 'error', 'message': 'Missing camera id'}), 400

    camera_id = data['id']
    if camera_id in camera_instances:
        instance = camera_instances[camera_id]
        instance['stop_event'].set()
        if 'thread' in instance:
            instance['thread'].join()
        return jsonify({'status': 'success'})

    return jsonify({'status': 'error', 'message': 'Camera not found'}), 404


@app.route('/api/plates', methods=['GET'])
def get_recognized_plates():
    plates = [{'text': plate['plate_text'], 'score': plate['plate_score']}
              for plate in tracked_plates.values()]
    return jsonify({'plates': plates})


@app.route('/api/threshold', methods=['GET', 'POST'])
@admin_required
def manage_threshold():
    if request.method == 'GET':
        return jsonify({'threshold': config['threshold']})

    elif request.method == 'POST':
        data = request.json
        threshold = float(data.get('threshold', 0.85))
        if 0.1 <= threshold <= 0.99:
            config['threshold'] = threshold
            save_config(config)
            return jsonify({'status': 'success', 'threshold': threshold})
        return jsonify({'status': 'error', 'message': 'Invalid threshold value'}), 400


def main_video_processing():
    global main_cap
    while True:
        if main_cap and main_cap.isOpened():
            ret, frame = main_cap.read()
            if ret:
                try:
                    processed_frame = process_frame(frame, config['current_camera'])
                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    # For main feed, we process directly in the generator
                except Exception as e:
                    logging.error(f"Error processing main frame: {e}")
        else:
            time.sleep(0.1)

def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if session.get('role') != role:
                return jsonify({'status': 'error', 'message': 'Недостаточно прав'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.errorhandler(403)
def forbidden(e):
    return jsonify({'status': 'error', 'message': 'Недостаточно прав для выполнения этого действия'}), 403

@app.before_request
def check_auth():
    allowed_routes = ['login', 'register', 'static']
    if request.endpoint not in allowed_routes and 'user_id' not in session:
        return redirect(url_for('login'))

if __name__ == '__main__':
    # Инициализация базы данных
    init_db()
    initialize_models()
    try:
        # Auto-connect main camera if set
        if config['current_camera'] is not None:
            camera = next((cam for cam in config['cameras'] if cam['id'] == config['current_camera']), None)
            if camera:
                main_cap = cv2.VideoCapture(camera['url'])
                if main_cap.isOpened():
                    main_processing_thread = threading.Thread(target=main_video_processing)
                    main_processing_thread.start()

        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        stop_event.set()
        if main_cap and main_cap.isOpened():
            main_cap.release()
        if main_processing_thread:
            main_processing_thread.join()
        for camera_id, instance in list(camera_instances.items()):
            instance['stop_event'].set()
            if 'thread' in instance:
                instance['thread'].join()