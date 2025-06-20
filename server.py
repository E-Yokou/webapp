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
    send_from_directory, send_file
from auth import admin_required, login_required
from sort.sort import Sort
from ultralytics import YOLO
from util import read_license_plate, get_car, is_plate_inside_car, get_plate_center, get_car_center
import io
import xlsxwriter
import re
import requests
from PIL import Image, ImageDraw, ImageFont

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
    'threshold': 0.85,
    'fps': 15,
    'vehicle_types': ['car', 'truck', 'bus', 'motorcycle']
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
    print('Сохраняю конфиг в:', os.path.abspath(CONFIG_FILE))
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info("Конфигурация успешно сохранена")


def init_db():
    conn = sqlite3.connect('plates.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS recognized_plates
                 (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     plate_text TEXT NOT NULL,
                     camera_id INTEGER,
                     camera_name TEXT,
                     vehicle_type TEXT,
                     score REAL,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                     image BLOB,
                     linked_plate_id INTEGER
                 )''')

    # Индекс для быстрой проверки дубликатов
    c.execute('''CREATE INDEX IF NOT EXISTS idx_plate_text_time
        ON recognized_plates(plate_text, timestamp)''')

    conn.commit()
    conn.close()

    # Инициализация базы данных пользователей
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Создаем таблицу с нужными колонками
    c.execute('''CREATE TABLE IF NOT EXISTS users
    (
        id
        INTEGER
        PRIMARY
        KEY
        AUTOINCREMENT,
        username
        TEXT
        UNIQUE
        NOT
        NULL,
        password
        TEXT
        NOT
        NULL,
        role
        TEXT
        DEFAULT
        'user',
        created_at
        DATETIME
        DEFAULT (
        datetime
                 (
        'now'
                 )))''')

    # Создаем администратора по умолчанию, если его нет
    c.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if c.fetchone()[0] == 0:
        hashed_password = generate_password_hash('admin')
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                  ('admin', hashed_password, 'admin'))

    conn.commit()
    conn.close()


def insert_plate_data(plate_text, image, camera_id, camera_name, vehicle_type, score, linked_plate_id=None):
    """Вставляет данные о распознанном номере с проверкой на дубликаты за последние 5 минут, с учетом типа транспорта и связки с прицепом/грузовиком"""
    moscow_tz = pytz.timezone('Europe/Moscow')
    current_time = datetime.now(moscow_tz)

    conn = sqlite3.connect('plates.db')
    c = conn.cursor()

    # Проверяем, был ли такой номер зарегистрирован в последние 5 минут
    five_minutes_ago = current_time - timedelta(minutes=5)
    c.execute("""
              SELECT id
              FROM recognized_plates
              WHERE plate_text = ?
                AND timestamp >= ?
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
                  (plate_text, camera_id, camera_name, vehicle_type, score, timestamp, image, linked_plate_id)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)
              """, (plate_text, camera_id, camera_name, vehicle_type, score,
                    current_time.strftime('%Y-%m-%d %H:%M:%S'), image, linked_plate_id))

    new_id = c.lastrowid
    conn.commit()
    conn.close()
    logging.info(f"Новый номер {plate_text} успешно записан")
    return new_id


def get_moscow_time():
    return datetime.now(pytz.timezone('Europe/Moscow'))


def get_recent_plates(limit=100):
    """Возвращает список последних распознанных номеров с московским временем, статусом въезд/выезд, типом транспорта и связью грузовик-прицеп"""
    conn = sqlite3.connect('plates.db')
    c = conn.cursor()
    c.execute("""
              SELECT id,
                     plate_text,
                     camera_id,
                     camera_name,
                     vehicle_type,
                     score,
                     datetime(timestamp, 'localtime') as local_timestamp,
                     timestamp,
                     linked_plate_id
              FROM recognized_plates
              ORDER BY timestamp DESC
                  LIMIT ?""", (limit,))

    plates = []
    for row in c.fetchall():
        plate = {
            'id': row[0],
            'text': row[1],
            'camera_id': row[2],
            'camera_name': row[3],
            'vehicle_type': row[4],
            'score': row[5],
            'timestamp': row[6],  # локальное время для отображения
            'raw_timestamp': row[7],  # для анализа истории
            'linked_plate_id': row[8]
        }
        # Если есть связь, получаем номер связанного участника
        if plate['linked_plate_id']:
            c2 = conn.cursor()
            c2.execute("SELECT plate_text FROM recognized_plates WHERE id = ?", (plate['linked_plate_id'],))
            linked_row = c2.fetchone()
            plate['linked_plate_text'] = linked_row[0] if linked_row else None
        plates.append(plate)
    # Для определения статуса въезд/выезд нужно получить историю для каждого номера
    plate_texts = list(set([p['text'] for p in plates]))
    history = {}
    for plate_text in plate_texts:
        c.execute("""
            SELECT id, timestamp
            FROM recognized_plates
            WHERE plate_text = ?
            ORDER BY timestamp ASC
        """, (plate_text,))
        history[plate_text] = c.fetchall()
    conn.close()
    for plate in plates:
        plate_id = plate['id']
        plate_text = plate['text']
        events = history[plate_text]
        idx = next((i for i, (pid, _) in enumerate(events) if pid == plate_id), None)
        if idx is not None:
            if idx % 2 == 0:
                plate['event_type'] = 'in'
                plate['event_time'] = events[idx][1]
            else:
                plate['event_type'] = 'out'
                plate['event_time'] = events[idx][1]
        else:
            plate['event_type'] = 'unknown'
            plate['event_time'] = plate['raw_timestamp']
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
    # Инициализируем модель с ограничением классов
    coco_model = YOLO('models/yolo11n.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
    # Устанавливаем только нужные классы (2:car, 3:motorcycle, 5:bus, 7:truck)
    coco_model.classes = [2, 3, 5, 7]
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
@admin_required
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        role = request.form.get('role', 'user')

        if password != confirm_password:
            flash('Пароли не совпадают')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                      (username, hashed_password, role))
            conn.commit()
            conn.close()
            flash('Пользователь успешно создан.')
            return redirect(url_for('user_management'))
        except sqlite3.IntegrityError:
            flash('Имя пользователя уже существует')
            return redirect(url_for('register'))

    return render_template('register.html')


def is_trailer_plate(plate_text, region='ru'):
    # РФ: 4 цифры + 2 буквы + 2 цифры (пример: 1234АА77)
    #     2 буквы + 6 цифр (пример: АР327328)
    if region == 'ru':
        return (
            bool(re.match(r'^[0-9]{4}[A-ZА-Я]{2}[0-9]{2}$', plate_text)) or
            bool(re.match(r'^[A-ZА-Я]{2}[0-9]{6}$', plate_text))
        )
    # Беларусь: 4 цифры + 2 буквы + 1 цифра (пример: 1234АА7)
    if region == 'by':
        return bool(re.match(r'^[0-9]{4}[A-ZА-Я]{2}[0-9]$', plate_text))
    # Казахстан: 4 цифры + 2 буквы + 2 цифры
    if region == 'kz':
        return bool(re.match(r'^[0-9]{4}[A-ZА-Я]{2}[0-9]{2}$', plate_text))
    # Европа: можно добавить шаблоны
    return False


def process_frame(frame, camera_id=None):
    try:
        start_time = time.time()
        
        # Получаем настройки камеры
        camera_settings = None
        for camera in config.get('cameras', []):
            if camera.get('id') == camera_id:
                camera_settings = camera.get('settings', {})
                break
        
        if not camera_settings:
            camera_settings = {
                'roi': {'enabled': False, 'x': 0, 'y': 0, 'width': 100, 'height': 100},
                'brightness': 0,
                'contrast': 0,
                'frame_skip': {'enabled': False, 'interval': 1}
            }

        # Apply ROI
        if camera_settings['roi']['enabled']:
            height, width = frame.shape[:2]
            x = int(camera_settings['roi']['x'] * width / 100)
            y = int(camera_settings['roi']['y'] * height / 100)
            w = int(camera_settings['roi']['width'] * width / 100)
            h = int(camera_settings['roi']['height'] * height / 100)
            # Ensure ROI is within frame bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            frame = frame[y:y + h, x:x + w]

        # Apply brightness and contrast
        if camera_settings['brightness'] != 0 or camera_settings['contrast'] != 0:
            brightness = camera_settings['brightness']
            contrast = camera_settings['contrast'] / 100.0 + 1.0  # Convert to scale factor
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

        # Vehicle detection - теперь модель уже настроена только на транспортные средства
        vehicle_detections = coco_model(frame)[0]
        vehicle_boxes = []
        vehicle_types = []
        allowed_types = config.get('vehicle_types', ['car', 'truck', 'bus', 'motorcycle'])
        
        for detection in vehicle_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) == 2:
                vtype = 'Легковое авто'
                vtype_key = 'car'
            elif int(class_id) == 7:
                vtype = 'Грузовик'
                vtype_key = 'truck'
            elif int(class_id) == 5:
                vtype = 'Автобус'
                vtype_key = 'bus'
            elif int(class_id) == 3:
                vtype = 'Мотоцикл'
                vtype_key = 'motorcycle'
            else:
                continue  # Пропускаем все остальные классы
                
            if vtype_key in allowed_types:
                vehicle_boxes.append([x1, y1, x2, y2, score])
                vehicle_types.append(vtype)

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
            best_vehicle_type = 'Неизвестно'
            for idx, track in enumerate(track_ids):
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
                        if idx < len(vehicle_types):
                            best_vehicle_type = vehicle_types[idx]
                        else:
                            best_vehicle_type = 'Неизвестно'

            if best_match:
                tracked_plates[best_match] = {
                    'plate_text': plate_text,
                    'plate_score': plate_score,
                    'last_seen': current_time
                }
                updated_vehicles.add(best_match)

                # --- Корректировка типа транспорта ---
                region = config.get('region', 'ru')
                if best_vehicle_type == 'Грузовик' and is_trailer_plate(plate_text, region):
                    vehicle_type_for_db = 'Прицеп'
                else:
                    vehicle_type_for_db = best_vehicle_type
                # --- Логика связки грузовик-прицеп ---
                is_trailer = (vehicle_type_for_db == 'Прицеп')
                is_truck = (vehicle_type_for_db == 'Грузовик')
                linked_plate_id = None
                # Если это прицеп, ищем грузовик за последние 60 секунд на той же камере
                if is_trailer:
                    conn = sqlite3.connect('plates.db')
                    c = conn.cursor()
                    c.execute("""
                        SELECT id FROM recognized_plates
                        WHERE vehicle_type = 'Грузовик'
                          AND camera_id = ?
                          AND ABS(strftime('%s', timestamp) - strftime('%s', ?)) <= 60
                        ORDER BY ABS(strftime('%s', timestamp) - strftime('%s', ?)) ASC
                        LIMIT 1
                    """, (camera_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    row = c.fetchone()
                    if row:
                        linked_plate_id = row[0]
                    conn.close()
                # Если это грузовик, ищем прицеп за последние 60 секунд на той же камере
                elif is_truck:
                    conn = sqlite3.connect('plates.db')
                    c = conn.cursor()
                    c.execute("""
                        SELECT id FROM recognized_plates
                        WHERE vehicle_type = 'Прицеп'
                          AND camera_id = ?
                          AND ABS(strftime('%s', timestamp) - strftime('%s', ?)) <= 60
                        ORDER BY ABS(strftime('%s', timestamp) - strftime('%s', ?)) ASC
                        LIMIT 1
                    """, (camera_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    row = c.fetchone()
                    if row:
                        linked_plate_id = row[0]
                    conn.close()
                # --- Конец логики связки ---

                # Сохраняем только если номер новый (без дубликатов за 5 минут)
                if plate_text != getattr(app, f'last_saved_plate_{camera_id}', None):
                    try:
                        _, buffer = cv2.imencode('.jpg', frame)
                        camera_name = next((cam['name'] for cam in config['cameras'] if cam['id'] == camera_id),
                                           'Unknown')
                        # Пытаемся вставить (функция сама проверит дубликаты)
                        new_id = insert_plate_data(plate_text, buffer.tobytes(), camera_id, camera_name, vehicle_type_for_db, plate_score, linked_plate_id)
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
                
                # Создаем PIL изображение для текста
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)

                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 24, encoding='utf-8')
                except:
                    font = ImageFont.load_default()

                # Рассчитываем размер текста
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Позиция над bounding box
                text_x = int(x1)
                text_y = int(y1) - text_height - 10

                # Рисуем прямоугольник фона
                draw.rectangle(
                    [(text_x - 5, text_y - 5),
                     (text_x + text_width + 5, text_y + text_height + 5)],
                    fill=(0, 0, 255)  # Синий фон
                )

                # Рисуем текст
                draw.text(
                    (text_x, text_y),
                    text,
                    font=font,
                    fill=(255, 255, 255)  # Белый текст
                )

                # Конвертируем обратно в OpenCV формат
                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Calculate and display FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame
    except Exception as e:
        logging.error(f"Error in process_frame: {e}")
        return frame


def is_valid_camera_url(url):
    if not url:
        return False
    # Разрешаем обычные потоки и .html-страницы
    if url.startswith(('rtsp://', 'http://', 'https://')):
        return True
    return False


@app.route('/api/camera/<int:camera_id>/reconnect', methods=['POST'])
def reconnect_camera(camera_id):
    if camera_id not in camera_instances:
        return jsonify({'status': 'error', 'message': 'Camera not found'}), 404

    instance = camera_instances[camera_id]
    instance['stop_event'].set()

    # Даем потоку время завершиться
    time.sleep(0.5)

    # Запускаем новый поток
    stop_event = threading.Event()
    thread = threading.Thread(
        target=camera_processing,
        args=(camera_id, instance['url'], stop_event),
        daemon=True
    )
    thread.start()

    return jsonify({'status': 'success'})


def camera_processing(camera_id, camera_url, stop_event):
    global camera_states, camera_instances

    # Инициализация состояния камеры
    camera_states[camera_id] = {
        'connected': False,
        'last_activity': datetime.now().isoformat(),
        'error_count': 0,
        'reconnect_attempts': 0
    }

    cap = None
    video_writer = None
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5
    MAX_RECONNECT_ATTEMPTS = 10
    RECONNECT_DELAY = 5  # секунд
    last_image_time = 0
    image_update_interval = 0.3  # 300ms для HTML-камер
    last_frame_time = 0  # Время последнего полученного кадра
    frame_interval = 0.1  # Уменьшаем интервал до 100ms

    def cleanup():
        nonlocal cap, video_writer
        try:
            if video_writer is not None:
                video_writer.release()
                video_writer = None

            if cap is not None:
                try:
                    if cap.isOpened():
                        cap.release()
                except:
                    pass
                cap = None

            if camera_id in camera_instances:
                del camera_instances[camera_id]

            camera_states[camera_id].update({
                'connected': False,
                'processing': False,
                'shutdown_time': datetime.now().isoformat()
            })
        except Exception as cleanup_error:
            logging.error(f"Ошибка при очистке ресурсов камеры {camera_id}: {str(cleanup_error)}")

    def connect_camera():
        nonlocal cap
        try:
            # Определяем тип камеры
            is_html_camera = camera_url.endswith('.html')
            
            if is_html_camera:
                # Для HTML-камер создаем специальный обработчик
                camera_instances[camera_id] = {
                    'stop_event': stop_event,
                    'thread': threading.current_thread(),
                    'recording': False,
                    'video_writer': None,
                    'recording_file': None,
                    'last_frame_time': time.time(),
                    'is_html_camera': True,
                    'url': camera_url,
                    'last_frame': None
                }
                
                # Успешное подключение для HTML-камеры
                camera_states[camera_id].update({
                    'connected': True,
                    'last_activity': datetime.now().isoformat(),
                    'error_count': 0,
                    'reconnect_attempts': 0
                })
                
                logging.info(f"HTML-камера {camera_id} успешно инициализирована")
                return True
            else:
                # Для IP-камер
                clean_url = camera_url.split('?')[0] if '?' in camera_url else camera_url
                logging.info(f"Подключение к IP-камере {camera_id} по адресу: {clean_url}")

                cap = cv2.VideoCapture(clean_url)
                
                # Устанавливаем параметры для стабильной работы
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Уменьшаем размер буфера
                    cap.set(cv2.CAP_PROP_FPS, 15)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                except AttributeError as e:
                    logging.warning(f"Некоторые свойства камеры недоступны: {str(e)}")

                # Проверяем подключение
                if not cap.isOpened():
                    raise Exception("Не удалось открыть видеопоток")

                # Проверяем, что камера действительно подключена
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise Exception("Не удалось получить кадр от камеры")

                camera_instances[camera_id] = {
                    'cap': cap,
                    'stop_event': stop_event,
                    'thread': threading.current_thread(),
                    'recording': False,
                    'video_writer': None,
                    'recording_file': None,
                    'last_frame_time': time.time(),
                    'last_frame': None
                }

                camera_states[camera_id].update({
                    'connected': True,
                    'last_activity': datetime.now().isoformat(),
                    'error_count': 0,
                    'reconnect_attempts': 0
                })

                logging.info(f"IP-камера {camera_id} успешно подключена")
                return True

        except Exception as e:
            logging.error(f"Ошибка подключения к камере {camera_id}: {str(e)}")
            cleanup()
            return False

    try:
        # Первоначальное подключение
        if not connect_camera():
            return

        while not stop_event.is_set():
            try:
                if camera_instances[camera_id].get('is_html_camera'):
                    # Обработка HTML-камеры
                    try:
                        current_time = time.time()
                        if current_time - last_image_time < image_update_interval:
                            time.sleep(0.1)
                            continue

                        response = requests.get(camera_url, timeout=5)
                        if response.status_code != 200:
                            raise requests.RequestException(f"HTTP {response.status_code}")

                        img_response = requests.get(camera_url, stream=True, timeout=5)
                        if img_response.status_code != 200:
                            raise requests.RequestException(f"HTTP {img_response.status_code}")

                        image_array = np.asarray(bytearray(img_response.content), dtype=np.uint8)
                        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                        
                        if frame is None:
                            raise ValueError("Не удалось декодировать изображение")

                        processed_frame = process_frame(frame, camera_id)
                        
                        camera_states[camera_id].update({
                            'connected': True,
                            'processing': True,
                            'last_activity': datetime.now().isoformat(),
                            'error_count': 0,
                            'fps': 1 / (current_time - camera_instances[camera_id]['last_frame_time'])
                        })
                        camera_instances[camera_id]['last_frame_time'] = current_time
                        camera_instances[camera_id]['last_frame'] = processed_frame
                        last_image_time = current_time

                        if camera_instances[camera_id].get('recording'):
                            try:
                                if video_writer is None:
                                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                    fps = int(1 / image_update_interval)
                                    height, width = processed_frame.shape[:2]
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename = f"recording_{camera_id}_{timestamp}.avi"
                                    filepath = os.path.join(RECORDINGS_DIR, filename)
                                    video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
                                    camera_instances[camera_id]['video_writer'] = video_writer
                                    camera_instances[camera_id]['recording_file'] = filepath
                                
                                video_writer.write(processed_frame)
                            except Exception as e:
                                logging.error(f"Ошибка записи видео: {str(e)}")
                                camera_instances[camera_id]['recording'] = False

                    except Exception as frame_error:
                        logging.error(f"Ошибка обработки кадра HTML-камеры: {str(frame_error)}")
                        camera_states[camera_id]['error_count'] += 1
                        camera_states[camera_id]['last_error'] = str(frame_error)
                        
                        if camera_states[camera_id]['error_count'] >= MAX_CONSECUTIVE_ERRORS:
                            logging.warning(f"Превышено максимальное количество ошибок для камеры {camera_id}, переподключение...")
                            cleanup()
                            if camera_states[camera_id]['reconnect_attempts'] < MAX_RECONNECT_ATTEMPTS:
                                camera_states[camera_id]['reconnect_attempts'] += 1
                                time.sleep(RECONNECT_DELAY)
                                if connect_camera():
                                    continue
                            else:
                                logging.error(f"Превышено максимальное количество попыток переподключения для камеры {camera_id}")
                                break
                        time.sleep(1)
                        continue

                else:
                    # Обработка IP-камеры
                    if not cap or not cap.isOpened():
                        raise Exception("Камера отключена")

                    current_time = time.time()
                    # Проверяем, прошло ли достаточно времени с последнего кадра
                    if current_time - last_frame_time >= frame_interval:
                        # Очищаем буфер камеры перед чтением нового кадра
                        for _ in range(5):  # Пропускаем несколько кадров из буфера
                            cap.grab()

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        raise Exception("Получен пустой кадр")

                    processed_frame = process_frame(frame, camera_id)

                    camera_states[camera_id].update({
                        'connected': True,
                        'processing': True,
                        'last_activity': datetime.now().isoformat(),
                        'error_count': 0,
                        'fps': 1 / (current_time - camera_instances[camera_id]['last_frame_time'])
                    })
                    camera_instances[camera_id]['last_frame_time'] = current_time
                    camera_instances[camera_id]['last_frame'] = processed_frame
                    last_frame_time = current_time

                    if camera_instances[camera_id].get('recording'):
                        try:
                            if video_writer is None:
                                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                try:
                                    fps = cap.get(cv2.CAP_PROP_FPS)
                                except:
                                    fps = 15

                                try:
                                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                except:
                                    width, height = 960, 720

                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"recording_{camera_id}_{timestamp}.avi"
                                filepath = os.path.join(RECORDINGS_DIR, filename)
                                video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
                                camera_instances[camera_id]['video_writer'] = video_writer
                                camera_instances[camera_id]['recording_file'] = filepath

                            video_writer.write(processed_frame)
                        except Exception as e:
                            logging.error(f"Ошибка записи видео: {str(e)}")
                            camera_instances[camera_id]['recording'] = False
                    else:
                        # Если секунда еще не прошла, делаем небольшую паузу
                        time.sleep(0.1)

            except Exception as frame_error:
                logging.error(f"Ошибка обработки кадра камеры {camera_id}: {str(frame_error)}")
                camera_states[camera_id]['error_count'] += 1
                camera_states[camera_id]['last_error'] = str(frame_error)
                
                if camera_states[camera_id]['error_count'] >= MAX_CONSECUTIVE_ERRORS:
                    logging.warning(f"Превышено максимальное количество ошибок для камеры {camera_id}, переподключение...")
                    cleanup()
                    if camera_states[camera_id]['reconnect_attempts'] < MAX_RECONNECT_ATTEMPTS:
                        camera_states[camera_id]['reconnect_attempts'] += 1
                        time.sleep(RECONNECT_DELAY)
                        if connect_camera():
                            continue
                    else:
                        logging.error(f"Превышено максимальное количество попыток переподключения для камеры {camera_id}")
                        break
                time.sleep(1)

    except Exception as main_error:
        logging.critical(f"Критическая ошибка обработки камеры {camera_id}: {str(main_error)}")
        camera_states[camera_id].update({
            'connected': False,
            'last_error': str(main_error)
        })

    finally:
        cleanup()

def gen_camera_frames(camera_id):
    while True:
        if camera_id in camera_instances:
            try:
                last_frame = camera_instances[camera_id].get('last_frame')
                if last_frame is not None:
                    _, buffer = cv2.imencode('.jpg', last_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.01)  # Уменьшаем задержку до 10ms
            except Exception as e:
                logging.error(f"Ошибка генерации кадра для камеры {camera_id}: {str(e)}")
                time.sleep(0.1)
        else:
            time.sleep(0.1)

def generate_combined_feed():
    while True:
        try:
            frames = []
            for camera_id, instance in camera_instances.items():
                last_frame = instance.get('last_frame')
                if last_frame is not None:
                    frames.append((camera_id, last_frame))

            if frames:
                # Выбираем первый доступный кадр
                camera_id, frame = frames[0]
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            logging.error(f"Ошибка генерации комбинированного потока: {str(e)}")
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
        # Удаляем raw_timestamp из ответа (он только для внутренней логики)
        if 'raw_timestamp' in plate:
            del plate['raw_timestamp']

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

        if role not in ['user', 'admin']:
            return jsonify({'status': 'error', 'message': 'Invalid role'}), 400

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

        # Запрещаем пользователю изменять свою собственную роль
        if user_id == session.get('user_id'):
            return jsonify({'status': 'error', 'message': 'Cannot change your own role'}), 400

        if new_role:
            if new_role not in ['user', 'admin']:
                return jsonify({'status': 'error', 'message': 'Invalid role'}), 400
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
    return None


@app.route('/api/settings', methods=['GET', 'POST'])
@admin_required
def manage_settings():
    if request.method == 'GET':
        # Return current settings
        return jsonify({
            'threshold': config.get('threshold', 0.85),
            'region': config.get('region', 'ru'),
            'enable_alpr': config.get('enable_alpr', True),
            'save_images': config.get('save_images', True),
            'email_notifications': config.get('email_notifications', False),
            'max_age': config.get('max_age', 30),
            'delete_before_date': config.get('delete_before_date', ''),
            'fps': config.get('fps', 15),
            'vehicle_types': config.get('vehicle_types', ['car', 'truck', 'bus', 'motorcycle']),
            'video_processing': config.get('video_processing', {
                'roi': {'enabled': False, 'x': 0, 'y': 0, 'width': 100, 'height': 100},
                'brightness': {'enabled': False, 'value': 0},
                'contrast': {'enabled': False, 'value': 1.0},
                'frame_skip': {'enabled': False, 'value': 1}
            })
        })

    elif request.method == 'POST':
        # Update settings
        data = request.json
        config.update({
            'threshold': float(data.get('threshold', config.get('threshold', 0.85))),
            'region': data.get('region', config.get('region', 'ru')),
            'enable_alpr': bool(data.get('enable_alpr', config.get('enable_alpr', True))),
            'save_images': bool(data.get('save_images', config.get('save_images', True))),
            'email_notifications': bool(data.get('email_notifications', config.get('email_notifications', False))),
            'max_age': int(data.get('max_age', config.get('max_age', 30))),
            'delete_before_date': data.get('delete_before_date', config.get('delete_before_date', '')),
            'fps': int(data.get('fps', config.get('fps', 15))),
            'vehicle_types': data.get('vehicle_types', config.get('vehicle_types', ['car', 'truck', 'bus', 'motorcycle'])),
            'video_processing': data.get('video_processing', config.get('video_processing', {
                'roi': {'enabled': False, 'x': 0, 'y': 0, 'width': 100, 'height': 100},
                'brightness': {'enabled': False, 'value': 0},
                'contrast': {'enabled': False, 'value': 1.0},
                'frame_skip': {'enabled': False, 'value': 1}
            }))
        })
        save_config(config)
        return jsonify({'status': 'success'})
    return None


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


@app.route('/api/cameras', methods=['GET'])
@login_required
def get_cameras():
    """Возвращает список камер (доступно всем авторизованным пользователям)"""
    return jsonify({'cameras': config['cameras']})


@app.route('/api/cameras', methods=['POST', 'DELETE'])
@admin_required
def manage_cameras():
    """Изменение камер доступно только администраторам"""
    if request.method == 'POST':
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


@app.route('/api/cameras/status')
def get_cameras_status():
    statuses = {}
    for camera_id in config['cameras']:
        camera_id = camera_id['id']
        status = camera_states.get(camera_id, {'connected': False})
        
        # Проверяем реальное состояние камеры
        if camera_id in camera_instances:
            instance = camera_instances[camera_id]
            if 'cap' in instance:
                # Проверяем, открыта ли камера и получаем кадр
                if instance['cap'].isOpened():
                    ret, _ = instance['cap'].read()
                    if ret:
                        status['connected'] = True
                        status['last_activity'] = datetime.now().isoformat()
                    else:
                        status['connected'] = False
                        status['last_error'] = 'Не удалось получить кадр'
                else:
                    status['connected'] = False
                    status['last_error'] = 'Камера не открыта'
            else:
                status['connected'] = False
                status['last_error'] = 'Экземпляр камеры не инициализирован'
        else:
            status['connected'] = False
            status['last_error'] = 'Камера не найдена в списке активных'
            
        # Определяем статус processing
        is_processing = camera_id in camera_instances and not camera_instances[camera_id]['stop_event'].is_set() if camera_id in camera_instances else False
        
        # Если камера обрабатывает кадры, она должна считаться подключенной
        if is_processing:
            status['connected'] = True
            
        statuses[camera_id] = {
            'connected': status['connected'],
            'processing': is_processing
        }
    return jsonify(statuses)


@app.route('/api/camera_status/<int:camera_id>')
@login_required
def get_camera_status(camera_id):
    """Возвращает текущий статус камеры"""
    status = camera_states.get(camera_id, {'connected': False})

    # Проверяем реальное состояние камеры
    if camera_id in camera_instances:
        instance = camera_instances[camera_id]
        if 'cap' in instance:
            # Проверяем, открыта ли камера и получаем кадр
            if instance['cap'].isOpened():
                ret, _ = instance['cap'].read()
                if ret:
                    status['connected'] = True
                    status['last_activity'] = datetime.now().isoformat()
                else:
                    status['connected'] = False
                    status['last_error'] = 'Не удалось получить кадр'
            else:
                status['connected'] = False
                status['last_error'] = 'Камера не открыта'
        else:
            status['connected'] = False
            status['last_error'] = 'Экземпляр камеры не инициализирован'
    else:
        status['connected'] = False
        status['last_error'] = 'Камера не найдена в списке активных'

    # Добавляем дополнительную информацию для администраторов
    if session.get('role') == 'admin':
        camera = next((cam for cam in config['cameras'] if cam['id'] == camera_id), None)
        if camera:
            status['camera_info'] = {
                'name': camera['name'],
                'url': camera['url']
            }

    return jsonify(status)


def is_camera_alive(camera_id):
    if camera_id not in camera_instances:
        return False
    cap = camera_instances[camera_id].get('cap')
    return cap is not None and cap.isOpened()


@app.route('/api/connect', methods=['POST'])
@admin_required
def connect_camera():
    data = request.json
    if 'id' not in data:
        return jsonify({'status': 'error', 'message': 'Missing camera id'}), 400

    camera = next((cam for cam in config['cameras'] if cam['id'] == data['id']), None)
    if not camera:
        return jsonify({'status': 'error', 'message': 'Camera not found'}), 404

    # Проверяем, не запущена ли уже камера
    if is_camera_alive(data['id']):
        return jsonify({'status': 'success', 'camera': camera})

    if not is_valid_camera_url(camera['url']):
        return jsonify({'status': 'error', 'message': 'Invalid camera URL'}), 400

    # Если это .html-камера, не запускаем обработку потока
    if camera['url'].endswith('.html'):
        return jsonify({'status': 'success', 'camera': camera})

    # Если камера в словаре, но не работает - очищаем
    if data['id'] in camera_instances:
        old_instance = camera_instances[data['id']]
        if 'cap' in old_instance:
            old_instance['cap'].release()
        if 'stop_event' in old_instance:
            old_instance['stop_event'].set()
        del camera_instances[data['id']]

    # Запускаем новый поток
    stop_event = threading.Event()
    thread = threading.Thread(
        target=camera_processing,
        args=(data['id'], camera['url'], stop_event),
        daemon=True  # Поток завершится при выходе основного
    )
    thread.start()

    time.sleep(2)

    # Проверяем, что камера реально запустилась
    if data['id'] in camera_instances and camera_instances[data['id']]['cap'].isOpened():
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
        # else:
        #     time.sleep(0.1)


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


@app.route('/api/plates/filter')
@login_required
def get_filtered_plates():
    """Возвращает отфильтрованные номера по заданным критериям"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    plate_number = request.args.get('plate_number')
    camera_id = request.args.get('camera_id')
    confidence = request.args.get('confidence', type=float)

    conn = sqlite3.connect('plates.db')
    c = conn.cursor()

    query = """
            SELECT id, \
                   plate_text, \
                   camera_id, \
                   camera_name, \
                   score,
                   datetime(timestamp, 'localtime') as local_timestamp
            FROM recognized_plates
            WHERE 1 = 1 \
            """
    params = []

    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    if plate_number:
        query += " AND plate_text LIKE ?"
        params.append(f'%{plate_number}%')
    if camera_id:
        query += " AND camera_id = ?"
        params.append(camera_id)
    if confidence:
        query += " AND score >= ?"
        params.append(confidence)

    query += " ORDER BY timestamp DESC LIMIT 1000"

    c.execute(query, params)
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
    return jsonify({'plates': plates})


@app.route('/api/plates/export')
@login_required
def export_plates():
    """Экспортирует отфильтрованные номера в Excel"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    plate_number = request.args.get('plate_number')
    camera_id = request.args.get('camera_id')
    confidence = request.args.get('confidence', type=float)

    conn = sqlite3.connect('plates.db')
    c = conn.cursor()

    query = """
            SELECT plate_text, \
                   camera_name, \
                   score,
                   datetime(timestamp, 'localtime') as local_timestamp
            FROM recognized_plates
            WHERE 1 = 1 \
            """
    params = []

    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    if plate_number:
        query += " AND plate_text LIKE ?"
        params.append(f'%{plate_number}%')
    if camera_id:
        query += " AND camera_id = ?"
        params.append(camera_id)
    if confidence:
        query += " AND score >= ?"
        params.append(confidence)

    query += " ORDER BY timestamp DESC"

    c.execute(query, params)
    plates = c.fetchall()
    conn.close()

    # Создаем Excel файл
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet()

    # Форматирование
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#D9D9D9',
        'border': 1
    })
    cell_format = workbook.add_format({
        'border': 1
    })

    # Заголовки
    headers = ['Номер', 'Камера', 'Уверенность', 'Дата и время']
    for col, header in enumerate(headers):
        worksheet.write(0, col, header, header_format)

    # Данные
    for row, plate in enumerate(plates, start=1):
        worksheet.write(row, 0, plate[0], cell_format)  # Номер
        worksheet.write(row, 1, plate[1], cell_format)  # Камера
        worksheet.write(row, 2, f"{plate[2] * 100:.1f}%", cell_format)  # Уверенность
        worksheet.write(row, 3, plate[3], cell_format)  # Дата и время

    # Автоматическая ширина столбцов
    for col in range(len(headers)):
        worksheet.set_column(col, col, 15)

    workbook.close()
    output.seek(0)

    # Генерируем имя файла
    filename = f"plates_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )


@app.route('/api/plates/delete', methods=['POST'])
@admin_required
def delete_plates():
    """Удаляет записи до указанной даты"""
    data = request.json
    before_date = data.get('before_date')

    if not before_date:
        return jsonify({'status': 'error', 'message': 'Не указана дата'}), 400

    conn = sqlite3.connect('plates.db')
    c = conn.cursor()

    # Сначала получаем количество записей для удаления
    c.execute("SELECT COUNT(*) FROM recognized_plates WHERE timestamp < ?", (before_date,))
    count = c.fetchone()[0]

    # Удаляем записи
    c.execute("DELETE FROM recognized_plates WHERE timestamp < ?", (before_date,))
    conn.commit()
    conn.close()

    return jsonify({
        'status': 'success',
        'message': f'Удалено {count} записей',
        'deleted_count': count
    })


@app.route('/api/settings/reset', methods=['POST'])
@admin_required
def reset_settings():
    global config
    config = DEFAULT_CONFIG.copy()
    save_config(config)
    return jsonify({'status': 'success'})


@app.route('/api/data/cleanup', methods=['POST'])
@admin_required
def cleanup_data():
    try:
        before_date = request.json.get('before_date')
        max_age = request.json.get('max_age')

        conn = sqlite3.connect('plates.db')
        c = conn.cursor()

        conditions = []
        params = []

        if before_date:
            conditions.append("timestamp < ?")
            params.append(before_date)

        if max_age:
            try:
                max_age_int = int(max_age)
                cutoff_date = (datetime.now() - timedelta(days=max_age_int)).strftime('%Y-%m-%d %H:%M:%S')
                conditions.append("timestamp < ?")
                params.append(cutoff_date)
            except ValueError:
                return jsonify({'status': 'error', 'message': 'Invalid max_age value'}), 400

        if not conditions:
            return jsonify({'status': 'error', 'message': 'No cleanup criteria provided'}), 400

        # First count records to be deleted
        count_query = f"SELECT COUNT(*) FROM recognized_plates WHERE {' OR '.join(conditions)}"
        c.execute(count_query, params)
        count = c.fetchone()[0]

        # Then delete them
        delete_query = f"DELETE FROM recognized_plates WHERE {' OR '.join(conditions)}"
        c.execute(delete_query, params)

        # Also delete images if they exist in the filesystem
        if config.get('save_images', True):
            # Get IDs of deleted records to delete their images
            c.execute(f"SELECT id FROM recognized_plates WHERE {' OR '.join(conditions)}", params)
            deleted_ids = [row[0] for row in c.fetchall()]

            for plate_id in deleted_ids:
                image_path = os.path.join(SNAPSHOTS_DIR, f'plate_{plate_id}.jpg')
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception as e:
                        logging.error(f"Error deleting image {image_path}: {e}")

        conn.commit()
        conn.close()

        return jsonify({
            'status': 'success',
            'deleted_count': count
        })

    except Exception as e:
        logging.error(f"Error in cleanup_data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Internal server error: {str(e)}"
        }), 500


@app.route('/api/stats')
@login_required
def get_system_stats():
    conn = sqlite3.connect('plates.db')
    c = conn.cursor()

    # Total records
    c.execute("SELECT COUNT(*) FROM recognized_plates")
    total_records = c.fetchone()[0]

    # Total images (only if saving images is enabled)
    if config.get('save_images', True):
        c.execute("SELECT COUNT(*) FROM recognized_plates WHERE image IS NOT NULL")
        total_images = c.fetchone()[0]
    else:
        total_images = 0

    # Database size
    db_size = os.path.getsize('plates.db') / (1024 * 1024)  # in MB

    conn.close()

    return jsonify({
        'total_records': total_records,
        'total_images': total_images,
        'db_size': round(db_size, 2)
    })


@app.route('/api/stats/hourly')
@login_required
def get_hourly_stats():
    """Возвращает статистику по часам за последние 24 часа"""
    conn = sqlite3.connect('plates.db')
    c = conn.cursor()

    # Получаем данные по часам
    c.execute("""
              SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
              FROM recognized_plates
              WHERE timestamp >= datetime('now', '-24 hours')
              GROUP BY hour
              ORDER BY hour
              """)

    # Создаем массив для всех часов
    hourly_data = [0] * 24
    for row in c.fetchall():
        hour = int(row[0])
        count = row[1]
        hourly_data[hour] = count

    conn.close()
    return jsonify({'hourly_data': hourly_data})


@app.route('/api/stats/cameras')
@login_required
def get_camera_stats():
    """Возвращает статистику по камерам"""
    conn = sqlite3.connect('plates.db')
    c = conn.cursor()

    # Получаем данные по камерам
    c.execute("""
              SELECT camera_name, COUNT(*) as count, AVG(score) as avg_score
              FROM recognized_plates
              WHERE timestamp >= datetime('now', '-24 hours')
              GROUP BY camera_name
              ORDER BY count DESC
              """)

    cameras = []
    for row in c.fetchall():
        cameras.append({
            'name': row[0],
            'count': row[1],
            'avg_score': round(row[2] * 100, 1)
        })

    conn.close()
    return jsonify({'cameras': cameras})


@app.route('/api/cameras', methods=['PUT'])
@admin_required
def update_camera():
    data = request.json
    if 'id' not in data:
        return jsonify({'status': 'error', 'message': 'Missing camera id'}), 400

    # Находим индекс камеры в списке
    camera_index = None
    for i, cam in enumerate(config['cameras']):
        if cam['id'] == data['id']:
            camera_index = i
            break

    if camera_index is None:
        return jsonify({'status': 'error', 'message': 'Camera not found'}), 404

    # Проверяем URL на дубликаты (исключая текущую камеру)
    new_url = data.get('url')
    if new_url:
        for cam in config['cameras']:
            if cam['url'] == new_url and cam['id'] != data['id']:
                return jsonify({
                    'status': 'error',
                    'message': 'Камера с таким URL уже существует'
                }), 400

    # Обновляем только существующую камеру (не создаём новую)
    if 'name' in data:
        config['cameras'][camera_index]['name'] = data['name']
    if 'url' in data:
        config['cameras'][camera_index]['url'] = data['url']
    if 'location' in data:
        config['cameras'][camera_index]['location'] = data.get('location', '')

    save_config(config)

    # Переподключаем камеру если она активна
    if data['id'] in camera_instances:
        instance = camera_instances[data['id']]
        instance['stop_event'].set()
        time.sleep(0.5)

        stop_event = threading.Event()
        thread = threading.Thread(
            target=camera_processing,
            args=(data['id'], config['cameras'][camera_index]['url'], stop_event),
            daemon=True
        )
        thread.start()

    return jsonify({
        'status': 'success',
        'camera': config['cameras'][camera_index],
        'message': 'Camera updated successfully'
    })

@app.route('/api/stats/top_plates')
@login_required
def get_top_plates():
    """Возвращает топ часто встречающихся номеров"""
    conn = sqlite3.connect('plates.db')
    c = conn.cursor()

    # Получаем топ номеров
    c.execute("""
              SELECT plate_text, COUNT(*) as count
              FROM recognized_plates
              WHERE timestamp >= datetime('now', '-24 hours')
              GROUP BY plate_text
              HAVING count > 1
              ORDER BY count DESC
                  LIMIT 10
              """)

    top_plates = []
    for row in c.fetchall():
        top_plates.append({
            'plate': row[0],
            'count': row[1]
        })

    conn.close()
    return jsonify({'top_plates': top_plates})


@app.route('/api/process_html_image/<int:camera_id>')
@login_required
def process_html_image(camera_id):
    """Обрабатывает изображение от HTML-камеры"""
    try:
        # Получаем URL изображения из параметров
        image_url = request.args.get('url')
        if not image_url:
            return jsonify({'status': 'error', 'message': 'URL изображения не указан'}), 400

        # Проверяем, что это действительно HTML-камера
        camera = next((cam for cam in config['cameras'] if cam['id'] == camera_id), None)
        if not camera or not camera['url'].endswith('.html'):
            return jsonify({'status': 'error', 'message': 'Неверный тип камеры'}), 400

        # Обновляем статус камеры
        camera_states[camera_id] = {
            'connected': True,
            'processing': True,
            'last_activity': datetime.now().isoformat(),
            'error_count': 0
        }

        # Загружаем изображение с таймаутом
        try:
            response = requests.get(image_url, timeout=5, verify=False)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Ошибка загрузки изображения с {image_url}: {str(e)}")
            camera_states[camera_id].update({
                'connected': False,
                'processing': False,
                'last_error': f'Ошибка загрузки изображения: {str(e)}'
            })
            return jsonify({'status': 'error', 'message': 'Не удалось загрузить изображение'}), 400

        # Конвертируем в формат OpenCV
        try:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Не удалось декодировать изображение")
        except Exception as e:
            logging.error(f"Ошибка обработки изображения: {str(e)}")
            camera_states[camera_id].update({
                'connected': False,
                'processing': False,
                'last_error': f'Ошибка обработки изображения: {str(e)}'
            })
            return jsonify({'status': 'error', 'message': 'Не удалось обработать изображение'}), 400

        # Обрабатываем кадр через нейронные сети
        try:
            processed_frame = process_frame(frame, camera_id)
        except Exception as e:
            logging.error(f"Ошибка обработки кадра нейронными сетями: {str(e)}")
            camera_states[camera_id].update({
                'connected': True,
                'processing': False,
                'last_error': f'Ошибка обработки нейронными сетями: {str(e)}'
            })
            # Возвращаем оригинальный кадр в случае ошибки обработки
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return Response(buffer.tobytes(), mimetype='image/jpeg')

        # Конвертируем обратно в JPEG
        try:
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        except Exception as e:
            logging.error(f"Ошибка кодирования обработанного кадра: {str(e)}")
            camera_states[camera_id].update({
                'connected': True,
                'processing': False,
                'last_error': f'Ошибка кодирования кадра: {str(e)}'
            })
            return jsonify({'status': 'error', 'message': 'Ошибка кодирования кадра'}), 500

    except Exception as e:
        logging.error(f"Неожиданная ошибка обработки HTML-изображения: {str(e)}")
        if camera_id in camera_states:
            camera_states[camera_id].update({
                'connected': False,
                'processing': False,
                'last_error': f'Неожиданная ошибка: {str(e)}'
            })
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/camera/settings', methods=['GET'])
@login_required
def get_camera_settings():
    try:
        camera_id = request.args.get('camera_id')
        if not camera_id:
            return jsonify({'error': 'Не указан ID камеры'}), 400

        # Получаем настройки из config.json
        camera_settings = None
        for camera in config.get('cameras', []):
            if camera.get('id') == int(camera_id):
                camera_settings = camera.get('settings', {})
                break

        if not camera_settings:
            return jsonify({'error': 'Камера не найдена'}), 404

        # Преобразуем формат настроек для фронтенда
        settings = {
            'roi_enabled': camera_settings['roi']['enabled'],
            'roi_x': camera_settings['roi']['x'],
            'roi_y': camera_settings['roi']['y'],
            'roi_width': camera_settings['roi']['width'],
            'roi_height': camera_settings['roi']['height'],
            'brightness': camera_settings['brightness'],
            'contrast': camera_settings['contrast'],
            'frame_skip_enabled': camera_settings['frame_skip']['enabled'],
            'frame_skip_interval': camera_settings['frame_skip']['interval']
        }

        return jsonify(settings)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/settings', methods=['POST'])
@login_required
def update_camera_settings():
    try:
        data = request.json
        if not data or 'camera_id' not in data:
            return jsonify({'error': 'Не указан ID камеры'}), 400

        camera_id = int(data['camera_id'])
        
        # Находим камеру в конфигурации
        camera_found = False
        for camera in config.get('cameras', []):
            if camera.get('id') == camera_id:
                camera_found = True
                # Обновляем настройки
                camera['settings'] = {
                    'roi': {
                        'enabled': data.get('roi_enabled', False),
                        'x': data.get('roi_x', 0),
                        'y': data.get('roi_y', 0),
                        'width': data.get('roi_width', 100),
                        'height': data.get('roi_height', 100)
                    },
                    'brightness': data.get('brightness', 0),
                    'contrast': data.get('contrast', 0),
                    'frame_skip': {
                        'enabled': data.get('frame_skip_enabled', False),
                        'interval': data.get('frame_skip_interval', 1)
                    }
                }
                break

        if not camera_found:
            return jsonify({'error': 'Камера не найдена'}), 404

        # Валидация значений
        settings = camera['settings']
        if not isinstance(settings['roi']['enabled'], bool):
            return jsonify({'error': 'roi_enabled должен быть булевым значением'}), 400

        if not isinstance(settings['frame_skip']['enabled'], bool):
            return jsonify({'error': 'frame_skip_enabled должен быть булевым значением'}), 400

        for field in ['x', 'y', 'width', 'height']:
            if not isinstance(settings['roi'][field], int) or settings['roi'][field] < 0 or settings['roi'][field] > 100:
                return jsonify({'error': f'roi_{field} должен быть целым числом от 0 до 100'}), 400

        for field in ['brightness', 'contrast']:
            if not isinstance(settings[field], int) or settings[field] < -100 or settings[field] > 100:
                return jsonify({'error': f'{field} должен быть целым числом от -100 до 100'}), 400

        if not isinstance(settings['frame_skip']['interval'], int) or settings['frame_skip']['interval'] < 1 or settings['frame_skip']['interval'] > 10:
            return jsonify({'error': 'frame_skip_interval должен быть целым числом от 1 до 10'}), 400

        # Сохраняем конфигурацию
        save_config(config)

        # Преобразуем настройки для ответа
        response_settings = {
            'roi_enabled': settings['roi']['enabled'],
            'roi_x': settings['roi']['x'],
            'roi_y': settings['roi']['y'],
            'roi_width': settings['roi']['width'],
            'roi_height': settings['roi']['height'],
            'brightness': settings['brightness'],
            'contrast': settings['contrast'],
            'frame_skip_enabled': settings['frame_skip']['enabled'],
            'frame_skip_interval': settings['frame_skip']['interval']
        }

        return jsonify({'success': True, 'settings': response_settings})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/settings/reset', methods=['POST'])
@login_required
def reset_camera_settings():
    try:
        camera_id = request.args.get('camera_id')
        if not camera_id:
            return jsonify({'error': 'Не указан ID камеры'}), 400

        camera_id = int(camera_id)
        
        # Находим камеру в конфигурации
        camera_found = False
        for camera in config.get('cameras', []):
            if camera.get('id') == camera_id:
                camera_found = True
                # Сбрасываем настройки к значениям по умолчанию
                camera['settings'] = {
                    'roi': {
                        'enabled': False,
                        'x': 0,
                        'y': 0,
                        'width': 100,
                        'height': 100
                    },
                    'brightness': 0,
                    'contrast': 0,
                    'frame_skip': {
                        'enabled': False,
                        'interval': 1
                    }
                }
                break

        if not camera_found:
            return jsonify({'error': 'Камера не найдена'}), 404

        # Сохраняем конфигурацию
        save_config(config)

        # Преобразуем настройки для ответа
        settings = camera['settings']
        response_settings = {
            'roi_enabled': settings['roi']['enabled'],
            'roi_x': settings['roi']['x'],
            'roi_y': settings['roi']['y'],
            'roi_width': settings['roi']['width'],
            'roi_height': settings['roi']['height'],
            'brightness': settings['brightness'],
            'contrast': settings['contrast'],
            'frame_skip_enabled': settings['frame_skip']['enabled'],
            'frame_skip_interval': settings['frame_skip']['interval']
        }

        return jsonify({'success': True, 'settings': response_settings})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Инициализация базы данных
    init_db()
    initialize_models()
    
    app.run(host='0.0.0.0', port=5000, threaded=True)