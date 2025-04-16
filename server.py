# file server.py

import json
import os
import time
import logging
import threading
import queue
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import torch
from sort.sort import Sort
from ultralytics import YOLO
from util import read_license_plate, get_car, insert_car_data, is_plate_inside_car, get_plate_center, get_car_center

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG_FILE = 'config.json'
DEFAULT_CONFIG = {
    'cameras': [],
    'current_camera': None,
    'threshold': 0.85
}

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


def initialize_models():
    global coco_model, license_plate_detector, mot_tracker
    coco_model = YOLO('models/yolo11n.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
    license_plate_detector = YOLO('models/license_plate_detector.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
    mot_tracker = Sort()


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

                # Save to database
                if plate_text != getattr(app, f'last_saved_plate_{camera_id}', None):
                    try:
                        _, buffer = cv2.imencode('.jpg', frame)
                        insert_car_data(
                            plate_text,
                            buffer.tobytes(),
                            "Car",
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            camera_id
                        )
                        setattr(app, f'last_saved_plate_{camera_id}', plate_text)
                    except Exception as e:
                        logging.error(f"Database save error: {e}")

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
    cap = cv2.VideoCapture(camera_url)
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
        'thread': threading.current_thread()
    }

    logging.info(f"Started processing for camera {camera_id}")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            try:
                processed_frame = process_frame(frame, camera_id)
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_queue.put(buffer.tobytes())
            except queue.Full:
                pass  # Skip frame if queue is full
            except Exception as e:
                logging.error(f"Error processing frame for camera {camera_id}: {e}")
        else:
            logging.warning(f"Camera {camera_id} returned no frame")
            time.sleep(0.1)

    cap.release()
    if camera_id in camera_instances:
        del camera_instances[camera_id]
    logging.info(f"Stopped processing for camera {camera_id}")


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
    return render_template('index.html', config=config)

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


@app.route('/api/cameras', methods=['GET', 'POST', 'DELETE'])
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


if __name__ == '__main__':
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