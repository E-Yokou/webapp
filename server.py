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

# Global variables
coco_model = None
license_plate_detector = None
mot_tracker = None
vehicles = [2, 3, 5, 7]  # IDs of vehicles in COCO
frame_queue = queue.Queue(maxsize=2)
recognized_plates = set()
tracked_plates = {}
recognition_threshold = 0.85
is_streaming = False
current_camera_url = None
cap = None
processing_thread = None
stop_event = threading.Event()


def initialize_models():
    global coco_model, license_plate_detector, mot_tracker
    try:
        coco_model = YOLO('models/yolo11n.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
        license_plate_detector = YOLO('models/license_plate_detector.pt').to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        mot_tracker = Sort()
        logging.info("Models loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise


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

                        if plate_text and plate_score >= recognition_threshold:
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

            if not plate_text or plate_score < recognition_threshold:
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

        # Draw annotations on frame
        for track in track_ids:
            x1, y1, x2, y2, track_id = track
            plate_info = tracked_plates.get(track_id)

            if plate_info:
                # Draw vehicle bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Draw license plate text
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


def video_processing():
    global cap
    while not stop_event.is_set():
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                try:
                    processed_frame = process_frame(frame)
                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    frame_queue.put(buffer.tobytes())
                except Exception as e:
                    logging.error(f"Error processing frame: {e}")
        else:
            time.sleep(0.1)


def gen_frames():
    while True:
        frame_data = frame_queue.get()
        if frame_data is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/connect_camera', methods=['POST'])
def connect_camera():
    global cap, processing_thread, current_camera_url
    data = request.json
    camera_url = data.get('camera_url')

    if cap and cap.isOpened():
        cap.release()

    if camera_url:
        try:
            cap = cv2.VideoCapture(camera_url)
            if cap.isOpened():
                current_camera_url = camera_url
                if processing_thread is None:
                    processing_thread = threading.Thread(target=video_processing)
                    processing_thread.start()
                return jsonify({'status': 'success', 'message': 'Camera connected'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to open camera'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    else:
        return jsonify({'status': 'error', 'message': 'No camera URL provided'})


@app.route('/disconnect_camera', methods=['POST'])
def disconnect_camera():
    global cap, processing_thread, current_camera_url
    if cap and cap.isOpened():
        cap.release()
    current_camera_url = None
    return jsonify({'status': 'success', 'message': 'Camera disconnected'})


@app.route('/get_recognized_plates', methods=['GET'])
def get_recognized_plates():
    plates = [{'text': plate['plate_text'], 'score': plate['plate_score']}
              for plate in tracked_plates.values()]
    return jsonify({'plates': plates})


@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global recognition_threshold
    data = request.json
    threshold = float(data.get('threshold', 0.85))
    if 0.1 <= threshold <= 0.99:
        recognition_threshold = threshold
        return jsonify({'status': 'success', 'threshold': threshold})
    return jsonify({'status': 'error', 'message': 'Invalid threshold value'})


if __name__ == '__main__':
    initialize_models()
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        stop_event.set()
        if cap and cap.isOpened():
            cap.release()
        if processing_thread:
            processing_thread.join()