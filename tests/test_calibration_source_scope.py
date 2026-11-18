"""A laptop's focal calibration must not create metre readings for foreign media."""
from config import load_config
from detection import Detection
from helpers import FakeDetector,running,wait_for
from test_analyze import encoded,scene


PERSON=Detection(1,'People',.8,(600,100,700,500))
CALIBRATION=dict(focal_px=900,frame_height=720)


def test_uploaded_image_does_not_use_live_camera_intrinsics():
    import io
    with running({'local':FakeDetector('yolov8n',detections=(PERSON,))}) as (state,client):
        state.calibration=dict(CALIBRATION)
        response=client.post('/analyze',data={'image':(io.BytesIO(encoded(scene(),'jpg')),'foreign-camera.jpg'),'debug':'1'},content_type='multipart/form-data')
        assert response.status_code==200
        detection=response.json['detections'][0]
        assert detection.get('distance_est_m') is None
        assert 'camera calibration' in detection.get('distance_note','')


def test_matching_live_source_keeps_known_height_estimate():
    with running({'local':FakeDetector('yolov8n',detections=(PERSON,))}) as (state,client):
        state.calibration=dict(CALIBRATION)
        assert wait_for(lambda:client.get('/status').json['detections'] and client.get('/status').json['detections'][0].get('distance_est_m')==3.8)


def test_page_limitations_follow_active_four_class_scope():
    with running({'local':FakeDetector('yolov8n')}) as (state,client):
        page=client.get('/').get_data(as_text=True)
        assert 'Active detection classes:' in page
        assert 'Pillar' not in page and 'Trained on five' not in page
        for name in load_config()['classes']:assert name in page
