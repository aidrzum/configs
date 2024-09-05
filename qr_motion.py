import io
import logging
import os
import signal
import subprocess
import sys
from configparser import ConfigParser
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from threading import Thread, Lock, Condition
from time import time, sleep

import cv2
import libcamera
import numpy as np
import requests
from picamera2 import Picamera2, MappedArray
from picamera2.encoders import H264Encoder, MJPEGEncoder
from picamera2.outputs import FileOutput
from pyzbar import pyzbar

def load_config():
    config = ConfigParser()
    config.add_section('camera')
    config.set('camera', 'width', '960')
    config.set('camera', 'height', '720')
    config.set('camera', 'fps', '15.0')
    config.set('camera', 'mode', '1')
    config.set('camera', 'hflip', '0')
    config.set('camera', 'vflip', '0')
    config.set('camera', 'apply_scaler_crop', 'no')
    config.set('camera', 'scaler_crop_x', '0')
    config.set('camera', 'scaler_crop_y', '2')
    config.set('camera', 'scaler_crop_w', '3280')
    config.set('camera', 'scaler_crop_h', '2460')
    config.add_section('output')
    config.set('output', 'path_rec', os.path.expanduser('~/cam/rec/'))
    config.set('output', 'path_log', os.path.expanduser('~/cam/log/'))
    config.set('output', 'max_used_space', str(32 * 1024 ** 3))
    config.set('output', 'min_length_video', '5.0')
    config.set('output', 'convert_mp4', 'yes')
    config.read(os.path.expanduser('~/.config/picam2_surveillance.ini'))
    return config

def init_logging(config):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    loghandler = TimedRotatingFileHandler(
        config.get('output', 'path_log') + 'picam2_surveillance.log',
        'midnight')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    loghandler.setFormatter(formatter)
    logger.addHandler(loghandler)

class FfmpegThread(Thread):

    def __init__(self, fps, filepath):
        super().__init__()
        self._fps = fps
        self._filepath = filepath

    def run(self):
        command = ['ffmpeg', '-loglevel', 'error', '-y']
        video_input = ['-r', str(self._fps), '-i', self._filepath]
        video_codec = ['-c:v', 'copy']
        output = ['-movflags', 'faststart', self._filepath[:-4] + 'mp4']
        command += video_input + video_codec + output
        ffmpeg = subprocess.Popen(command)
        if ffmpeg.wait() == 0:
            os.remove(self._filepath)

class StreamingOutput(io.BufferedIOBase):

    def __init__(self):
        super().__init__()
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class SurveillanceCamera:

    def __init__(self, config):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._width = config.getint('camera', 'width')
        self._height = config.getint('camera', 'height')
        self._fps = config.getfloat('camera', 'fps')
        self._sensor_mode = config.getint('camera', 'mode')
        self._hflip = config.getint('camera', 'hflip')
        self._vflip = config.getint('camera', 'vflip')

        self._scaler_crop = None
        if config.getboolean('camera', 'apply_scaler_crop'):
            self._scaler_crop = (
                config.getint('camera', 'scaler_crop_x'),
                config.getint('camera', 'scaler_crop_y'),
                config.getint('camera', 'scaler_crop_w'),
                config.getint('camera', 'scaler_crop_h'))

        self._path_output = config.get('output', 'path_rec')
        self._min_length_video = config.getfloat('output', 'min_length_video')
        self._convert_mp4 = config.getboolean('output', 'convert_mp4')

        self._picam2 = Picamera2()
        self._capturing = False
        self._encoder_recording = None
        self._filepath_recording = None
        self._encoder_serving = None
        self._output = None
        self._lock_serving = Lock()
        self._counter_serving = 0

    @staticmethod
    def _annotate_timestamp(request):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        position = (8, 16)
        font = cv2.FONT_HERSHEY_PLAIN
        color = (255, 255, 255)
        with MappedArray(request, "main") as frame:
            cv2.putText(frame.array, timestamp, position, font, 1.0, color)

    def _has_frame_changed(self, frame_1, frame_2, min_changed_pixels):
        frame_diff = np.abs(np.subtract(frame_1.astype(int), frame_2))
        num_changed_pixels = (frame_diff > 32).sum()
        return num_changed_pixels > min_changed_pixels

    def get_serial(self):
        cpu_serial = "0000000000000000"
        try:
            f = open('/proc/cpuinfo', 'r')
            for line in f:
                if line[0:6] == 'Serial':
                    cpuserial = line[10:26]
            f.close()
        except Exception as e:
            print(f"Error: Unable to cpu serial number- {e}")

        return cpu_serial

    def has_qr_code(self, frame):
        qr_codes = pyzbar.decode(frame)
        self._logger.info("Length %s", len(qr_codes))
        return len(qr_codes) > 0

    def _detect_qr_codes(self, frame):
        qr_codes = pyzbar.decode(frame)
        self._logger.info("QR code: ", qr_codes)
        for qr_code in qr_codes:
            self._logger.info("QR Detected: ", qr_code.data.decode('utf-8'))
            #(x, y, w, h) = qr_code.rect
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            qr_data = qr_code.data.decode('utf-8')
            barcode_type = qr_code.type
            params = {
                'auth': 'VqIJQvhPVPQOGUvDzBgSHvNeMAojle',
                'qrcode': qr_data,
                'serialNumber': self.get_serial()
            }
            x = requests.get("https://zy.zumrides.com/api/api_get_qr_cycle_stop.php", params=params)
            text = "{} ({})".format(qr_data, barcode_type)
            self._logger.info("Response: %s", x)
            self._logger.info(text)

    def start_recording(self):
        dt_now = datetime.now()
        dirname = dt_now.strftime('%Y-%m-%d')
        basename = dt_now.strftime('%Y-%m-%d_%H%M%S')
        if not os.path.exists(self._path_output + dirname):
            os.makedirs(self._path_output + dirname)
        self._filepath_recording = (
                self._path_output + dirname + '/' + basename + '.h264')
        self._logger.info("Start recording to %s", self._filepath_recording)
        self._encoder_recording = H264Encoder()
        self._encoder_recording.framerate = self._fps
        # The FfmpegOutput creates a jitter because the timestamps of the
        # video frames are not forwarded. Ffmpeg stamps the video frames
        # as it gets them. Unfortunately, ffmpeg takes a second to start.
        # output = FfmpegOutput(filepath)
        output = FileOutput(self._filepath_recording)
        self._picam2.start_encoder(self._encoder_recording, output)

    def stop_recording(self):
        self._picam2.stop_encoder(self._encoder_recording)
        self._encoder_recording = None
        self._logger.info("Recording stopped")
        if self._convert_mp4:
            self._logger.info("Converting %s", self._filepath_recording)
            FfmpegThread(self._fps, self._filepath_recording).start()

    def start_capturing(self):
        self._logger.info(
            "Selected sensor mode: %s",
            self._picam2.sensor_modes[self._sensor_mode])
        config = self._picam2.create_video_configuration(

            # Picamera2 Manual - Appendix A: Pixel and image formats
            main={
                'format': 'BGR888',
                'size': (self._width, self._height)
            },
            lores={
                'format': 'YUV420',
                'size': (int(self._width / 4), int(self._height / 4))
            },

            # Although the IDs of the sensor modes are different in picamera,
            # there is good documentation to get an idea about sensor modes:
            # https://picamera.readthedocs.io/en/latest/fov.html#sensor-modes
            raw=self._picam2.sensor_modes[self._sensor_mode],

            # Picamera2 Manual - Appendix B: Camera configuration parameters
            transform=libcamera.Transform(hflip=self._hflip, vflip=self._vflip),

            # Picamera2 Manual - Appendix C: Camera controls
            controls={
                'FrameDurationLimits': (int(1e6 / self._fps), int(1e6 / self._fps))
            }
        )
        if self._scaler_crop is not None:
            config['controls']['ScalerCrop'] = self._scaler_crop
        self._logger.info("Used config: %s", config)
        self._picam2.configure(config)
        self._picam2.post_callback = SurveillanceCamera._annotate_timestamp

        self._logger.info("Start capturing")
        self._capturing = True
        self._picam2.start()

        frame_curr = None
        frame_prev = None
        time_last_change = None
        while self._capturing:
            frame = self._picam2.capture_array()
            (lores_w, lores_h) = self._picam2.video_configuration.lores.size
            frame_curr = self._picam2.capture_buffer('lores')[:lores_w * lores_h]
            frame_curr = frame_curr.reshape(lores_h, lores_w)
            if frame_prev is not None:
                if self._encoder_recording is None:
                    if self._has_frame_changed(frame_prev, frame_curr, 32):
                        time_last_change = time()
                        self.start_recording()
                else:
                    if self._has_frame_changed(frame_prev, frame_curr, 8):
                        time_last_change = time()
                    elif time_last_change + self._min_length_video < time():
                        self.stop_recording()
            frame_prev = frame_curr
            
            # Detect QR codes in the current frame
            if self.has_qr_code(frame):
                self._detect_qr_codes(frame)
                
        if self._encoder_recording is not None:
            self.stop_recording()
        self._picam2.stop()
        self._logger.info("Capturing stopped")

    def stop_capturing(self):
        self._capturing = False

class StorageCleaner:

    def __init__(self, path, max_used_space):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._path = path
        self._max_used_space = max_used_space

    def _get_filelist(self):
        filelist = []
        for dirpath, _, filenames in os.walk(self._path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                filemtime = os.path.getmtime(filepath)
                filesize = os.path.getsize(filepath)
                fileinfo = (filemtime, filesize, filepath)
                filelist.append(fileinfo)
        filelist.sort()
        return filelist

    def remove_oldest_files(self):
        filelist = self._get_filelist()
        sum_file_sizes = sum(filesize for _, filesize, _ in filelist)
        for fileinfo in filelist:
            if sum_file_sizes > self._max_used_space:
                filepath = fileinfo[2]
                self._logger.info("Removing %s", filepath)
                os.remove(filepath)
                sum_file_sizes -= fileinfo[1]
            else:
                break

    def remove_empty_dirs(self):
        for dir_path, dir_names, filenames in os.walk(self._path):
            if (not dir_names) and (not filenames):
                self._logger.info("Removing empty dir %s", dir_path)
                os.rmdir(dir_path)

class StorageCleanerThread(Thread):

    def __init__(self, config):
        super().__init__()
        self._cleaning = False
        self._cleaner = StorageCleaner(
            config.get('output', 'path_rec'),
            config.getint('output', 'max_used_space'))

    def run(self):
        self._cleaning = True
        while self._cleaning:
            self._cleaner.remove_oldest_files()
            self._cleaner.remove_empty_dirs()
            for _ in range(60):
                sleep(1)
                if not self._cleaning:
                    break

    def stop_cleaning(self):
        self._cleaning = False

class SignalHandler:

    def __init__(self, surveillance_camera):
        self._surveillance_camera = surveillance_camera
        self._logger = logging.getLogger(self.__class__.__name__)

    def _handle_signal(self, signum, frame):
        self._logger.info("Handling signal %s", signum)
        self._surveillance_camera.stop_capturing()

    def register_signal_handlers(self):
        signal.signal(signal.SIGINT, self._handle_signal)  # Ctrl-C
        signal.signal(signal.SIGTERM, self._handle_signal)  # kill

def main():
    config = load_config()
    if not os.path.exists(config.get('output', 'path_rec')):
        os.makedirs(config.get('output', 'path_rec'))
    if not os.path.exists(config.get('output', 'path_log')):
        os.makedirs(config.get('output', 'path_log'))
    init_logging(config)
    logging.info("Logging initialized")
    surveillance_camera = SurveillanceCamera(config)

    storage_cleaner_thread = None
    if config.getint('output', 'max_used_space') > 0:
        storage_cleaner_thread = StorageCleanerThread(config)
        storage_cleaner_thread.start()
    signal_handler = SignalHandler(surveillance_camera)
    signal_handler.register_signal_handlers()
    surveillance_camera.start_capturing()
    if storage_cleaner_thread:
        storage_cleaner_thread.stop_cleaning()

if __name__ == '__main__':
    sys.exit(main())
