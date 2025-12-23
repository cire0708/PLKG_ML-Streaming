# gsn_receiver.py  (LIBRARY, memory-only, epoch-ready)

import socket
import struct
import threading
import queue
import time
import numpy as np
import cv2
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class GSNReceiver:
    def __init__(self, get_aes_key, on_frame, video_port=5005):
        """
        get_aes_key(epoch) -> bytes
        on_frame(frame_bgr, latency_ms)
        """
        self.get_aes_key = get_aes_key
        self.on_frame = on_frame
        self.video_port = video_port
        self.frames = {}
        self.queue = queue.Queue()

    def start(self):
        threading.Thread(target=self._recv, daemon=True).start()
        threading.Thread(target=self._show, daemon=True).start()

    def _recv(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.video_port))

        while True:
            data, _ = sock.recvfrom(65535)
            t = data[0:1]

            if t == b"H":
                fid, epoch, ts, cnt, _ = struct.unpack("!I I d I 1s", data[1:])
                self.frames[fid] = {"epoch": epoch, "ts": ts, "pkts": {}, "max": cnt}

            elif t == b"P":
                fid, pid = struct.unpack("!I I", data[1:9])
                if fid not in self.frames:
                    continue

                self.frames[fid]["pkts"][pid] = data[9:]
                info = self.frames[fid]

                if len(info["pkts"]) == info["max"]:
                    blob = b"".join(info["pkts"][i] for i in range(info["max"]))
                    self.queue.put((info["epoch"], info["ts"], blob))
                    self.frames.pop(fid)

    def _show(self):
        while True:
            epoch, ts, blob = self.queue.get()
            key = self.get_aes_key(epoch)
            if key is None:
                continue

            nonce, cipher = blob[:12], blob[12:]
            try:
                jpg = AESGCM(key).decrypt(nonce, cipher, None)
            except:
                continue

            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            latency = (time.time() - ts) * 1000
            self.on_frame(frame, latency)
