# uav_stream.py  (LIBRARY, memory-only, epoch-ready)  [方案A：在 streamer 內顯示預覽]
#
# 最小改動：
# - 加 preview 相關參數
# - capture_array() 後面加 imshow/waitKey
# - 按 q 離開
# - finally 做資源清理

import socket
import struct
import time
import os
import cv2
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from picamera2 import Picamera2


class UAVVideoStreamer:
    def __init__(
        self,
        get_video_key,
        gsn_ip,
        port=5005,
        chunk=4096,
        preview=False,
        window_name="UAV Camera",
        preview_wait=1,
    ):
        """
        get_video_key() -> (epoch, aes_key)

        preview:      True 就在 UAV 本地顯示相機畫面
        window_name:  視窗名稱
        preview_wait: cv2.waitKey 的等待時間(ms)，1 通常就夠
        """
        self.get_video_key = get_video_key
        self.gsn_ip = gsn_ip
        self.port = port
        self.chunk = chunk

        self.preview = preview
        self.window_name = window_name
        self.preview_wait = preview_wait

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        cam = Picamera2()
        cam.configure(cam.create_video_configuration(main={"size": (640, 480)}))
        cam.start()

        frame_id = 0
        print("[UAV] video stream started")

        try:
            while True:
                frame_id += 1

                # 1) capture
                frame = cam.capture_array()

                # 2) 本地預覽（最小新增）
                if self.preview:
                    # 如果你發現顏色顛倒，再改成：
                    # cv2.imshow(self.window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    cv2.imshow(self.window_name, frame)
                    key = cv2.waitKey(self.preview_wait) & 0xFF
                    if key == ord("q"):
                        print("[UAV] preview quit (q pressed)")
                        break

                # 3) encode jpg
                ok, jpg = cv2.imencode(".jpg", frame)
                if not ok:
                    continue

                # 4) get key
                epoch, aes_key = self.get_video_key()
                if aes_key is None:
                    continue

                # 5) encrypt (nonce + ciphertext)
                aes = AESGCM(aes_key)
                nonce = os.urandom(12)
                encrypted = nonce + aes.encrypt(nonce, jpg.tobytes(), None)

                # 6) chunk + send
                chunks = [
                    encrypted[i : i + self.chunk]
                    for i in range(0, len(encrypted), self.chunk)
                ]
                ts = time.time()

                header = (
                    b"H"
                    + struct.pack("!I I d I 1s", frame_id, epoch, ts, len(chunks), b"A")
                )
                sock.sendto(header, (self.gsn_ip, self.port))

                for idx, ch in enumerate(chunks):
                    pkt = b"P" + struct.pack("!I I", frame_id, idx) + ch
                    sock.sendto(pkt, (self.gsn_ip, self.port))

        finally:
            # 清理資源，避免相機/視窗卡住
            try:
                cam.stop()
            except:
                pass
            try:
                sock.close()
            except:
                pass
            if self.preview:
                try:
                    cv2.destroyWindow(self.window_name)
                except:
                    pass
