# gsn_key_matcher.py  (LIBRARY, memory-only)

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class LiveKDRPlotter:
    def __init__(self, interval_ms=500):
        self.interval_ms = interval_ms
        self.x = []
        self.kdr_raw = []
        self.kdr_corr = []
        self.idx = 0

        self.fig, self.ax = plt.subplots()

    def update(self, uav_raw, gsn_raw, corrected):
        def kdr(a, b):
            L = min(len(a), len(b))
            return sum(a[i] != b[i] for i in range(L)) / L

        self.x.append(self.idx)
        self.kdr_raw.append(kdr(uav_raw, gsn_raw) * 100)
        self.kdr_corr.append(kdr(uav_raw, corrected) * 100)
        self.idx += 1

    def _draw(self, _):
        self.ax.clear()
        self.ax.plot(self.x, self.kdr_raw, label="UAV vs GSN Raw")
        self.ax.plot(self.x, self.kdr_corr, label="UAV vs Corrected")
        self.ax.set_ylim(0, 100)
        self.ax.legend()

    def start(self):
        self.ani = animation.FuncAnimation(
            self.fig, self._draw, interval=self.interval_ms
        )
        plt.show()
