import argparse
import os
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
QUAD_PATH = "/Quadcopter"
BASE_ALIAS = "base"
TARGET_ALIAS = "target"
SCANNER_ALIAS = "fast3DLaserScanner"

CAMERA_TILT_DEG = 45.0             # sudut kamera dari vertikal (0 = ke bawah, 90 = mendatar)

# --- kontrol manual ---
MOVE_SPEED = 0.8                   # m/s maju/mundur/geser
CLIMB_SPEED = 0.6                  # m/s naik/turun
YAW_RATE_DEG = 60.0                # derajat/detik putar
TARGET_LEAD_MAX = 0.7              # jarak maks target di depan drone (m)
Z_MIN, Z_MAX = 0.3, 3.0            # batas ketinggian (m)

# --- scanning (dibuat rapat) ---
SCAN_EVERY = 0.10                  # detik waktu simulasi (lebih sering = lebih banyak titik)
MIN_RANGE, MAX_RANGE = 0.20, 6.0
PIXEL_STRIDE = 1                   # 1 = pakai semua pixel depth
RAW_GRID = 0.02                    # point cloud mentah dirapikan ke grid 2 cm (hemat memori)
FLIP_X = True                      # jika peta ter-mirror kiri-kanan -> False

# --- panel kontrol (jendela keybind + telemetri) ---
CONTROL_PANEL = True

# --- tampilan realtime ---
LIVE_VIEW = True
LIVE_EVERY_S = 0.7                 # perbarui jendela live tiap ... detik (waktu nyata)
LIVE_MAX_POINTS = 25000            # batas titik yang digambar di live (biar ringan)

# --- post process ---
VOXEL_SIZE = 0.05                  # voxel kecil = titik lebih banyak/rapat
MIN_PTS_PER_VOXEL = 1              # naikkan (mis. 2-3) jika ingin buang noise
CMAP = "RdYlGn_r"                  # hijau = rendah, merah = tinggi
OUT_DIR = "output_mapping"


# ----------------------------------------------------------------------------
# Helper API CoppeliaSim (kompatibel beberapa versi)
# ----------------------------------------------------------------------------
def connect():
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError:
        from zmqRemoteApi import RemoteAPIClient
    client = RemoteAPIClient()
    sim = client.require("sim") if hasattr(client, "require") else client.getObject("sim")
    return client, sim


def find_by_alias(sim, root, alias):
    for h in sim.getObjectsInTree(root, sim.handle_all, 0):
        if sim.getObjectAlias(h) == alias:
            return h
    return None


def _try_both(sim, name, handle, value):
    """Panggil sim.<name>(handle, value, world); fallback ke urutan argumen API lama."""
    f = getattr(sim, name)
    value = [float(v) for v in value]
    try:
        f(handle, value, sim.handle_world)
    except Exception:
        f(handle, sim.handle_world, value)


def set_position(sim, h, pos):
    _try_both(sim, "setObjectPosition", h, pos)


def set_orientation(sim, h, euler):
    _try_both(sim, "setObjectOrientation", h, euler)


def set_matrix(sim, h, m12):
    _try_both(sim, "setObjectMatrix", h, m12)


def get_position(sim, h):
    return np.array(sim.getObjectPosition(h, sim.handle_world), dtype=float)


def get_matrix(sim, h):
    return np.array(sim.getObjectMatrix(h, sim.handle_world), dtype=float).reshape(3, 4)


def _fparam(sim, h, name):
    pid = getattr(sim, name)
    try:
        return float(sim.getObjectFloatParam(h, pid))
    except Exception:
        return float(sim.getObjectFloatParameter(h, pid)[1])


def sensor_info(sim, h):
    return dict(handle=h, alias=sim.getObjectAlias(h),
                near=_fparam(sim, h, "visionfloatparam_near_clipping"),
                far=_fparam(sim, h, "visionfloatparam_far_clipping"),
                angle=_fparam(sim, h, "visionfloatparam_perspective_angle"))


def read_depth(sim, info):
    """Depth ternormalisasi (ry, rx) dalam [0..1]; 1 = tidak ada hit."""
    if info.get("explicit", True):
        try:
            sim.handleVisionSensor(info["handle"])
        except Exception:
            info["explicit"] = False      # sensor dirender otomatis tiap step
    buf, res = sim.getVisionSensorDepth(info["handle"], 1)
    if isinstance(buf, (bytes, bytearray)):
        d = np.frombuffer(buf, dtype=np.float32)
    else:
        d = np.asarray(buf, dtype=np.float32)
    rx, ry = int(res[0]), int(res[1])
    d = d.reshape(ry, rx)
    if np.nanmax(d) > 1.0001:             # ternyata meter -> normalisasikan
        d = np.clip((d - info["near"]) / (info["far"] - info["near"]), 0.0, 1.0)
    return d


def depth_to_cam_points(depth, info, flip_x=FLIP_X):
    ry, rx = depth.shape
    z = info["near"] + depth * (info["far"] - info["near"])
    f = (max(rx, ry) / 2.0) / np.tan(info["angle"] / 2.0)
    u = np.arange(rx) + 0.5 - rx / 2.0
    v = np.arange(ry) + 0.5 - ry / 2.0
    U, V = np.meshgrid(u, v)
    x = (-1.0 if flip_x else 1.0) * U / f * z
    y = V / f * z
    pts = np.stack([x, y, z], axis=-1)
    valid = depth < 0.999
    rng = np.linalg.norm(pts, axis=-1)
    valid &= (rng >= MIN_RANGE) & (rng <= MAX_RANGE)
    return pts, valid


def to_world(sim, handle, pts):
    M = get_matrix(sim, handle)
    return pts @ M[:, :3].T + M[:, 3]


# ----------------------------------------------------------------------------
# Rotasi: hadapkan sensor ke bawah
# ----------------------------------------------------------------------------
def rot_between(a, b):
    """Matriks rotasi minimal yang memutar vektor satuan a menjadi b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:                     # berlawanan: putar 180 derajat
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(a, b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def aim_sensors(sim, sensors, base, tilt_deg):
    """Arahkan sensor ke bawah, miring tilt_deg dari vertikal ke arah depan drone."""
    Mb = get_matrix(sim, base)
    fwd = Mb[:, 0].copy()
    fwd[2] = 0.0
    fwd = fwd / np.linalg.norm(fwd) if np.linalg.norm(fwd) > 1e-6 else np.array([1.0, 0.0, 0.0])
    t = np.radians(tilt_deg)
    want = np.cos(t) * np.array([0.0, 0.0, -1.0]) + np.sin(t) * fwd
    for info in sensors:
        M = get_matrix(sim, info["handle"])
        Rn = rot_between(M[:, 2], want) @ M[:, :3]
        set_matrix(sim, info["handle"], np.hstack([Rn, M[:, 3:4]]).ravel())
        axis = get_matrix(sim, info["handle"])[:, 2]
        print("Sensor '%s' dimiringkan %.0f deg dari vertikal, arah pandang = %s"
              % (info["alias"], tilt_deg, np.round(axis, 2)))


# ----------------------------------------------------------------------------
# Panel kontrol: keybind yang menyala saat ditekan + telemetri
# ----------------------------------------------------------------------------
PANEL_KEYS = ("w", "a", "s", "d", "space", "ctrl", "left", "right", "esc")


class ControlPanel:
    W, H = 520, 650
    BG, PANEL, KEY, EDGE = "#0d1117", "#161b26", "#232b3d", "#33415f"
    CYAN, PINK, GREEN = "#00e5ff", "#ff4d8d", "#39ff88"
    TXT, DIM = "#e6edf7", "#6b7a99"
    GLOW = {"#00e5ff": "#06323c", "#ff4d8d": "#42102a"}
    FONT = "Segoe UI"

    def __init__(self):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Drone Control")
        self.root.geometry("%dx%d+40+40" % (self.W, self.H))
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG)
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass
        self.cv = tk.Canvas(self.root, width=self.W, height=self.H, bg=self.BG, highlightthickness=0)
        self.cv.pack()
        self.alive = True
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.keys, self.on = {}, {}
        self.last_hud = 0.0
        self._build()
        self.root.update()

    # -- gambar dasar ------------------------------------------------------
    def _rr(self, x1, y1, x2, y2, r, **kw):
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
               x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return self.cv.create_polygon(pts, smooth=True, **kw)

    def _section(self, y, text, x_line):
        self.cv.create_text(30, y, text=text, anchor="w", fill=self.DIM, font=(self.FONT, 9, "bold"))
        self.cv.create_line(x_line, y, self.W - 30, y, fill="#222b3f", width=2)

    def _key(self, name, x, y, w, h, label, sub=None, accent=None, size=15):
        accent = accent or self.CYAN
        glow = self._rr(x - 6, y - 6, x + w + 6, y + h + 6, 18,
                        fill=self.GLOW[accent], outline="", state="hidden")
        body = self._rr(x, y, x + w, y + h, 13, fill=self.KEY, outline=self.EDGE, width=2)
        dy = -8 if sub else 0
        txt = self.cv.create_text(x + w / 2, y + h / 2 + dy, text=label, fill=self.TXT,
                                  font=(self.FONT, size, "bold"))
        subt = None
        if sub:
            subt = self.cv.create_text(x + w / 2, y + h / 2 + 13, text=sub, fill=self.DIM,
                                       font=(self.FONT, 8, "bold"))
        self.keys[name] = dict(glow=glow, body=body, txt=txt, sub=subt, accent=accent)
        self.on[name] = False

    def _build(self):
        cv = self.cv
        # header
        cv.create_text(30, 34, text="DRONE CONTROL", anchor="w", fill=self.CYAN,
                       font=(self.FONT, 22, "bold"))
        cv.create_text(30, 62, text="3D Mapping  \u2022  CoppeliaSim", anchor="w", fill=self.DIM,
                       font=(self.FONT, 10))
        self.dot = cv.create_oval(378, 32, 392, 46, fill=self.GREEN, outline="")
        cv.create_text(490, 39, text="SCANNING", anchor="e", fill=self.TXT, font=(self.FONT, 10, "bold"))

        # controls
        self._section(92, "C O N T R O L S", 160)
        K = 66
        self._key("w", 114, 112, K, K, "W", "FWD")
        self._key("a", 40, 186, K, K, "A", "LEFT")
        self._key("s", 114, 186, K, K, "S", "BACK")
        self._key("d", 188, 186, K, K, "D", "RIGHT")
        cv.create_text(402, 160, text="R O T A T E", fill=self.DIM, font=(self.FONT, 9, "bold"))
        self._key("left", 330, 186, K, K, "\u2190", "CCW", size=20)
        self._key("right", 404, 186, K, K, "\u2192", "CW", size=20)
        self._key("space", 40, 270, 270, 58, "SPACE", "\u25B2  ASCEND")
        self._key("ctrl", 326, 270, 154, 58, "CTRL", "\u25BC  DESCEND")
        self._key("esc", 40, 338, 440, 52, "ESC", "\u25A0  FINISH  &  SHOW FULL RESULT",
                  accent=self.PINK)

        # telemetry
        self._section(420, "T E L E M E T R Y", 190)
        self._rr(30, 436, 490, 634, 16, fill=self.PANEL, outline="#222b3f", width=2)
        cx, cy, r = 112, 534, 62
        cv.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#2c3a58", width=2)
        cv.create_oval(cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2, outline="#1f2a42")
        for ang in (0, 90, 180, 270):
            a = np.radians(ang)
            cv.create_line(cx + (r - 8) * np.cos(a), cy - (r - 8) * np.sin(a),
                           cx + r * np.cos(a), cy - r * np.sin(a), fill="#4a5a7d", width=2)
        cv.create_text(cx, cy - r - 12, text="Y+", fill=self.DIM, font=(self.FONT, 8, "bold"))
        cv.create_text(cx + r + 14, cy, text="X+", fill=self.DIM, font=(self.FONT, 8, "bold"))
        self.arrow = cv.create_line(cx, cy, cx + r - 6, cy, fill=self.CYAN, width=4,
                                    arrow=self.tk.LAST, arrowshape=(12, 14, 6), capstyle="round")
        cv.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=self.TXT, outline="")
        self.cx, self.cy, self.r = cx, cy, r
        self.t_hdg = cv.create_text(cx, 614, text="HDG 0\u00B0", fill=self.TXT, font=(self.FONT, 10, "bold"))

        cv.create_text(212, 458, text="ALTITUDE", anchor="w", fill=self.DIM, font=(self.FONT, 8, "bold"))
        self.t_alt = cv.create_text(476, 458, text="0.00 m", anchor="e", fill=self.CYAN,
                                    font=(self.FONT, 13, "bold"))
        self._rr(212, 474, 476, 490, 6, fill=self.BG, outline="#222b3f")
        self.bar = cv.create_rectangle(214, 476, 220, 488, fill=self.CYAN, outline="")
        cv.create_text(212, 502, text="%.1f" % Z_MIN, anchor="w", fill=self.DIM, font=(self.FONT, 7))
        cv.create_text(476, 502, text="%.1f" % Z_MAX, anchor="e", fill=self.DIM, font=(self.FONT, 7))

        rows = [("POSITION", 534), ("POINTS", 570), ("SIM TIME", 604)]
        self.rows = {}
        for label, y in rows:
            cv.create_line(212, y - 17, 476, y - 17, fill="#222b3f")
            cv.create_text(212, y, text=label, anchor="w", fill=self.DIM, font=(self.FONT, 8, "bold"))
            self.rows[label] = cv.create_text(476, y, text="-", anchor="e", fill=self.TXT,
                                              font=("Consolas", 11, "bold"))

    # -- update ------------------------------------------------------------
    def _set_key(self, name, on):
        k = self.keys[name]
        self.on[name] = on
        self.cv.itemconfigure(k["glow"], state="normal" if on else "hidden")
        self.cv.itemconfigure(k["body"], fill=k["accent"] if on else self.KEY,
                              outline=k["accent"] if on else self.EDGE)
        self.cv.itemconfigure(k["txt"], fill="#04121a" if on else self.TXT)
        if k["sub"]:
            self.cv.itemconfigure(k["sub"], fill="#06323c" if on else self.DIM)

    def update(self, pressed, hud=None):
        """pressed: {nama_tombol: bool}; hud: dict x,y,z,yaw(rad),points,t."""
        if not self.alive:
            return
        for name, v in pressed.items():
            v = bool(v)
            if name in self.on and self.on[name] != v:
                self._set_key(name, v)
        now = time.time()
        if hud is not None and now - self.last_hud > 0.1:
            self.last_hud = now
            ex = self.cx + (self.r - 6) * np.cos(hud["yaw"])
            ey = self.cy - (self.r - 6) * np.sin(hud["yaw"])
            self.cv.coords(self.arrow, self.cx, self.cy, ex, ey)
            self.cv.itemconfigure(self.t_hdg, text="HDG %d\u00B0" % (np.degrees(hud["yaw"]) % 360))
            frac = float(np.clip((hud["z"] - Z_MIN) / (Z_MAX - Z_MIN), 0.0, 1.0))
            warn = frac < 0.05 or frac > 0.95
            col = self.PINK if warn else self.CYAN
            self.cv.coords(self.bar, 214, 476, 214 + max(6.0, frac * 260), 488)
            self.cv.itemconfigure(self.bar, fill=col)
            self.cv.itemconfigure(self.t_alt, text="%.2f m" % hud["z"], fill=col)
            self.cv.itemconfigure(self.rows["POSITION"], text="x %+.2f   y %+.2f" % (hud["x"], hud["y"]))
            self.cv.itemconfigure(self.rows["POINTS"], text="{:,}".format(int(hud["points"])))
            self.cv.itemconfigure(self.rows["SIM TIME"], text="%.1f s" % hud["t"])
            a = 0.5 + 0.5 * np.sin(now * 4.0)                 # titik 'SCANNING' berdenyut
            g = int(70 + 185 * a)
            self.cv.itemconfigure(self.dot, fill="#%02x%02x%02x" % (int(g * 0.2), g, int(g * 0.5)))
        try:
            self.root.update()
        except self.tk.TclError:
            self.alive = False

    def _on_close(self):
        self.alive = False
        try:
            self.root.destroy()
        except Exception:
            pass

    def close(self):
        self.alive = False
        try:
            self.root.destroy()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Tampilan realtime
# ----------------------------------------------------------------------------
class LiveView:
    """Jendela live: peta 3D (voxel berwarna tinggi) + tampak atas + posisi drone."""
    OFF = 1 << 19
    MASK = (1 << 20) - 1

    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig = plt.figure("LIVE - tekan ESC di keyboard untuk selesai", figsize=(14, 7))
        self.ax3 = self.fig.add_subplot(121, projection="3d")
        self.ax2 = self.fig.add_subplot(122)
        self.ax3.set_title("Live 3D Map")
        self.ax3.set_xlabel("X")
        self.ax3.set_ylabel("Y")
        self.ax3.set_zlabel("Z")
        self.ax2.set_title("Live tampak atas")
        self.ax2.set_xlabel("X")
        self.ax2.set_ylabel("Y")
        self.ax2.set_aspect("equal")
        self.ax2.grid(alpha=0.3)
        z0 = np.zeros(1)
        self.sc3 = self.ax3.scatter(z0, z0, z0, c=z0, cmap=CMAP, s=2, depthshade=False, vmin=0, vmax=1)
        self.sc2 = self.ax2.scatter(z0, z0, c=z0, cmap=CMAP, s=3, vmin=0, vmax=1)
        self.line3, = self.ax3.plot([0], [0], [0], color="blue", lw=2)
        self.line2, = self.ax2.plot([0], [0], color="blue", lw=1.5)
        self.dr3 = self.ax3.scatter(z0, z0, z0, c="red", s=50)
        self.dr2, = self.ax2.plot([0], [0], "ro", ms=8)
        self.fig.colorbar(self.sc3, ax=self.ax3, shrink=0.55, pad=0.08, label="Tinggi z (m)")
        self.keys = np.zeros(0, dtype=np.int64)
        self.half = 3.0
        self._limits()
        self.last = 0.0
        self.alive = True
        plt.show(block=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _limits(self, zmax=2.1):
        h = self.half
        self.ax3.set_xlim(-h, h)
        self.ax3.set_ylim(-h, h)
        self.ax3.set_zlim(-0.3, max(2.1, zmax))
        self.ax2.set_xlim(-h, h)
        self.ax2.set_ylim(-h, h)

    def _encode(self, pts):
        k = np.floor(pts / VOXEL_SIZE).astype(np.int64) + self.OFF
        return (k[:, 0] << 40) | (k[:, 1] << 20) | k[:, 2]

    def _decode(self, keys):
        k = np.stack([(keys >> 40) & self.MASK, (keys >> 20) & self.MASK, keys & self.MASK], axis=1)
        return (k - self.OFF + 0.5) * VOXEL_SIZE

    def update(self, m, force=False):
        if not self.alive:
            return
        if not self.plt.fignum_exists(self.fig.number):
            self.alive = False                       # jendela ditutup user
            return
        now = time.time()
        if not force and now - self.last < LIVE_EVERY_S:
            self.fig.canvas.flush_events()           # jaga jendela tetap responsif
            return
        self.last = now
        if m.new_pts:
            new = np.vstack(m.new_pts)
            m.new_pts = []
            self.keys = np.unique(np.concatenate([self.keys, self._encode(new)]))
        if len(self.keys) == 0:
            self.fig.canvas.flush_events()
            return
        v = self._decode(self.keys)
        v = v[:: max(1, len(v) // LIVE_MAX_POINTS)]
        z = v[:, 2]
        vmin, vmax = float(z.min()), float(max(z.max(), z.min() + 0.1))
        self.sc3._offsets3d = (v[:, 0], v[:, 1], z)
        self.sc3.set_array(z)
        self.sc3.set_clim(vmin, vmax)
        self.sc2.set_offsets(v[:, :2])
        self.sc2.set_array(z)
        self.sc2.set_clim(vmin, vmax)
        fl = np.array(m.flown)
        pl = fl[:: max(1, len(fl) // 2000)]
        self.line3.set_data(pl[:, 0], pl[:, 1])
        self.line3.set_3d_properties(pl[:, 2])
        self.line2.set_data(pl[:, 0], pl[:, 1])
        d = fl[-1]
        self.dr3._offsets3d = ([d[0]], [d[1]], [d[2]])
        self.dr2.set_data([d[0]], [d[1]])
        need = max(float(np.abs(v[:, :2]).max()), float(np.abs(fl[:, :2]).max()))
        if need > self.half:
            self.half = float(np.ceil(need))
        self._limits(zmax=float(max(z.max(), fl[:, 2].max())) + 0.1)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        try:
            self.plt.close(self.fig)
            self.plt.ioff()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Setup misi
# ----------------------------------------------------------------------------
class Mission:
    def __init__(self):
        self.client, self.sim = connect()
        sim = self.sim
        quad = sim.getObject(QUAD_PATH)

        def find_any(alias):
            return find_by_alias(sim, quad, alias) or find_by_alias(sim, sim.handle_scene, alias)

        self.base = find_any(BASE_ALIAS)
        self.target = find_any(TARGET_ALIAS)
        scanner = find_any(SCANNER_ALIAS)
        if self.base is None or self.target is None:
            names = [sim.getObjectAlias(h) for h in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)]
            sys.exit("Objek base/target tidak ditemukan.\nObjek di scene: %s" % names)
        root = scanner if scanner is not None else quad
        self.sensors = [sensor_info(sim, h) for h in
                        sim.getObjectsInTree(root, sim.object_visionsensor_type, 0)]
        if not self.sensors:
            sys.exit("Tidak ada vision sensor di bawah scanner.")
        print("Vision sensor ditemukan: %d" % len(self.sensors))

        if sim.getSimulationState() != sim.simulation_stopped:
            sim.stopSimulation()
            while sim.getSimulationState() != sim.simulation_stopped:
                time.sleep(0.1)
        self.client.setStepping(True)
        sim.startSimulation()
        self.client.step()

        aim_sensors(sim, self.sensors, self.base, CAMERA_TILT_DEG)

        self.flown, self.scan3d, self.scan2d = [], [], []
        self.last_scan, self.n_scan = -1e9, 0
        self.new_pts = []                  # titik baru sejak update live terakhir
        self.n_points = 0                  # total titik yang pernah diterima (untuk HUD)

    # -- scanning ----------------------------------------------------------
    def do_scan(self):
        sim = self.sim
        for info in self.sensors:
            depth = read_depth(sim, info)
            pts, valid = depth_to_cam_points(depth, info)
            if self.n_scan < 3:
                M = get_matrix(sim, info["handle"])
                print("[DEBUG scan %d] '%s' res=%dx%d near=%.3f far=%.2f fov=%.1f deg"
                      % (self.n_scan, info["alias"], depth.shape[1], depth.shape[0],
                         info["near"], info["far"], np.degrees(info["angle"])))
                print("   depth min/mean/max = %.3f / %.3f / %.3f ; hit = %.1f%% ; valid = %.1f%%"
                      % (depth.min(), depth.mean(), depth.max(),
                         100 * (depth < 0.999).mean(), 100 * valid.mean()))
                print("   posisi sensor =", np.round(M[:, 3], 2), " arah pandang =", np.round(M[:, 2], 2))
                if self.n_scan == 0:
                    os.makedirs(OUT_DIR, exist_ok=True)
                    import matplotlib.pyplot as plt
                    plt.imsave(os.path.join(OUT_DIR, "debug_depth.png"), depth,
                               cmap="viridis", origin="lower")
            sel = valid[::PIXEL_STRIDE, ::PIXEL_STRIDE]
            p3 = pts[::PIXEL_STRIDE, ::PIXEL_STRIDE][sel]
            if len(p3):
                w = to_world(sim, info["handle"], p3).astype(np.float32)
                self.scan3d.append(w)
                self.new_pts.append(w)
                self.n_points += len(w)
            mid = depth.shape[0] // 2
            p2 = pts[mid][valid[mid]]
            if len(p2):
                self.scan2d.append(to_world(sim, info["handle"], p2).astype(np.float32))
        self.n_scan += 1
        if self.n_scan % 50 == 0:
            self._compress()

    def _compress(self):
        """Gabungkan & buang titik duplikat pada grid RAW_GRID agar memori terkendali."""
        if not self.scan3d:
            return
        allp = np.vstack(self.scan3d)
        keys = np.floor(allp / RAW_GRID).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        self.scan3d = [allp[np.sort(idx)]]

    def step(self, scan=True):
        self.client.step()
        t = self.sim.getSimulationTime()
        self.flown.append(get_position(self.sim, self.base))
        if scan and t - self.last_scan >= SCAN_EVERY:
            self.do_scan()
            self.last_scan = t
        return t

    def finish(self):
        self._compress()
        try:
            self.client.setStepping(False)
            self.sim.stopSimulation()
        except Exception:
            pass
        allp = np.vstack(self.scan3d) if self.scan3d else np.zeros((0, 3))
        print("Selesai. Paket scan 3D: %d, 2D: %d, total titik 3D: %d"
              % (len(self.scan3d), len(self.scan2d), len(allp)))
        if len(allp):
            print("z min/max titik = %.2f / %.2f" % (allp[:, 2].min(), allp[:, 2].max()))
        return (np.array(self.flown), np.vstack(self.scan2d) if self.scan2d else np.zeros((0, 3)), allp)


# ----------------------------------------------------------------------------
# MODE MANUAL (keyboard)
# ----------------------------------------------------------------------------
def run_manual(live_enabled=True, panel_enabled=True):
    try:
        import keyboard
    except ImportError:
        sys.exit("Library 'keyboard' belum ada. Jalankan:  pip install keyboard")

    m = Mission()
    sim = m.sim
    dt = sim.getSimulationTimeStep()
    T = get_position(sim, m.base)
    Mb = get_matrix(sim, m.base)
    yaw = float(np.arctan2(Mb[1, 0], Mb[0, 0]))
    t_hud = 0.0

    print("Kontrol aktif. Lihat jendela 'Drone Control'. Tekan ESC untuk selesai.\n")
    panel = ControlPanel() if (panel_enabled and CONTROL_PANEL) else None
    live = LiveView() if (live_enabled and LIVE_VIEW) else None

    try:
        while True:
            w0 = time.time()
            ks = {k: keyboard.is_pressed(k) for k in PANEL_KEYS}
            if ks["esc"]:
                break
            fwd = float(ks["w"]) - float(ks["s"])
            lat = float(ks["a"]) - float(ks["d"])          # a = kiri (+Y badan)
            vz = float(ks["space"]) - float(ks["ctrl"])
            yaw += (float(ks["left"]) - float(ks["right"])) * np.radians(YAW_RATE_DEG) * dt

            c, s = np.cos(yaw), np.sin(yaw)
            T = T + dt * np.array([MOVE_SPEED * (fwd * c - lat * s),
                                   MOVE_SPEED * (fwd * s + lat * c),
                                   CLIMB_SPEED * vz])
            p = get_position(sim, m.base)
            off = T - p
            n = np.linalg.norm(off)
            if n > TARGET_LEAD_MAX:                         # target jangan lari terlalu jauh
                T = p + off / n * TARGET_LEAD_MAX
            T[2] = np.clip(T[2], Z_MIN, Z_MAX)

            set_position(sim, m.target, T)
            set_orientation(sim, m.target, [0.0, 0.0, yaw])  # controller drone mengikuti yaw target
            t = m.step(scan=True)
            if live is not None:
                live.update(m)
            if panel is not None:
                panel.update(ks, dict(x=p[0], y=p[1], z=p[2], yaw=yaw, points=m.n_points, t=t))

            if t - t_hud > 1.0:
                t_hud = t
                print("\rpos = (%.2f, %.2f, %.2f)  yaw = %5.1f deg  titik = %d      "
                      % (p[0], p[1], p[2], np.degrees(yaw) % 360, sum(len(a) for a in m.scan3d)),
                      end="", flush=True)
            time.sleep(max(0.0, dt - (time.time() - w0)))   # jaga kira-kira real-time
    except KeyboardInterrupt:
        pass
    finally:
        if panel is not None:
            panel.close()
        if live is not None:
            live.close()                                     # tutup live; hasil lengkap ditampilkan setelahnya
    print()
    return m.finish()


# ----------------------------------------------------------------------------
# Post process
# ----------------------------------------------------------------------------
def voxelize(pts, voxel=VOXEL_SIZE, min_pts=MIN_PTS_PER_VOXEL):
    keys = np.floor(pts / voxel).astype(np.int64)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    keep = counts >= min_pts
    return (uniq[keep] + 0.5) * voxel, counts[keep]


# ----------------------------------------------------------------------------
# Visualisasi
# ----------------------------------------------------------------------------
def _limits(ax, *arrays):
    allp = np.vstack([a for a in arrays if len(a)])
    half = max(3.0, float(np.ceil(np.abs(allp[:, :2]).max())))
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-0.3, max(2.1, float(allp[:, 2].max()) + 0.1))


def _axes3d(title):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig, ax


def make_figures(flown, scan2d, scan3d, out_dir, show):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    # 1) Jalur drone
    fig, ax = _axes3d("Jalur Drone")
    ax.plot(*flown.T, color="blue", lw=2, label="jalur terbang")
    ax.scatter(*flown[0], c="green", s=40, label="start")
    ax.scatter(*flown[-1], c="red", s=40, label="finish")
    _limits(ax, flown)
    ax.legend()
    fig.savefig(os.path.join(out_dir, "1_jalur_drone.png"), dpi=150)

    # 2) Scan 2D (tampak atas, warna = tinggi)
    fig2, ax2 = plt.subplots(figsize=(7, 7))

    if len(scan3d):
        # Menggunakan titik yang sama dengan live scan tampak atas.
        sc2 = ax2.scatter(
            scan3d[:, 0],
            scan3d[:, 1],
            s=3,
            c=scan3d[:, 2],
            cmap=CMAP
        )
        fig2.colorbar(
            sc2, ax=ax2, shrink=0.8,
            label="Tinggi Z (m)"
        )

    ax2.plot(
        flown[:, 0],
        flown[:, 1],
        color="blue",
        lw=1.5,
        label="Jalur drone"
    )

    ax2.set_aspect("equal")
    ax2.set_title("Scan 2D — Live Top View")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig2.savefig(
        os.path.join(out_dir, "2_scan_2D.png"),
        dpi=150
    )

    # 3) Scan 3D mentah
    fig3, ax3 = _axes3d("Scan 3D (point cloud mentah)")
    sub = scan3d[:: max(1, len(scan3d) // 60000)]
    ax3.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=0.5, c="0.35")
    ax3.plot(*flown.T, color="blue", lw=2)
    _limits(ax3, scan3d, flown)
    fig3.savefig(os.path.join(out_dir, "3_scan_3D_raw.png"), dpi=150)

    # 4+5) Voxel + color mapping ketinggian = "3D Map"
    vox, _ = voxelize(scan3d)
    fig4, ax4 = _axes3d("3D Map")
    show_v = vox[:: max(1, len(vox) // 200000)]          # batasi render agar tidak lambat
    sc = ax4.scatter(show_v[:, 0], show_v[:, 1], show_v[:, 2], c=show_v[:, 2], cmap=CMAP,
                     s=2, depthshade=False)
    ax4.plot(*flown.T, color="blue", lw=2)
    _limits(ax4, vox, flown)
    fig4.colorbar(sc, ax=ax4, shrink=0.55, pad=0.08, label="Tinggi z (m)")
    fig4.savefig(os.path.join(out_dir, "4_3D_map_voxel_color.png"), dpi=150)

    print("Titik mentah: %d -> voxel (%.2f m): %d" % (len(scan3d), VOXEL_SIZE, len(vox)))
    if show:
        plt.show()


# ----------------------------------------------------------------------------
# Demo sintetis
# ----------------------------------------------------------------------------
def synthetic_data(seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    gx, gy = np.meshgrid(np.arange(-2.2, 2.2, 0.04), np.arange(-2.2, 2.2, 0.04))
    pts.append(np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    boxes = [(-1.4, -0.4, 0.2, 1.0, 0.4), (0.3, 1.1, -0.9, 0.2, 0.6),
             (-0.5, 0.5, -1.5, -0.6, 0.3), (-1.8, -1.5, 0.9, 1.2, 1.0)]
    for x0, x1, y0, y1, h in boxes:
        bx, by = np.meshgrid(np.arange(x0, x1, 0.04), np.arange(y0, y1, 0.04))
        pts.append(np.c_[bx.ravel(), by.ravel(), np.full(bx.size, h)])
        for zz in np.arange(0, h, 0.04):
            for xs in (x0, x1):
                yy = np.arange(y0, y1, 0.04)
                pts.append(np.c_[np.full(yy.size, xs), yy, np.full(yy.size, zz)])
            for ys in (y0, y1):
                xx = np.arange(x0, x1, 0.04)
                pts.append(np.c_[xx, np.full(xx.size, ys), np.full(xx.size, zz)])
    scan3d = np.vstack(pts) + rng.normal(0, 0.008, (sum(len(p) for p in pts), 3))
    th = np.linspace(0, 2 * np.pi, 300)
    flown = np.c_[2.3 * np.cos(th), 2.3 * np.sin(th), 1.5 + 0.2 * np.sin(2 * th)]
    flown += rng.normal(0, 0.02, flown.shape)
    line = scan3d[np.abs(scan3d[:, 1] - 0.5) < 0.03]
    return flown, line, scan3d


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="data sintetis, tanpa CoppeliaSim")
    ap.add_argument("--no-panel", action="store_true", help="matikan jendela keybind")
    ap.add_argument("--no-live", action="store_true", help="matikan tampilan realtime")
    ap.add_argument("--no-show", action="store_true", help="hanya simpan PNG")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    if args.demo:
        flown, scan2d, scan3d = synthetic_data()
    else:
        flown, scan2d, scan3d = run_manual(live_enabled=not args.no_live and not args.no_show,
                                          panel_enabled=not args.no_panel)
        os.makedirs(args.out, exist_ok=True)
        np.savez_compressed(os.path.join(args.out, "scan_data.npz"),
                            flown=flown, scan2d=scan2d, scan3d=scan3d)
    if len(scan3d) == 0:
        sys.exit("Tidak ada titik scan yang terkumpul. Lihat baris [DEBUG scan ...] di atas "
                 "dan file debug_depth.png.")
    make_figures(flown, scan2d, scan3d, args.out, show=not args.no_show)
    print("Hasil tersimpan di folder:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()