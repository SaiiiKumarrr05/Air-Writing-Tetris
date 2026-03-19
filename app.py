"""
Air Writing Tetris — Streamlit web interface.

Run:
    streamlit run app.py
"""
from __future__ import annotations
import threading, time
import numpy as np
import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from auto_train import ensure_model
ensure_model()

from game.engine import TetrisEngine
from vision.hand_tracker import HandTracker
from vision.drawing_buffer import DrawingBuffer
from vision.gesture_recognizer import GestureRecognizer
from models.classifier import TetrominoClassifier, PIECE_NAMES, CONF_THRESHOLD
from utils.renderer import make_frame, W_TOTAL, H_TOTAL, FONT

RTC_CFG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
DRAWING_ZONE_FRAC = 0.65
ASK_LABEL_CONF    = 0.45

# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Air Writing Tetris",
    layout="wide",
    page_icon="🎮",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d0d1a; }
  [data-testid="stSidebar"]          { background: #12122a; }
  h1, h2, h3 { color: #00f0f0; }
  .metric-card {
      background: #1a1a2e; border-radius: 10px;
      padding: 14px; text-align: center;
      margin-bottom: 8px;
  }
  .badge {
      display:inline-block; padding:4px 14px;
      border-radius:14px; font-weight:bold; font-size:0.85em;
  }
</style>
""", unsafe_allow_html=True)

st.title("🎮 Air Writing Tetris")
st.caption("Draw tetromino shapes in the air — your webcam does the rest.")


# -----------------------------------------------------------------------
class GameProcessor(VideoProcessorBase):
    """
    WebRTC video processor: runs vision + game engine per frame.
    All engine state lives here so it persists across frames.
    """
    def __init__(self):
        self.tracker    = HandTracker()
        self.buf        = DrawingBuffer()
        self.gesture    = GestureRecognizer()
        self.clf        = TetrominoClassifier()
        self.engine     = TetrisEngine()
        self._lk        = threading.Lock()

        self.state         : str             = "drawing"
        self.top_pred      : list            = []
        self.drawing_img   : np.ndarray|None = None
        self.pending       : str|None        = None
        self.pending_conf  : float           = 0.0
        self.ask_label     : bool            = False

        # Signals from UI buttons (written by Streamlit thread, read by video thread)
        self.sig_confirm   = False
        self.sig_reset     = False
        self.sig_spawn     : str|None = None   # piece name for manual spawn

    # ------------------------------------------------------------------
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        with self._lk:
            self._handle_signals()
            self._process_vision(img)
            self._tick_engine()

        game_f = make_frame(
            self.engine,
            drawing_img=self.drawing_img,
            top_pred=self.top_pred or None,
            state="game_over" if self.engine.game_over else self.state,
        )
        # Combine game frame + webcam side-by-side
        gh        = game_f.shape[0]
        ch, cw    = img.shape[:2]
        cam_r     = cv2.resize(img, (int(cw * gh / ch), gh))
        combined  = np.concatenate([game_f, cam_r], axis=1)
        return av.VideoFrame.from_ndarray(combined, format="bgr24")

    # ------------------------------------------------------------------
    def _handle_signals(self):
        if self.sig_reset:
            self.engine.reset()
            self.state       = "drawing"
            self.ask_label   = False
            self.top_pred    = []
            self.drawing_img = None
            self.buf.clear()
            self.sig_reset   = False

        if self.sig_spawn is not None:
            self._do_spawn(self.sig_spawn)
            self.sig_spawn = None

        if self.sig_confirm and self.state == "drawing":
            self._confirm()
            self.sig_confirm = False

    def _process_vision(self, img: np.ndarray):
        h, w     = img.shape[:2]
        dz_w     = int(w * DRAWING_ZONE_FRAC)
        dz_x     = (w - dz_w) // 2
        dz_color = (0, 210, 200) if self.state == "drawing" else (50, 50, 80)
        cv2.rectangle(img, (dz_x, 0), (dz_x + dz_w, h), dz_color, 2)

        results, small_shape = self.tracker.process(img)

        if not results or not results.multi_hand_landmarks:
            if self.buf.is_drawing:
                self.buf.end_stroke()
            return

        lm       = results.multi_hand_landmarks[0]
        self.tracker.draw(img, lm)
        idx_tip  = self.tracker.index_tip(lm, img.shape)
        wrist    = self.tracker.wrist(lm, img.shape)
        pinching = self.tracker.is_pinching(lm, img.shape)

        if self.state == "drawing":
            if pinching:
                if not self.buf.is_drawing:
                    self.buf.start_stroke()
                ix, iy = idx_tip
                if dz_x <= ix <= dz_x + dz_w:
                    self.buf.add_point((float(ix - dz_x), float(iy)))
                cv2.circle(img, idx_tip, 9, (0, 255, 100), -1)
            else:
                if self.buf.is_drawing:
                    self.buf.end_stroke()
            if self.buf.has_enough(15):
                raw = self.buf.to_image()
                self.drawing_img = raw
                self.top_pred    = self.clf.predict_topk(raw, k=3)

        elif self.state == "playing":
            self.gesture.update(wrist, idx_tip)
            g = self.gesture.detect()
            if g == "move_left":   self.engine.move(0, -1)
            elif g == "move_right": self.engine.move(0,  1)
            elif g == "fast_drop":  self.engine.hard_drop()
            if self.gesture.detect_rotate():
                self.engine.rotate()

        # Overlays
        if self.state == "drawing":
            cv2.putText(img, "Pinch & draw", (dz_x+8, 28),
                        FONT, 0.58, (0, 255, 200), 2, cv2.LINE_AA)
            self.buf.draw_on_frame(img, offset_x=dz_x)

    def _tick_engine(self):
        if self.state == "playing" and not self.engine.game_over:
            self.engine.update()
            if self.engine.current is None:
                self.state       = "drawing"
                self.buf.clear()
                self.top_pred    = []
                self.drawing_img = None

    def _confirm(self):
        if not self.buf.has_enough(15):
            return
        raw = self.buf.to_image()
        top = self.clf.predict_topk(raw, k=3)
        if not top:
            return
        self.drawing_img  = raw
        self.top_pred     = top
        name, conf        = top[0]
        self.pending      = name
        self.pending_conf = conf
        if conf < ASK_LABEL_CONF:
            self.ask_label = True
        else:
            self._do_spawn(name)

    def _do_spawn(self, name: str):
        self.engine.spawn(name)
        self.state       = "playing"
        self.ask_label   = False
        self.buf.clear()
        self.drawing_img = None
        self.top_pred    = []


# -----------------------------------------------------------------------
# Page layout
# -----------------------------------------------------------------------
col_vid, col_ctrl = st.columns([4, 1])

with col_vid:
    ctx = webrtc_streamer(
        key="air-tetris",
        video_processor_factory=GameProcessor,
        rtc_configuration=RTC_CFG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_ctrl:
    st.subheader("Controls")

    proc: GameProcessor | None = ctx.video_processor if ctx else None

    if proc:
        if st.button("✅ Confirm Drawing", use_container_width=True):
            with proc._lk:
                proc.sig_confirm = True

        if st.button("🔄 Reset Game", use_container_width=True):
            with proc._lk:
                proc.sig_reset = True

        st.divider()

        # Manual piece spawn (when ask_label is True)
        with proc._lk:
            ask = proc.ask_label
            pend = proc.pending
            pconf = proc.pending_conf

        if ask:
            st.warning(f"Low confidence ({pconf*100:.0f}%) — pick the correct piece:")
            for name in PIECE_NAMES:
                if st.button(f"Spawn  {name}", key=f"spawn_{name}", use_container_width=True):
                    with proc._lk:
                        proc.sig_spawn = name

        st.divider()
        st.subheader("📊 Live Stats")
        stats_ph = st.empty()

        # Live stats polling loop
        while ctx.state.playing and proc:
            with proc._lk:
                sc  = proc.engine.score
                lv  = proc.engine.level
                ln  = proc.engine.lines
                el  = proc.engine.elapsed
                st_ = "game_over" if proc.engine.game_over else proc.state
                tp  = list(proc.top_pred)

            with stats_ph.container():
                c1, c2 = st.columns(2)
                c1.metric("Score",  sc)
                c2.metric("Level",  lv)
                c1.metric("Lines",  ln)
                c2.metric("Time",   f"{el}s")
                st.markdown(
                    f'<span class="badge" style="background:'
                    f'{"#006600" if st_=="playing" else "#003366" if st_=="drawing" else "#660000"}">'
                    f'{st_.upper()}</span>',
                    unsafe_allow_html=True,
                )
                if tp:
                    name, conf = tp[0]
                    st.progress(float(conf), text=f"Detected: **{name}**  {conf*100:.0f}%")
            time.sleep(0.12)

    else:
        # Instructions before camera starts
        st.info("▶ Click **START** above to enable your webcam.")
        st.markdown("""
**Gesture Guide**

| Gesture | Action |
|---|---|
| 👌 Pinch + move | Draw shape |
| 👋 Release pinch | End stroke |
| ✅ SPACE | Confirm piece |
| 👈 Swipe left | Move left |
| 👉 Swipe right | Move right |
| 👇 Swipe down | Hard drop |
| 🌀 Circle | Rotate |

**Shape Reference**
```
I: ────────
O: ██
T: ─┬─  S: ─┐   Z: ┌─
     └      └─     ─┘
J: ┌──  L: ──┐
   └         └
```
        """)

# Footer
st.markdown("---")
st.caption("Air Writing Tetris · MediaPipe + PyTorch CNN + OpenCV · CUDA-accelerated")
