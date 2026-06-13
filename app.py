import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2

st.title("像個偉(偽)人一樣 📸 (終極測試版)")
st.write("👉 移除了所有複雜設定，測試鏡頭是否能正常開啟！")

if "history" not in st.session_state:
    st.session_state.history = []

class VideoProcessor:
    def __init__(self):
        self.latest_orig = None
        self.latest_filter = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_orig = img.copy()
        self.latest_filter = img.copy()
        return frame.from_ndarray(img, format="bgr24")

# --- 💡 這裡把所有的 rtc_configuration 徹底刪除了 ---
ctx = webrtc_streamer(
    key="pure-camera-no-config",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True, 
        "audio": False
    }
)

if st.button("📸 Capture (拍照)", use_container_width=True):
    if ctx.video_processor and ctx.video_processor.latest_orig is not None:
        orig_rgb = cv2.cvtColor(
            ctx.video_processor.latest_orig, 
            cv2.COLOR_BGR2RGB
        )
        filter_rgb = cv2.cvtColor(
            ctx.video_processor.latest_filter, 
            cv2.COLOR_BGR2RGB
        )
        st.session_state.history.insert(0, (orig_rgb, filter_rgb))
        st.success("拍照成功！已加到下方紀錄中。")
    else:
        st.warning("請先點擊 START 開啟鏡頭再拍照喔！")

st.markdown("---")

if st.session_state.history:
    st.subheader("🖼️ 剛剛拍到的影像")
    current_orig, current_filter = st.session_state.history[0]
    
    col_orig, col_filt = st.columns(2)
    with col_orig:
        st.image(
            current_orig, 
            caption="拍到的原影像", 
            use_container_width=True
        )
    with col_filt:
        st.image(
            current_filter, 
            caption="濾鏡影像", 
            use_container_width=True
        )
