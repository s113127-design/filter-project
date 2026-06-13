import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2

st.title("像個偉(偽)人一樣 📸 (基礎相機版)")
st.write("👉 先測試鏡頭與拍照功能是否正常！")

# 初始化拍照歷史紀錄
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

# --- WebRTC 鏡頭畫面 ---
ctx = webrtc_streamer(
    key="pure-camera-no-config",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True, 
        "audio": False
    }
)

# 📸 拍照按鈕
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
        
        # 把照片塞進歷史紀錄的第一筆
        st.session_state.history.insert(0, (orig_rgb, filter_rgb))
        
        # 💡 【要求 1】：如果紀錄超過 8 張，就把最後面（最舊的）那一張刪掉
        if len(st.session_state.history) > 8:
            st.session_state.history.pop()
            
        st.success("拍照成功！已加到下方紀錄中。")
    else:
        st.warning("請先點擊 START 開啟鏡頭再拍照喔！")

st.markdown("---")

# 🖼️ 顯示剛拍好的大照片
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

    st.markdown("---")

    # 📜 【要求 2】：歷史紀錄一列排 4 張（大小剛好是原本主圖的四分之一）
    st.subheader("📜 歷史拍照紀錄 (最多儲存 8 張)")
    
    num_hist = len(st.session_state.history)
    
    # 建立一個寬度為 4 格的橫列排版
    cols = st.columns(4)
    
    # 依序把歷史紀錄畫出來
    for idx, (orig, filt) in enumerate(st.session_state.history):
        # 利用餘數（% 4）決定照片要放在第幾行
        col_idx = idx % 4
        with cols[col_idx]:
            st.image(
                filt, 
                caption=f"紀錄 #{num_hist-idx}", 
                use_container_width=True
            )
