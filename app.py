import streamlit as st
import os
from PIL import Image
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.workflow import build_workflow

st.set_page_config(page_title="TrioPost Dashboard", layout="wide")

st.title("🤖 TrioPost: AI Social Commerce Dashboard")
st.divider()

if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = "ขอแบบทางการหน่อย และขายราคา 990 บาทเท่านั้นนะ ห้ามตั้งราคาอื่น"

def set_prompt(text):
    st.session_state.current_prompt = text

col_input, col_output = st.columns([1, 1.5], gap="large")

with col_input:
    st.subheader("📥 ข้อมูลสินค้า (User Input)")

    tab_upload, tab_sample = st.tabs(["📸 อัปโหลดรูปเอง", "🎁 เลือกรูปตัวอย่าง"])
    selected_image_path = None  # ตัวแปรเก็บพาทรูปที่จะส่งให้ AI

    with tab_upload:
        uploaded_file = st.file_uploader("อัปโหลดรูปภาพสินค้า (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, width='content')
            temp_path = "assets/temp_image.jpg"
            os.makedirs("assets", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_image_path = temp_path

    with tab_sample:
        st.markdown("เลือกรูปตัวอย่างเพื่อทดสอบระบบอย่างรวดเร็ว:")
        sample_choice = st.radio("รูปตัวอย่าง:", ["ไม่มี", "👕 เสื้อยืด (ทดสอบราคาตลาด)", "⌚ นาฬิกา (ทดสอบความหรูหรา)",
                                                  "🪆 ตุ๊กตา (ทดสอบความน่ารัก)"], horizontal=True)

        if sample_choice != "ไม่มี":
            sample_map = {
                "👕 เสื้อยืด (ทดสอบราคาตลาด)": "assets/test_image.jpg",  # รูปเดิมที่เราใช้เทสต์
                "⌚ นาฬิกา (ทดสอบความหรูหรา)": "assets/sample_watch.jpg",
                "🪆 ตุ๊กตา (ทดสอบความน่ารัก)": "assets/sample_doll.jpg"
            }
            selected_sample = sample_map[sample_choice]

            if os.path.exists(selected_sample):
                image = Image.open(selected_sample)
                st.image(image, width='content')
                selected_image_path = selected_sample
            else:
                st.warning(f"⚠️ ไม่พบไฟล์ `{selected_sample}` (อย่าลืมเอารูปไปใส่ในโฟลเดอร์ assets ก่อนทดสอบนะครับ!)")

    st.divider()
    st.markdown("**🏬 1. เลือกร้านค้าของคุณ (RAG Style):**")


    @st.cache_resource
    def get_embedding_model():
        print("📥 กำลังโหลด Embedding Model เข้าสู่ระบบ (โหลดแค่ครั้งเดียว)...")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


    def get_store_names():
        try:
            # เรียกใช้โมเดลที่โหลดเตรียมไว้แล้ว (ใช้เวลา 0.001 วินาที)
            embeddings = get_embedding_model()
            db = Chroma(persist_directory="./database/chroma_db", embedding_function=embeddings)
            data = db.get()
            if data['metadatas']:
                return list(set([meta.get('store_name', 'ไม่ระบุ') for meta in data['metadatas']]))
            return []
        except:
            return []


    available_stores = get_store_names()

    if available_stores:
        selected_store = st.selectbox("เลือกแบรนด์ที่ต้องการสวมบทบาท:", available_stores)
    else:
        st.info("⚠️ ยังไม่มีข้อมูลร้านค้าในระบบ")
        selected_store = "ไม่มีข้อมูล"

    st.markdown("**✍️ 2. คำสั่งพิเศษ (Highest Priority):**")
    user_prompt = st.text_area(
        "ระบุราคา โปรโมชั่น หรือจุดเด่นที่ต้องการบังคับให้ AI เขียน",
        value="ราคา 550,000 บาทถ้วน ห้ามพิมพ์ราคาอื่น",
        height=100
    )

    btn_generate = st.button("🚀 เริ่มต้นสร้างคอนเทนต์", use_container_width=True, type="primary")

with col_output:
    st.subheader("✨ ผลลัพธ์จาก AI (AI Content)")

    if btn_generate:
        if selected_image_path is not None:
            with st.spinner("🤖 Agents กำลังทำงานร่วมกัน..."):
                try:
                    app = build_workflow()

                    final_result = app.invoke({
                        "image_path": selected_image_path,
                        "user_prompt": user_prompt,
                        "store_name": selected_store
                    })

                    st.success("✨ สร้างแคปชั่นสำเร็จ!")
                    st.text_area("📋 แคปชั่นที่ได้ (คัดลอกไปใช้ได้เลย):", value=final_result["final_post"], height=500)

                    with st.expander("🔍 ดูขั้นตอนการวิเคราะห์ของ Agents"):
                        st.info(f"👁️ Vision Agent: {final_result.get('vision_detail')}")
                        st.warning(f"🔍 Research Data: {final_result.get('market_price')}")
                        st.success(f"🧠 RAG Context: {final_result.get('rag_context')}")

                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาด: {e}")
        else:
            st.warning("⚠️ กรุณาอัปโหลดรูปภาพ หรือ เลือกรูปตัวอย่างก่อนครับ!")
    else:
        st.info("👈 กรอกข้อมูลทางด้านซ้ายแล้วกดปุ่มเพื่อดูผลลัพธ์")