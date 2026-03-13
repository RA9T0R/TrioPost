import streamlit as st
import os
from PIL import Image
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.workflow import build_workflow

st.set_page_config(page_title="TrioPost Dashboard", layout="wide")

with st.sidebar:
    st.markdown("### 📃 TrioPost")
    st.caption("AI Social Commerce Assistant")

st.title("📃 TrioPost: AI Social Commerce Dashboard")
st.divider()

@st.cache_resource
def get_embedding_model():
    print("📥 กำลังโหลด Embedding Model เข้าสู่ระบบ (โหลดแค่ครั้งเดียว)...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def get_store_names():
    try:
        embeddings = get_embedding_model()
        db = Chroma(persist_directory="./database/chroma_db", embedding_function=embeddings)
        data = db.get()
        if data['metadatas']:
            return list(set([meta.get('store_name', 'ไม่ระบุ') for meta in data['metadatas']]))
        return []
    except:
        return []


col_input, col_output = st.columns([1, 1.5], gap="large")

with col_input:
    st.subheader("📥 ข้อมูลสินค้า (User Input)")

    with st.container(border=True):
        tab_upload, tab_sample = st.tabs(["📸 อัปโหลดรูปเอง", "🎁 เลือกรูปตัวอย่าง"])
        selected_image_path = None

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
            sample_choice = st.radio("รูปตัวอย่าง:",
                                     ["ไม่มี", "👕 เสื้อยืด", "⌚ นาฬิกา",
                                      "🪆 ตุ๊กตา"], horizontal=True)

            if sample_choice != "ไม่มี":
                sample_map = {
                    "👕 เสื้อยืด": "assets/test_image.jpg",
                    "⌚ นาฬิกา": "assets/sample_watch.jpg",
                    "🪆 ตุ๊กตา": "assets/sample_doll.jpg"
                }
                selected_sample = sample_map[sample_choice]

                if os.path.exists(selected_sample):
                    image = Image.open(selected_sample)
                    st.image(image, width='content')
                    selected_image_path = selected_sample
                else:
                    st.warning(f"⚠️ ไม่พบไฟล์ `{selected_sample}`")

    with st.container(border=True):
        st.markdown("**🏷️ 1.ข้อมูลสินค้า :**")
        product_name = st.text_input("ชื่อแบรนด์/รุ่นสินค้า",
                                     placeholder="เช่น นาฬิกา Rolex Air-King (ถ้าไม่ระบุ AI จะเดาจากภาพ)",
                                     label_visibility="collapsed")

    with st.container(border=True):
        st.markdown("**🏬 2.เลือกร้านค้าของคุณ (RAG Style):**")
        available_stores = get_store_names()
        if available_stores:
            selected_store = st.selectbox("เลือกแบรนด์ที่ต้องการสวมบทบาท:", available_stores,
                                          label_visibility="collapsed")
        else:
            st.info("⚠️ ยังไม่มีข้อมูลร้านค้าในระบบ")
            selected_store = "ไม่มีข้อมูล"

    with st.container(border=True):
        st.markdown("**✍️ 3.คำสั่งพิเศษ :**")
        user_prompt = st.text_area("ระบุราคา หรือจุดเด่น", value="ราคา 550,000 บาทถ้วน ห้ามพิมพ์ราคาอื่น", height=100,
                                   label_visibility="collapsed")

    with st.container(border=True):
        st.markdown("**📱 4.เลือกแพลตฟอร์มปลายทาง :**")
        selected_platform = st.radio("แพลตฟอร์ม:",
                                     ["Facebook (จัดเต็ม)", "Instagram (เน้นแฮชแท็ก)", "X / Twitter (สั้นกระชับ)"],
                                     horizontal=True, label_visibility="collapsed")

    btn_generate = st.button("🚀 เริ่มต้นสร้างคอนเทนต์", use_container_width=True, type="primary")

with col_output:
    st.subheader("✨ ผลลัพธ์จาก AI (AI Content)")

    if "generated_caption" not in st.session_state:
        st.session_state.generated_caption = ""

    if btn_generate:
        if selected_image_path is not None:
            with st.status("🤖 ทีม Agents กำลังระดมสมอง...", expanded=True) as status:
                try:
                    st.write("🏃‍♂️ กำลังดำเนินการ (LangGraph Workflow)...")
                    app = build_workflow()

                    final_result = app.invoke({
                        "image_path": selected_image_path,
                        "user_prompt": user_prompt,
                        "store_name": selected_store,
                        "platform": selected_platform,
                        "product_name": product_name
                    })

                    st.session_state.generated_caption = final_result["final_post"]

                    st.divider()
                    st.markdown("### 🕵️‍♂️ เบื้องหลังการทำงานของทีม Agents")

                    st.markdown("**👁️ 1. Vision Agent (สิ่งที่ AI มองเห็น):**")
                    st.info(final_result.get('vision_detail', 'ไม่มีข้อมูล'))

                    st.markdown("**🔍 2. Researcher Agent (ข้อมูลจากอินเทอร์เน็ต):**")
                    with st.container(height=150, border=True):
                        st.markdown(final_result.get('research_data', 'ไม่มีข้อมูล'))

                    st.markdown(f"**🧠 3. RAG Node (คู่มือแบรนด์: {selected_store}):**")
                    st.success(final_result.get('rag_context', 'ไม่มีข้อมูล'))

                    status.update(label="✨ สร้างแคปชั่นเสร็จสมบูรณ์! (คลิกเพื่อดูเบื้องหลังการคิดของ AI)",
                                  state="complete", expanded=False)

                except Exception as e:
                    status.update(label="❌ เกิดข้อผิดพลาด", state="error", expanded=True)
                    st.error(f"รายละเอียด: {e}")
        else:
            st.warning("⚠️ กรุณาอัปโหลดรูปภาพ หรือ เลือกรูปตัวอย่างก่อนครับ!")
