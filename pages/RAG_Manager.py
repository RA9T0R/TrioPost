import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

st.set_page_config(page_title="RAG Management", layout="wide")

st.title("📂 จัดการฐานข้อมูลร้านค้า (RAG Management)")
st.divider()

@st.cache_resource
def get_embedding_model():
    print("📥 [RAG Manager] กำลังโหลด Embedding Model (โหลดแค่ครั้งเดียว)...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def load_db():
    embeddings = get_embedding_model()
    return Chroma(persist_directory="./database/chroma_db", embedding_function=embeddings)

db = load_db()

def delete_rag_data(doc_id):
    db.delete(ids=[doc_id])
    st.session_state['deleted'] = True

if 'deleted' in st.session_state and st.session_state['deleted']:
    st.success("🗑️ ลบข้อมูลร้านค้าออกจากความจำเรียบร้อยแล้ว!")
    st.session_state['deleted'] = False

if 'added' in st.session_state and st.session_state['added']:
    st.success("✅ เพิ่มข้อมูลร้านค้าใหม่เรียบร้อยแล้ว!")
    st.balloons()
    st.session_state['added'] = False

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("🏪 ร้านค้าที่มีอยู่ใน RAG ปัจจุบัน")

    all_data = db.get()

    if all_data['documents']:
        for i in range(len(all_data['documents'])):
            doc_id = all_data['ids'][i]
            store_name = all_data['metadatas'][i].get('store_name', 'ไม่ระบุชื่อร้าน')

            with st.expander(f"🔹 ร้าน: {store_name}", expanded=False):
                st.text(all_data['documents'][i])
                st.write(f"**Document ID:** `{doc_id}`")

                st.button(
                    "🗑️ ลบร้านค้านี้",
                    key=f"del_btn_{doc_id}",
                    on_click=delete_rag_data,
                    args=(doc_id,)
                )
    else:
        st.info("💡 ยังไม่มีข้อมูลร้านค้าในระบบ กรุณาเพิ่มที่หน้าต่างด้านขวา")

with col_right:
    st.subheader("➕ เพิ่มร้านค้าใหม่เข้าสู่ RAG")

    with st.form("add_store_form", clear_on_submit=True):
        new_store_name = st.text_input("ชื่อร้านค้า")
        new_style = st.selectbox("สไตล์หลัก", ["ทางการ (Formal)", "เป็นกันเอง (Informal)", "น่ารักสายหวาน (Cute)"])

        new_details = st.text_area("📝 กฎการเขียนและรายละเอียดร้าน", height=120,
                                   placeholder="- ใช้สรรพนามเรียกตัวเองว่า...\n- ห้ามใช้คำว่า...\n- โปรโมชั่นปัจจุบันคือ...")

        new_example = st.text_area("📌 ตัวอย่างโพสต์เก่าของร้าน (สำคัญมาก! ช่วยให้ AI จำสไตล์ได้เป๊ะขึ้น)", height=150,
                                   placeholder="ก๊อปปี้แคปชั่นเก่าๆ ของร้านมาวางที่นี่สัก 1 โพสต์ เพื่อให้ AI ดูเป็นแบบอย่าง...")

        submitted = st.form_submit_button("💾 บันทึกลงความจำ (Update DB)", use_container_width=True)

        if submitted:
            if new_store_name and new_details:

                content_builder = f"ชื่อร้าน: {new_store_name}\nสไตล์: {new_style}\nกฎของร้าน:\n{new_details}"

                if new_example:
                    content_builder += f"\n\n📌 ตัวอย่างโพสต์ (สำหรับดูสไตล์การเขียน ห้ามนำสินค้าในนี้ไปโฆษณา):\n{new_example}"

                new_doc = Document(
                    page_content=content_builder,
                    metadata={"store_name": new_store_name, "style": new_style}
                )

                db.add_documents([new_doc])

                st.session_state['added'] = True
                st.rerun()
            else:
                st.error("⚠️ กรุณากรอก 'ชื่อร้านค้า' และ 'กฎการเขียน' ให้ครบถ้วน")