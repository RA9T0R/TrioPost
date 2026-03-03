import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_vector_db():
    print("🚀 กำลังเริ่มต้นสร้าง Vector Database สำหรับ TrioPost...")

    # ---------------------------------------------------------
    # 1. เตรียมข้อมูล Brand Guidelines (เอกลักษณ์ของร้าน 2 สไตล์)
    # ---------------------------------------------------------
    docs = [
        Document(
            page_content=(
                "ชื่อร้าน: GadgetZeed (ขายอุปกรณ์ไอทีและของเล่นวัยรุ่น) \n"
                "คีย์เวิร์ดคำสั่ง: ขอแบบวัยรุ่น, เป็นกันเอง, สนุกสนาน, ไม่เป็นทางการ, วัยรุ่นกระแส \n"
                "สไตล์การเขียน: สนุกสนาน เป็นกันเอง ตื่นเต้น พลังงานล้นเหลือ \n"
                "กฎของร้าน: \n"
                "- ใช้สรรพนามเรียกตัวเองว่า 'แอด' และเรียกลูกค้าว่า 'พวกแก' หรือ 'วัยรุ่น' \n"
                "- ต้องมีคำสแลงฮิตๆ เช่น 'ของมันต้องมี', 'อย่างตึง', 'สุดปัง' \n"
                "- ห้ามพูดจาทางการเด็ดขาด ห้ามใช้คำว่า 'เรียนคุณลูกค้า' \n"
                "- ปิดท้ายโพสต์ด้วยอิโมจิ 🔥😎 เสมอ \n"
                "- โปรโมชั่น: ส่งฟรีทั่วประเทศเมื่อโอนเต็มจำนวน"
            ),
            metadata={"store_name": "GadgetZeed", "style": "informal"}
        ),
        Document(
            page_content=(
                "ชื่อร้าน: LuxeAura (ขายเครื่องประดับ นาฬิกา และสินค้าพรีเมียม) \n"
                "คีย์เวิร์ดคำสั่ง: ขอแบบทางการหน่อย, ทางการ, สุภาพ, หรูหรา, เรียบร้อย, ผู้ใหญ่ \n"
                "สไตล์การเขียน: ทางการ สุภาพ หรูหรา อ่อนน้อม และน่าเชื่อถือ \n"
                "กฎของร้าน: \n"
                "- ใช้สรรพนามเรียกตัวเองว่า 'ทางแบรนด์' และเรียกลูกค้าว่า 'คุณลูกค้า' \n"
                "- เน้นบรรยายถึงความประณีต มูลค่า และความคลาสสิกของสินค้า \n"
                "- ห้ามใช้คำสแลงหรือภาษาวัยรุ่นโดยเด็ดขาด \n"
                "- ปิดท้ายโพสต์ด้วยประโยค 'LuxeAura ยินดีให้บริการอย่างยิ่งค่ะ' ✨ \n"
                "- โปรโมชั่น: บริการจัดส่งด้วยแมสเซนเจอร์ส่วนตัวฟรีในเขตกรุงเทพฯ"
            ),
            metadata={"store_name": "LuxeAura", "style": "formal"}
        )
    ]

    # ---------------------------------------------------------
    # 2. โหลด Embedding Model (ตัวแปลงข้อความเป็นตัวเลข)
    # ---------------------------------------------------------
    print("🧠 กำลังโหลด Embedding Model (อาจใช้เวลาสักครู่ในการโหลดครั้งแรก)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ---------------------------------------------------------
    # 3. สร้างและบันทึก Vector Database (ChromaDB)
    # ---------------------------------------------------------
    db_path = "./chroma_db"

    print(f"💾 กำลังบันทึกข้อมูลลง Vector DB ที่โฟลเดอร์: {db_path} ...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_path
    )

    print("✅ สร้าง Vector Database เสร็จสมบูรณ์แล้ว!")

    print("\n🔍 ทดสอบการดึงข้อมูล (Retrieval Test):")
    query = "ร้านที่ขายเครื่องประดับ สไตล์ทางการ"
    results = vectorstore.similarity_search(query, k=1)

    print(f"คำค้นหา: '{query}'")
    print(f"ร้านที่ค้นพบ: {results[0].metadata['store_name']}")
    print(f"ข้อมูลที่ดึงมาได้: \n{results[0].page_content}")

if __name__ == "__main__":
    build_vector_db()