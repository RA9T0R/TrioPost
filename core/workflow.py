import os
import base64
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from core.state import TrioPostState

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ==========================================
# 1. Agents
# ==========================================
def vision_node(state: TrioPostState):
    image_path = state.get("image_path")
    print(f"👁️ [Vision Agent] กำลังวิเคราะห์รูปภาพจาก: {image_path}")

    try:
        base64_image = encode_image(image_path)

        vision_llm = ChatOpenAI(
            api_key=os.getenv("TYPHOON_API_KEY"),
            base_url="https://api.opentyphoon.ai/v1",
            model="typhoon-ocr",
            max_tokens=500
        )

        message = HumanMessage(
            content=[
                {"type": "text",
                 "text": "ภาพนี้คือภาพของสินค้าอะไร? โปรดอธิบายรูปลักษณ์ สี สภาพ และจุดเด่นของสินค้าในภาพอย่างละเอียด เพื่อนำข้อมูลไปให้ Copywriter เขียนโฆษณาขายของ"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        response = vision_llm.invoke([message])
        vision_detail = response.content

        print("✅ วิเคราะห์ภาพสำเร็จ! สกัดจุดเด่นได้เรียบร้อย")
        print(f"[สิ่งที่ AI เห็น]: {vision_detail[:100]}...")  # ปริ้นท์ให้ดูแค่ 100 ตัวอักษรแรก

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: {e}")
        vision_detail = "สินค้าแฟชั่น (ข้อมูลภาพขัดข้อง)"

    return {"vision_detail": vision_detail}


def researcher_node(state: TrioPostState):
    item_to_search = state.get("vision_detail", "สินค้าทั่วไป")

    print("🔍 [Researcher Agent] กำลังสกัดคีย์เวิร์ดเพื่อไปค้นหา...")

    llm_for_search = ChatOpenAI(
        api_key=os.getenv("TYPHOON_API_KEY"),
        base_url="https://api.opentyphoon.ai/v1",
        model="typhoon-v2.5-30b-a3b-instruct",
        temperature=0.2
    )

    keyword_prompt = f"จากรายละเอียดสินค้านี้: '{item_to_search}' จงสกัดชื่อสินค้าหลักเพื่อนำไปค้นหาราคาในเว็บ E-commerce ให้ตอบกลับมาเป็นคำสั้นๆ ไม่เกิน 3-5 คำ (เช่น 'เสื้อยืดลายพราง' หรือ 'กระเป๋าหนังสีดำ') ห้ามมีน้ำเด็ดขาด"
    short_keyword = llm_for_search.invoke(keyword_prompt).content.strip()

    search_query = f"ราคา {short_keyword} shopee lazada"
    print(f"🎯 คีย์เวิร์ดที่สกัดได้คือ: '{short_keyword}' -> นำไปค้นหา: '{search_query}'")
    try:
        tavily_tool = TavilySearchResults(max_results=3)
        search_results = tavily_tool.invoke(search_query)

        market_data = ""
        for i, res in enumerate(search_results):
            content_snippet = res['content'][:300]
            market_data += f"[เว็บที่ {i + 1}]: {content_snippet}...\n"

        if not market_data:
            market_data = "ไม่พบข้อมูลราคาที่ชัดเจนในอินเทอร์เน็ต"

        print("✅ ค้นหาราคาตลาดสำเร็จ! (และจำกัดขนาดข้อความแล้ว)")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูล Tavily: {e}")
        market_data = "ไม่สามารถเข้าถึงข้อมูลอินเทอร์เน็ตได้ในขณะนี้"
    print(market_data)
    return {"market_price": market_data}

def rag_node(state: TrioPostState):
    print(f"🧠 [RAG Node] กำลังดึงข้อมูลสไตล์ร้านค้าตามคำสั่ง: '{state['user_prompt']}'")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    db_path = "../database/chroma_db"
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

    results = vectorstore.similarity_search(state["user_prompt"], k=1)

    if results:
        retrieved_style = results[0].page_content
        store_name = results[0].metadata.get("store_name", "ไม่ระบุชื่อร้าน")
        print(f"✅ ดึงข้อมูลสำเร็จ! พบสไตล์ของร้าน: {store_name}")
    else:
        retrieved_style = "ใช้สไตล์การเขียนขายของออนไลน์แบบมาตรฐาน เป็นกันเองและสุภาพ"
        print("⚠️ ไม่พบข้อมูลที่ตรงกัน ใช้สไตล์มาตรฐานแทน")

    return {"rag_context": retrieved_style}


def copywriter_node(state: TrioPostState):
    print(f"✍️ [Copywriter Agent] กำลังแต่งโพสต์ขายของด้วย Typhoon LLM...")

    detail = state.get("vision_detail", "ไม่มีข้อมูลสินค้า")
    price = state.get("market_price", "ไม่มีข้อมูลราคา")
    style = state.get("rag_context", "เขียนสไตล์มาตรฐาน")
    user_prompt = state.get("user_prompt", "")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "คุณคือสุดยอดนักเขียน Copywriter โฆษณาสินค้าออนไลน์มืออาชีพของประเทศไทย\n"
                   "หน้าที่ของคุณคือเขียนแคปชั่นขายของให้น่าสนใจที่สุด\n"
                   "กฎเหล็ก (Guardrails):\n"
                   "1. จงเขียนตาม 'สไตล์และกฎของร้าน' ที่กำหนดให้อย่างเคร่งครัด\n"
                   "2. ห้ามแต่งเติมคุณสมบัติสินค้า หรือตั้งราคาเอาเองเด็ดขาด ให้ใช้จากข้อมูลที่ได้รับเท่านั้น\n"
                   "3. ใช้ภาษาไทยที่สละสลวย เป็นธรรมชาติ"
                   "🚨 กฎสูงสุด (Highest Priority): หาก [คำสั่งเพิ่มเติมจากลูกค้า] มีการระบุ 'ราคา', 'โปรโมชั่น' หรือ 'เนื้อหาเฉพาะเจาะจง' ใดๆ มาแล้ว ให้คุณยึดถือข้อมูลของลูกค้าเป็นหลักทันที! และห้ามใช้ข้อมูลราคาตลาดที่ขัดแย้งกันเด็ดขาด"),
        ("user", "ข้อมูลสำหรับเขียนโพสต์มีดังนี้:\n"
                 "📌 รายละเอียดสินค้า: {detail}\n"
                 "💰 ข้อมูลราคาตลาด: {price}\n"
                 "🧠 สไตล์และกฎของร้าน: {style}\n"
                 "🗣️ คำสั่งเพิ่มเติมจากลูกค้า: {user_prompt}\n\n"
                 "ช่วยเขียนแคปชั่นขายของ พร้อมใส่ Hashtag ที่เหมาะสมให้หน่อย พร้อมโพสต์เลย!")
    ])

    llm = ChatOpenAI(
        api_key=os.getenv("TYPHOON_API_KEY"),
        base_url="https://api.opentyphoon.ai/v1",
        model="typhoon-v2.5-30b-a3b-instruct",
        temperature=0.7,
        max_tokens = 2048
    )

    chain = prompt_template | llm

    response = chain.invoke({
        "detail": detail,
        "price": price,
        "style": style,
        "user_prompt": user_prompt
    })

    print("✅ แต่งแคปชั่นเสร็จสมบูรณ์!")

    return {"final_post": response.content}


# ==========================================
# 2. Build Graph
# ==========================================

def build_workflow():
    builder = StateGraph(TrioPostState)

    builder.add_node("vision", vision_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("rag", rag_node)
    builder.add_node("copywriter", copywriter_node)

    builder.add_edge(START, "vision")
    builder.add_edge("vision", "researcher")
    builder.add_edge("researcher", "rag")
    builder.add_edge("rag", "copywriter")
    builder.add_edge("copywriter", END)

    return builder.compile()

# ==========================================
# 3. ทดสอบการทำงาน
# ==========================================
if __name__ == "__main__":
    print("🚀 เริ่มต้นเดินสายพาน TrioPost Workflow (Mock Mode)...\n")

    app = build_workflow()

    initial_state = {
        "image_path": "../assets/test_image.jpg",
        "user_prompt": "ขอแบบทางการหน่อย และขายราคา 990 บาทเท่านั้นนะ ห้ามตั้งราคาอื่น"
    }

    final_result = app.invoke(initial_state)

    print("\n" + "=" * 40)
    print("🎉 ผลลัพธ์สุดท้ายที่ได้จากสายพาน (Final Post):")
    print("=" * 40)
    print(final_result["final_post"])