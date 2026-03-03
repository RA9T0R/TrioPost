import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

def test_api_connections():
    print("🔍 กำลังทดสอบการเชื่อมต่อ API...\n")
    print("--- 1. ทดสอบ Tavily Search API ---")
    try:
        tavily_tool = TavilySearchResults(max_results=2)
        search_result = tavily_tool.invoke("ราคาเสื้อยืดสีขาวมินิมอล 2024")

        print("✅ เชื่อมต่อ Tavily สำเร็จ! เจอข้อมูลดังนี้:")
        for res in search_result:
            print(f"- {res['url']}")
    except Exception as e:
        print(f"❌ Tavily เชื่อมต่อไม่สำเร็จ กรุณาเช็ค API Key: {e}")

    print("\n" + "=" * 50 + "\n")


    print("--- 2. ทดสอบ Typhoon LLM API ---")
    try:
        llm = ChatOpenAI(
            api_key=os.getenv("TYPHOON_API_KEY"),
            base_url="https://api.opentyphoon.ai/v1",
            model="typhoon-v2.5-30b-a3b-instruct",  # หรือรุ่นอื่นๆ ที่น้องมีสิทธิ์ใช้งาน
            temperature=0.7
        )

        response = llm.invoke("สวัสดีครับ ขอทราบชื่อและหน้าที่ของคุณสั้นๆ 1 ประโยคครับ")
        print("✅ เชื่อมต่อ Typhoon สำเร็จ! AI ตอบกลับมาว่า:")
        print(f"💬 '{response.content}'")
    except Exception as e:
        print(f"❌ Typhoon เชื่อมต่อไม่สำเร็จ กรุณาเช็ค API Key: {e}")

if __name__ == "__main__":
    test_api_connections()