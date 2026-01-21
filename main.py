import os
from dotenv import load_dotenv
from smolagents import OpenAIModel, ToolCallingAgent
from tools.rag_tool import retrieve_knowledge
from knowledge_base.vector_store.chroma_repo import ChromaVectorStore
from tools.rag_tool import set_vector_store

load_dotenv()

model = OpenAIModel(
    model_id="gemini-2.0-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
)

vector_store_instance = ChromaVectorStore()
retrieve_knowledge.vector_store = vector_store_instance

SYSTEM_PROMPT = """
Ты - вежливый и профессиональный оператор техподдержки
компании RideWay.

Отвечай только на основе информации из базы знаний (инструмент retrieve_knowledge).

Если информации недостаточно — честно скажи: "К сожалению, у меня нет точной
информации по этому вопросу. Рекомендую обратиться к руководству или в вторую линию поддержки."

Не придумывай факты.

Вопрос пользователя:
""".strip()

agent = ToolCallingAgent(
    tools=[retrieve_knowledge],
    model=model,
)


vector_store_instance = ChromaVectorStore()
set_vector_store(vector_store_instance)

print("Агент техподдержки запущен! Задавай вопросы (или 'exit' для выхода).\n")

while True:
    user_input = input("Вы: ").strip()
    if user_input.lower() in ["exit", "quit", "выход"]:
        print("До свидания!")
        break
    if not user_input:
        continue

    full_task = f"{SYSTEM_PROMPT}\n{user_input}"

    try:
        response = agent.run(full_task)
        print(f"\nАгент: {response}\n")
    except Exception as e:
        print(f"Ошибка: {e}")
