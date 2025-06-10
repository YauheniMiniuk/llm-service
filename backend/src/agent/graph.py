import os
import multiprocessing
import json
import traceback

from agent.tools_and_schemas import SearchQueryList, Reflection, RouteQuery
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client
from langchain_core.pydantic_v1 import ValidationError
from langchain_community.tools import DuckDuckGoSearchRun
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool
from typing import Literal

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
    router_prompt,
)
# Удален импорт ChatGoogleGenerativeAI, так как он больше не используется
from langchain_community.chat_models import ChatLlamaCpp
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from langchain_core.messages import HumanMessage

load_dotenv()

# === ИНИЦИАЛИЗАЦИЯ МОДЕЛИ (1 РАЗ) ===
# Создаем один экземпляр модели, который будет использоваться во всем приложении.
# Это экономит память и время на загрузку.
llm = ChatLlamaCpp(
    model_path=os.getenv("MODEL_PATH"),
    temperature=0.7,
    n_ctx=10000,
    n_gpu_layers=-1,  # Настрой под свою видеокарту. Если нет GPU, ставь 0.
    n_batch=300,
    max_tokens=4096, # Увеличил для более полных ответов
    n_threads=multiprocessing.cpu_count() - 1,
    verbose=True,
)
search_tool = DuckDuckGoSearchRun()


# --- Узлы графа ---

def route_question(state: OverallState, config: RunnableConfig) -> dict:
    """
    Узел-диспетчер. Решает, нужно ли искать информацию в интернете.
    """
    print("\n--- УЗЕЛ: Диспетчер (route_question) ---")
    try:
        topic = get_research_topic(state["messages"])
        
        # Формируем промпт для диспетчера
        prompt = router_prompt.format(research_topic=topic)
        
        # Вызываем LLM для принятия решения
        raw_result = llm.invoke(prompt)
        print(f"--- DEBUG: Ответ диспетчера от LLM: {raw_result.content}")
        
        # Парсим ответ
        json_block = raw_result.content[raw_result.content.find('{') : raw_result.content.rfind('}')+1]
        parsed_json = json.loads(json_block)
        route_object = RouteQuery.parse_obj(parsed_json)
        
        print(f"--- РЕШЕНИЕ: Выбран путь '{route_object.path}' ---")
        return {"path": route_object.path}

    except Exception as e:
        # В случае любой ошибки, для безопасности отправляем на простой ответ
        print(f"--- ОШИБКА ДИСПЕТЧЕРА: {e}. Выбираем путь 'simple_chat'. ---")
        return {"path": "simple_chat"}


def simple_answer(state: OverallState, config: RunnableConfig) -> dict:
    """
    Узел для прямого ответа на простые вопросы.
    """
    print("\n--- УЗЕЛ: Простой ответ (simple_answer) ---")
    topic = get_research_topic(state["messages"])
    
    # Просто просим модель ответить на вопрос
    result = llm.invoke(topic)
    
    return {"messages": [AIMessage(content=result.content)]}


def decide_path(state: OverallState) -> Literal["research", "simple_chat"]:
    """
    Функция-условие, которая направляет граф по выбранному пути.
    """
    return state["path"]

def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """Генерирует поисковые запросы (ФИНАЛЬНАЯ ВЕРСИЯ С .replace())."""
    
    try:
        print("\n--- DEBUG: 1. Вход ---")
        configurable = Configuration.from_runnable_config(config)

        if state.get("initial_search_query_count") is None:
            state["initial_search_query_count"] = configurable.number_of_initial_queries
        
        # --- ШАГ 1: Собираем все нужные значения ---
        number_queries = state["initial_search_query_count"]
        topic = get_research_topic(state["messages"])
        date = get_current_date()
        print(f"--- DEBUG: Тема: '{topic}', Дата: '{date}', Кол-во: {number_queries} ---")

        # --- ШАГ 2: Собираем шаблон промпта ---
        json_schema_string = json.dumps(SearchQueryList.schema(), indent=2)
        base_instructions = query_writer_instructions
        json_prompt_part = f"""\n\nYou MUST format your output as a JSON object that adheres to the following schema:\n```json\n{json_schema_string}\n```"""
        final_prompt_template = base_instructions + json_prompt_part
        
        # --- ШАГ 3: ЗАМЕНЯЕМ плейсхолдеры вручную через .replace() ---
        prompt_with_date = final_prompt_template.replace('{current_date}', date)
        prompt_with_topic = prompt_with_date.replace('{research_topic}', topic)
        formatted_prompt = prompt_with_topic.replace('{number_queries}', str(number_queries))
        
        print("\n--- УСПЕХ! Промпт собран. Отправка в LLM... ---")
        print("--- ИТОГОВЫЙ ПРОМПТ ---")
        print(formatted_prompt)
        print("------------------------")

        raw_result = llm.invoke(formatted_prompt)
        print(f"--- DEBUG: Получен сырой ответ от модели:\n{raw_result.content}")
    
        queries = []
        try:
            json_block = raw_result.content[raw_result.content.find('{') : raw_result.content.rfind('}')+1]
            if json_block:
                parsed_json = json.loads(json_block)
                pydantic_object = SearchQueryList.parse_obj(parsed_json)
                queries = pydantic_object.query
                print(f"--- DEBUG: Запросы успешно распознаны: {queries} ---")
            else:
                print(f"--- ERROR: Не удалось найти JSON блок в ответе модели. ---")
        except (json.JSONDecodeError, ValidationError, AttributeError) as e:
            print(f"--- ERROR: Не удалось распознать JSON из ответа модели. Ошибка: {e} ---")
            queries = []

        return {"query_list": queries}

    except Exception as e:
        print(f"\n\n--- КРИТИЧЕСКАЯ ОШИБКА ---\n")
        traceback.print_exc()
        raise e

def continue_to_web_research(state: QueryGenerationState):
    """Отправляет запросы в узел веб-поиска."""
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """
    Выполняет веб-поиск с помощью DuckDuckGo.
    """
    search_query = state["search_query"]
    print(f"--- ИЩУ В DUCKDUCKGO: '{search_query}' ---")

    # Просто запускаем инструмент поиска с нашим запросом
    search_result = search_tool.run(search_query)
    
    print(f"--- РЕЗУЛЬТАТ ПОЛУЧЕН. Длина: {len(search_result)} символов. ---")

    # Так как у нас нет структурированных источников, sources_gathered будет пустым
    return {
        "sources_gathered": [], # У DDG нет автоматических цитат
        "search_query": [search_query],
        "web_research_result": [search_result],
    }



def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """Анализирует найденную информацию и ищет пробелы в знаниях (РУЧНОЙ ПАРСИНГ)."""
    try:
        print("\n--- DEBUG: 1. Вход в функцию reflection ---")
        
        # --- ШАГ 1: Собираем все нужные значения ---
        research_topic = get_research_topic(state["messages"])
        summaries = "\n\n---\n\n".join(state["web_research_result"])
        current_date = get_current_date()

        # --- ШАГ 2: Собираем шаблон промпта ---
        # Получаем JSON-схему для нашего класса Reflection
        json_schema_string = json.dumps(Reflection.schema(), indent=2)
        base_instructions = reflection_instructions
        json_prompt_part = f"""\n\nYou MUST format your output as a JSON object that adheres to the following schema:\n```json\n{json_schema_string}\n```"""
        final_prompt_template = base_instructions + json_prompt_part
        
        # --- ШАГ 3: ЗАМЕНЯЕМ плейсхолдеры вручную через .replace() ---
        prompt_with_date = final_prompt_template.replace('{current_date}', current_date)
        prompt_with_topic = prompt_with_date.replace('{research_topic}', research_topic)
        formatted_prompt = prompt_with_topic.replace('{summaries}', summaries)
        
        print("\n--- УСПЕХ! Промпт для reflection собран. Отправка в LLM... ---")

        # --- ШАГ 4: Вызываем LLM и получаем сырой ответ ---
        raw_result = llm.invoke(formatted_prompt)
        print(f"--- DEBUG: Получен сырой ответ от модели:\n{raw_result.content}")

        # --- ШАГ 5: Парсим JSON вручную с обработкой ошибок ---
        # Задаем значения по умолчанию на случай ошибки
        is_sufficient = True # Лучше остановить цикл, если что-то пошло не так
        knowledge_gap = "Could not parse model output."
        follow_up_queries = []

        json_block = raw_result.content[raw_result.content.find('{') : raw_result.content.rfind('}')+1]
        if json_block:
            parsed_json = json.loads(json_block)
            pydantic_object = Reflection.parse_obj(parsed_json)
            
            is_sufficient = pydantic_object.is_sufficient
            knowledge_gap = pydantic_object.knowledge_gap
            follow_up_queries = pydantic_object.follow_up_queries
            print(f"--- DEBUG: Результат reflection успешно распознан. ---")
        else:
            print(f"--- ERROR: Не удалось найти JSON блок в ответе reflection. ---")

    except (json.JSONDecodeError, ValidationError, AttributeError) as e:
        print(f"--- ERROR: Не удалось распознать JSON из ответа reflection. Ошибка: {e} ---")
        # Оставляем значения по умолчанию
        pass
    
    except Exception as e:
        print(f"\n\n--- КРИТИЧЕСКАЯ ОШИБКА ВНУТРИ reflection ---\n")
        traceback.print_exc()
        raise e

    # --- ШАГ 6: Возвращаем результат ---
    research_loop_count = state.get("research_loop_count", 0) + 1
    return {
        "is_sufficient": is_sufficient,
        "knowledge_gap": knowledge_gap,
        "follow_up_queries": follow_up_queries,
        "research_loop_count": research_loop_count,
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(state: ReflectionState, config: RunnableConfig) -> str:
    """Решает, продолжать исследование или завершать."""
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """Формирует финальный ответ."""
    formatted_prompt = answer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )

    # Используем глобальную модель 'llm'
    result = llm.invoke(formatted_prompt)

    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)
            
    print("Final result: ", result)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# --- НОВАЯ СБОРКА ГРАФА ---

builder = StateGraph(OverallState, config_schema=Configuration)

# Добавляем все узлы, включая новые
builder.add_node("route_question", route_question)
builder.add_node("simple_answer", simple_answer)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Входная точка - диспетчер
builder.add_edge(START, "route_question")

# Добавляем ГЛАВНОЕ УСЛОВИЕ
builder.add_conditional_edges(
    "route_question", # Узел, после которого принимается решение
    decide_path,      # Функция, которая возвращает название следующего узла
    {
        # Словарь: "если вернулось это -> иди сюда"
        "research": "generate_query",
        "simple_chat": "simple_answer"
    }
)

# Старая логика для пути "Исследование"
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)

# Оба финальных узла теперь ведут к концу графа
builder.add_edge("simple_answer", END)
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")