import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";

export default function App() {
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  
  // Этот ref будет отслеживать, что финальный узел был достигнут
  const hasFinalEventOccurredRef = useRef(false);

  const thread = useStream<{
    messages: Message[];
    initial_search_query_count: number;
    max_research_loops: number;
    reasoning_model: string;
  }>({
    apiUrl: import.meta.env.DEV
      ? "http://localhost:2024"
      : "http://localhost:8123", // production URL
    assistantId: "agent",
    messagesKey: "messages",
    onFinish: (event: any) => {
      console.log("Run finished:", event);
    },
    onUpdateEvent: (event: any) => {
      // event - это объект, где ключ - это имя узла, который только что отработал
      // например, { "route_question": { "path": "simple_chat" } }

      let processedEvent: ProcessedEvent | null = null;
      
      // --- НАЧАЛО ИЗМЕНЕНИЙ ---

      if (event.route_question) {
        // --- НОВЫЙ ОБРАБОТЧИК для нашего диспетчера ---
        processedEvent = {
          title: "Принятие решения",
          data: `Выбран путь: ${event.route_question.path}`,
        };
      } else if (event.generate_query) {
        processedEvent = {
          title: "Генерация поисковых запросов",
          data: `Запросы: "${event.generate_query.query_list.join('", "')}"`,
        };
      } else if (event.web_research) {
        // --- ИЗМЕНЕННЫЙ ОБРАБОТЧИК для веб-поиска ---
        // Показываем сам запрос, а не кол-во источников (которое всегда 0)
        const searchQuery = event.web_research.search_query[0] || "N/A";
        processedEvent = {
          title: "Веб-поиск (DuckDuckGo)",
          data: `Ищу информацию по запросу: "${searchQuery}"`,
        };
      } else if (event.reflection) {
        processedEvent = {
          title: "Анализ информации",
          data: event.reflection.is_sufficient
            ? "Информации достаточно, генерирую финальный ответ."
            : `Нужна доп. информация: "${event.reflection.knowledge_gap}"`,
        };
      } else if (event.finalize_answer) {
        // --- Обработчик для финализации ИССЛЕДОВАНИЯ ---
        processedEvent = {
          title: "Подготовка ответа",
          data: "Собираю все части воедино и формирую итоговый ответ.",
        };
        hasFinalEventOccurredRef.current = true;
      } else if (event.simple_answer) {
        // --- НОВЫЙ ОБРАБОТЧИК для ПРОСТОГО ответа ---
         processedEvent = {
          title: "Прямой ответ",
          data: "Генерирую прямой ответ на ваш вопрос.",
        };
        hasFinalEventOccurredRef.current = true;
      }
      
      // --- КОНЕЦ ИЗМЕНЕНИЙ ---

      if (processedEvent) {
        setProcessedEventsTimeline((prevEvents) => [
          ...prevEvents,
          processedEvent!,
        ]);
      }
    },
  });

  // Этот useEffect отвечает за автопрокрутку чата вниз
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [thread.messages, processedEventsTimeline]); // Добавил processedEventsTimeline для лучшей прокрутки

  // Этот useEffect отвечает за сохранение истории "мыслей" агента
  useEffect(() => {
    if (
      hasFinalEventOccurredRef.current &&
      !thread.isLoading &&
      thread.messages.length > 0
    ) {
      const lastMessage = thread.messages[thread.messages.length - 1];
      if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...processedEventsTimeline],
        }));
      }
      hasFinalEventOccurredRef.current = false;
    }
  }, [thread.messages, thread.isLoading, processedEventsTimeline]);

  const handleSubmit = useCallback(
    (submittedInputValue: string, effort: string, model: string) => {
      if (!submittedInputValue.trim()) return;
      
      // Очищаем таймлайн для нового запуска
      setProcessedEventsTimeline([]);
      hasFinalEventOccurredRef.current = false;

      // ... твой код для определения initial_search_query_count и max_research_loops ...
      let initial_search_query_count = 3;
      let max_research_loops = 3;
      // ...

      const newMessages: Message[] = [
        ...(thread.messages || []),
        {
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        },
      ];

      thread.submit({
        messages: newMessages,
        initial_search_query_count: initial_search_query_count,
        max_research_loops: max_research_loops,
        reasoning_model: model,
      });
    },
    [thread]
  );

  const handleCancel = useCallback(() => {
    thread.stop();
  }, [thread]);

  return (
     // ... твой JSX остается без изменений ...
     <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="flex-1 flex flex-col overflow-hidden max-w-4xl mx-auto w-full">
        <div
          className={`flex-1 overflow-y-auto ${
            thread.messages.length === 0 ? "flex" : ""
          }`}
        >
          {thread.messages.length === 0 ? (
            <WelcomeScreen
              handleSubmit={handleSubmit}
              isLoading={thread.isLoading}
              onCancel={handleCancel}
            />
          ) : (
            <ChatMessagesView
              messages={thread.messages}
              isLoading={thread.isLoading}
              scrollAreaRef={scrollAreaRef}
              onSubmit={handleSubmit}
              onCancel={handleCancel}
              liveActivityEvents={processedEventsTimeline}
              historicalActivities={historicalActivities}
            />
          )}
        </div>
      </main>
    </div>
  );
}