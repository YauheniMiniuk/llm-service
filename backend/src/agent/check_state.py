
import typing
from agent.state import OverallState

print("--- Проверяем структуру OverallState ---")

# Эта функция показывает все поля, которые Python видит в твоем классе
type_hints = typing.get_type_hints(OverallState)

# Печатаем результат
for key, value in type_hints.items():
    print(f"Поле: '{key}', Тип: {value}")

if 'path' in type_hints:
    print("\n✅ ОТЛИЧНО! Поле 'path' найдено в схеме состояния.")
else:
    print("\n❌ ПРОБЛЕМА! Поле 'path' ОТСУТСТВУЕТ в схеме состояния.")