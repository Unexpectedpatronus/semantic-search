# Semantic Document Search

Десктопное приложение для интеллектуального семантического поиска по локальной базе документов с использованием модели Doc2Vec.

## 🚀 Возможности

- 🔍 **Семантический поиск** - находит документы по смыслу, а не только по ключевым словам
- 📄 **Поддержка популярных форматов** - PDF, DOCX, DOC
- 📝 **Автоматическое создание выжимок** - экстрактивная суммаризация документов
- 🧠 **Обучение собственных моделей** - создавайте модели на ваших документах
- 💻 **Удобный графический интерфейс** - современный и интуитивный
- 🚀 **Высокая производительность** - многопоточная обработка и кэширование

## 🛠️ Установка

### 1. Клонирование репозитория
```bash
git clone https://github.com/yourusername/semantic-search.git
cd semantic-search
```

### 2. Установка Poetry (если не установлен)
```bash
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# Linux/MacOS
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Установка зависимостей
```bash
# Установка всех зависимостей через Poetry
poetry install
```

### 4. Установка языковой модели SpaCy
```bash
# Автоматическая установка
poetry run python scripts/setup_spacy.py

# Или вручную
poetry run python -m spacy download ru_core_news_sm
```

## 🚀 Запуск приложения

### Графический интерфейс
```bash
poetry run semantic-search
```

### Командная строка
```bash
poetry run semantic-search-cli --help
```

## 📖 Использование

### Графический интерфейс

1. **Обучение модели**
   - Перейдите на вкладку "Обучение"
   - Выберите папку с документами
   - Настройте параметры модели
   - Нажмите "Начать обучение"

2. **Поиск документов**
   - Выберите обученную модель из списка
   - Введите поисковый запрос
   - Нажмите "Поиск"
   - Кликните на результат для просмотра

3. **Создание выжимок**
   - Перейдите на вкладку "Суммаризация"
   - Выберите файл документа
   - Укажите количество предложений
   - Нажмите "Создать выжимку"

### Командная строка

#### Обучение модели
```bash
# Базовое обучение
poetry run semantic-search-cli train -d /path/to/documents

# С настройкой параметров
poetry run semantic-search-cli train -d /path/to/documents \
    --model my_model \
    --vector-size 200 \
    --epochs 50
```

#### Поиск документов
```bash
poetry run semantic-search-cli search -d /path/to/documents \
    -q "машинное обучение и нейронные сети" \
    --top-k 5
```

#### Создание выжимок
```bash
# Для одного файла
poetry run semantic-search-cli summarize-file -f document.pdf -s 5

# Для всех документов в папке
poetry run semantic-search-cli summarize-batch -d /path/to/documents \
    -s 3 -o /path/to/summaries
```

#### Просмотр статистики
```bash
poetry run semantic-search-cli stats -d /path/to/documents -m my_model
```

#### Команды конфига
```bash
# Показать текущую конфигурацию
poetry run semantic-search-cli config --show

# Установить максимальную длину текста
poetry run semantic-search-cli config --set text_processing.max_text_length 5000000

# Перезагрузить конфигурацию
poetry run semantic-search-cli config --reload
```

## 🔬 Оценка и сравнение с OpenAI

Приложение включает модуль для количественного сравнения качества поиска между обученной моделью Doc2Vec и OpenAI embeddings.

### Установка дополнительных зависимостей

```bash
# Установка с поддержкой OpenAI
poetry install -E openai

# Или отдельно
pip install openai pandas
```

### Настройка OpenAI API

```bash
# Windows
setx OPENAI_API_KEY "your_api_key_here"

# Linux/MacOS
export OPENAI_API_KEY="your_api_key_here"
```

### Запуск сравнения

#### Через GUI

1. Откройте вкладку "📊 Оценка методов"
2. Введите OpenAI API ключ
3. Выберите набор тестов
4. Нажмите "Запустить сравнение"

#### Через CLI

```bash
# Быстрая оценка (3 теста)
poetry run semantic-search-cli evaluate --test-cases quick

# Стандартная оценка (5 тестов)
poetry run semantic-search-cli evaluate --test-cases standard

# Расширенная оценка (10 тестов)
poetry run semantic-search-cli evaluate --test-cases extended

# С указанием модели и директории результатов
poetry run semantic-search-cli evaluate \
    --model my_model \
    --test-cases standard \
    --output-dir ./evaluation_results
```

#### Демонстрационный скрипт

```bash
# Запуск демонстрации
poetry run python scripts/demo_evaluation.py
```

### Метрики оценки

Система оценивает следующие метрики:

- **MAP (Mean Average Precision)** - основная метрика качества ранжирования
- **MRR (Mean Reciprocal Rank)** - позиция первого релевантного результата
- **Precision@k** - точность в топ-k результатах
- **Recall@k** - полнота в топ-k результатах
- **NDCG@k** - нормализованный дисконтированный накопленный выигрыш
- **Скорость поиска** - среднее время на запрос
- **Экономическая эффективность** - стоимость использования

### Результаты оценки

После выполнения оценки создаются:

```
data/evaluation_results/
├── comparison_results.csv      # Таблица с метриками
├── comparison_report.txt       # Текстовый отчет
├── detailed_results.json       # Детальные результаты
└── plots/
    ├── comparison_plots.png    # Сравнительные графики
    └── *_detailed.png          # Детальные графики по методам
```

### Интерпретация результатов

#### Преимущества Doc2Vec:
- ✅ **Скорость**: В 10-100 раз быстрее OpenAI
- ✅ **Автономность**: Работает без интернета
- ✅ **Стоимость**: Бесплатно после обучения
- ✅ **Конфиденциальность**: Данные остаются локальными
- ✅ **Специализация**: Лучше для узкоспециализированных корпусов

#### Преимущества OpenAI:
- ✅ **Универсальность**: Работает без предварительного обучения
- ✅ **Многоязычность**: Поддержка 100+ языков
- ✅ **Обновления**: Постоянные улучшения модели

### Рекомендации по выбору метода

| Сценарий | Рекомендация |
|----------|--------------|
| Специализированный корпус документов | Doc2Vec |
| Высокая нагрузка (>1000 запросов/день) | Doc2Vec |
| Конфиденциальные данные | Doc2Vec |
| Работа без интернета | Doc2Vec |
| Универсальный поиск | OpenAI |
| Малый объем данных | OpenAI |
| Многоязычные документы | OpenAI |


## 🏗️ Сборка в исполняемый файл

```bash
# Установка PyInstaller (если не установлен)
poetry add --group dev pyinstaller

# Сборка в один файл
poetry run python scripts/build.py --onefile --windowed

# Сборка в папку
poetry run python scripts/build.py
```

## 🧪 Тестирование

```bash
# Запуск всех тестов
poetry run pytest

# С покрытием кода
poetry run pytest --cov=semantic_search

# Только юнит-тесты
poetry run pytest tests/test_core_functionality.py

# Бенчмарки производительности
poetry run pytest tests/ -k "performance" --benchmark-only
```

## ⚙️ Конфигурация

Приложение автоматически создает файл конфигурации `config/app_config.json` при первом запуске.

### Основные параметры:

```json
{
  "text_processing": {
    "min_text_length": 100,
    "max_text_length": 5000000,
    "min_tokens_count": 10,
    "min_token_length": 2,
    "min_sentence_length": 10,
    "remove_stop_words": true,
    "lemmatize": true,
    "max_file_size_mb": 100,
    "chunk_size": 800000,
    "spacy_max_length": 3000000
  },
  "doc2vec": {
    "vector_size": 150,
    "window": 10,
    "min_count": 2,
    "epochs": 40,
    "workers": 15,
    "seed": 42,
    "dm": 1,
    "negative": 5,
    "hs": 0,
    "sample": 0.0001
  },
  "search": {
    "default_top_k": 10,
    "max_top_k": 100,
    "similarity_threshold": 0.1,
    "enable_caching": true,
    "cache_size": 1000,
    "enable_filtering": true
  }
}
```

## 🔧 Решение проблем

### SpaCy модель не найдена
```bash
# Переустановите модель
poetry run python -m spacy download ru_core_news_sm --force
```

### Ошибка при обработке .doc файлов
- Убедитесь, что установлен Microsoft Word
- Или конвертируйте .doc файлы в .docx

### Недостаточно памяти при обучении
- Уменьшите размер батча в конфигурации
- Используйте меньшую размерность векторов
- Обрабатывайте документы частями

## 👤 Автор

**Evgeny Odintsov**
- Email: ev1genial@gmail.com
- GitHub: [@unexpectedpatronus](https://github.com/Unexpectedpatronus)

## 🙏 Благодарности

- [Gensim](https://radimrehurek.com/gensim/) - за отличную реализацию Doc2Vec
- [SpaCy](https://spacy.io/) - за мощные инструменты NLP
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - за возможность создания GUI

## 📈 Roadmap

- [ ] Поддержка английского языка
- [ ] Экспорт результатов поиска
- [ ] Интеграция с облачными хранилищами
- [ ] Веб-интерфейс
- [ ] Поддержка других форматов (TXT, RTF, ODT)
- [ ] Улучшенная визуализация результатов
- [ ] Пакетная обработка запросов
- [ ] API для интеграции с другими приложениями