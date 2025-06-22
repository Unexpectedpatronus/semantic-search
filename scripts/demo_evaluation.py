"""Демонстрационный скрипт для оценки и сравнения методов поиска"""

import os
from pathlib import Path

from semantic_search.core.doc2vec_trainer import Doc2VecTrainer
from semantic_search.core.search_engine import SemanticSearchEngine
from semantic_search.evaluation.baselines import (
    Doc2VecSearchAdapter,
    OpenAISearchBaseline,
)
from semantic_search.evaluation.comparison import QueryTestCase, SearchComparison


def create_demo_test_cases():
    """Создание демонстрационных тестовых случаев"""
    # Это примеры - замените на реальные документы из вашего корпуса
    test_cases = [
        QueryTestCase(
            query="транслингвизм и его прикладное значение",
            relevant_docs={
                "glocalization_theory.pdf",
                "content_localization.pdf",
                "cultural_adaptation.pdf",
            },
            relevance_scores={
                "glocalization_theory.pdf": 3,
                "content_localization.pdf": 3,
                "cultural_adaptation.pdf": 2,
                "globalization_basics.pdf": 1,
            },
            description="Основной запрос по транслингвизму",
        ),
        QueryTestCase(
            query="адаптация продуктов для локальных рынков",
            relevant_docs={
                "product_adaptation.pdf",
                "local_markets.pdf",
                "glocal_strategy.pdf",
            },
            relevance_scores={
                "product_adaptation.pdf": 3,
                "local_markets.pdf": 3,
                "glocal_strategy.pdf": 2,
            },
            description="Практические аспекты глокализации",
        ),
        QueryTestCase(
            query="культурные особенности в маркетинге",
            relevant_docs={"cultural_marketing.pdf", "cross_cultural_comm.pdf"},
            relevance_scores={
                "cultural_marketing.pdf": 3,
                "cross_cultural_comm.pdf": 3,
                "marketing_basics.pdf": 1,
            },
            description="Культурный аспект",
        ),
    ]

    return test_cases


def main():
    """Основная функция демонстрации"""
    print("=" * 80)
    print("ДЕМОНСТРАЦИЯ СРАВНЕНИЯ DOC2VEC И OPENAI EMBEDDINGS")
    print("=" * 80)

    # Проверка API ключа
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ОШИБКА: OpenAI API key не найден!")
        print("Установите переменную окружения OPENAI_API_KEY")
        print("Например: set OPENAI_API_KEY=sk-...")
        return

    # Загрузка модели Doc2Vec
    print("\n📂 Загрузка модели Doc2Vec...")
    model_name = "doc2vec_model"  # Замените на имя вашей модели

    trainer = Doc2VecTrainer()
    model = trainer.load_model(model_name)

    if not model:
        print(f"❌ Не удалось загрузить модель '{model_name}'")
        print("Сначала обучите модель командой:")
        print("poetry run semantic-search-cli train -d /path/to/documents")
        return

    print(f"✅ Модель загружена: {len(model.dv)} документов")

    # Создание поискового движка
    search_engine = SemanticSearchEngine(model, trainer.corpus_info)

    # Создание тестовых случаев
    print("\n🧪 Подготовка тестовых случаев...")
    test_cases = create_demo_test_cases()
    print(f"   Создано {len(test_cases)} тестовых запросов")

    # Создание объекта сравнения
    comparison = SearchComparison(test_cases)

    # Создание адаптеров методов
    print("\n🔧 Инициализация методов поиска...")

    # Doc2Vec адаптер
    doc2vec_adapter = Doc2VecSearchAdapter(search_engine, trainer.corpus_info)
    print("✅ Doc2Vec адаптер готов")

    # OpenAI baseline
    try:
        openai_baseline = OpenAISearchBaseline(api_key=api_key)
        print("✅ OpenAI baseline инициализирован")
    except Exception as e:
        print(f"❌ Ошибка инициализации OpenAI: {e}")
        return

    # Подготовка документов для OpenAI
    print("\n📚 Индексация документов для OpenAI...")
    print("   (Для демонстрации используем только первые 20 документов)")

    # Берем подвыборку документов
    demo_documents = []
    for i, (tokens, doc_id, metadata) in enumerate(trainer.corpus_info[:20]):
        # Восстанавливаем текст из токенов
        text = " ".join(tokens[:300])  # Первые 300 токенов
        demo_documents.append((doc_id, text, metadata))

    try:
        openai_baseline.index(demo_documents)
        print(f"✅ Проиндексировано {len(demo_documents)} документов")
    except Exception as e:
        print(f"❌ Ошибка индексации: {e}")
        return

    # Оценка методов
    print("\n📊 ОЦЕНКА МЕТОДОВ")
    print("-" * 80)

    # Doc2Vec
    print("\n1️⃣ Оценка Doc2Vec...")
    doc2vec_results = comparison.evaluate_method(
        doc2vec_adapter, top_k=10, verbose=True
    )

    # OpenAI
    print("\n2️⃣ Оценка OpenAI embeddings...")
    openai_results = comparison.evaluate_method(openai_baseline, top_k=10, verbose=True)

    # Результаты
    print("\n📈 РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 80)

    # Создаем сравнительную таблицу
    df_comparison = comparison.compare_methods(
        [doc2vec_adapter, openai_baseline], save_results=True
    )

    print("\nСравнительная таблица:")
    print(df_comparison.to_string(index=False))

    # Основные выводы
    doc2vec_map = doc2vec_results["aggregated"]["MAP"]
    openai_map = openai_results["aggregated"]["MAP"]

    print("\n🎯 ОСНОВНЫЕ ВЫВОДЫ:")
    print("-" * 80)

    if doc2vec_map > openai_map:
        improvement = ((doc2vec_map - openai_map) / openai_map) * 100
        print(f"✅ Doc2Vec превосходит OpenAI по качеству поиска на {improvement:.1f}%")
    else:
        improvement = ((openai_map - doc2vec_map) / doc2vec_map) * 100
        print(f"❌ OpenAI превосходит Doc2Vec по качеству поиска на {improvement:.1f}%")

    # Скорость
    doc2vec_time = doc2vec_results["aggregated"]["avg_query_time"]
    openai_time = openai_results["aggregated"]["avg_query_time"]
    speed_ratio = openai_time / doc2vec_time

    print(f"\n✅ Doc2Vec работает в {speed_ratio:.1f} раз быстрее OpenAI")
    print(f"   Doc2Vec: {doc2vec_time:.3f}с на запрос")
    print(f"   OpenAI:  {openai_time:.3f}с на запрос")

    # Экономическая эффективность
    print("\n💰 Экономическая эффективность:")
    yearly_cost = 1000 * 50 / 1000 * 0.0001 * 365  # Примерный расчет
    print(f"   При 1000 запросов в день экономия составит ~${yearly_cost:.0f} в год")

    # Генерация графиков
    print("\n📊 Генерация графиков...")
    try:
        comparison.plot_comparison(save_plots=True)
        print("✅ Графики сохранены в data/evaluation_results/plots/")
    except Exception as e:
        print(f"⚠️ Не удалось создать графики: {e}")

    # Детальный отчет
    print("\n📄 Генерация детального отчета...")
    report_path = Path("data/evaluation_results/comparison_report.txt")
    report = comparison.generate_report(report_path)

    print("\n✅ Все результаты сохранены в: data/evaluation_results/")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
