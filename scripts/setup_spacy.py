# Обновите файл scripts/setup_spacy.py:

"""Скрипт для установки и проверки SpaCy моделей"""

import subprocess
import sys

import click
import spacy
from loguru import logger


def check_spacy_model(model_name: str) -> bool:
    """Проверка наличия модели SpaCy"""
    try:
        spacy.load(model_name)
        return True
    except OSError:
        return False


def download_spacy_model(model_name: str) -> bool:
    """Загрузка модели SpaCy"""
    try:
        logger.info(f"Загрузка модели {model_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        return False


@click.command()
@click.option(
    "--russian/--no-russian",
    default=True,
    help="Установить русскую модель",
)
@click.option(
    "--english/--no-english",
    default=True,
    help="Установить английскую модель",
)
def main(russian: bool, english: bool):
    """Установка и проверка SpaCy моделей для русского и английского языков"""

    logger.info("Проверка установки SpaCy...")

    try:
        import spacy

        logger.info(f"SpaCy версии {spacy.__version__} установлен")
    except ImportError:
        logger.error("SpaCy не установлен!")
        logger.info("Установите SpaCy командой: pip install spacy")
        sys.exit(1)

    models_to_install = []

    if russian:
        models_to_install.append(("ru_core_news_sm", "Русская"))
    if english:
        models_to_install.append(("en_core_web_sm", "Английская"))

    if not models_to_install:
        logger.warning("Не выбрана ни одна модель для установки")
        return

    installed_count = 0

    for model_name, model_desc in models_to_install:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Проверка {model_desc} модели ({model_name})...")

        if check_spacy_model(model_name):
            logger.success(f"{model_desc} модель уже установлена!")

            # Проверяем работу модели
            try:
                nlp = spacy.load(model_name)
                test_text = (
                    "Это тестовое предложение."
                    if "ru" in model_name
                    else "This is a test sentence."
                )
                doc = nlp(test_text)
                logger.info(f"Модель работает корректно. Токенов: {len(doc)}")
                logger.info(f"Язык: {nlp.lang}")
                logger.info(f"Компоненты: {nlp.pipe_names}")
                installed_count += 1

            except Exception as e:
                logger.error(f"Ошибка при тестировании модели: {e}")

        else:
            logger.warning(f"{model_desc} модель не найдена")

            response = click.confirm(
                f"Хотите загрузить {model_desc} модель?", default=True
            )
            if response:
                if download_spacy_model(model_name):
                    logger.success(f"{model_desc} модель успешно загружена!")

                    # Проверяем после загрузки
                    if check_spacy_model(model_name):
                        logger.success("Модель готова к использованию!")
                        installed_count += 1
                    else:
                        logger.error("Модель загружена, но не может быть загружена")
                else:
                    logger.error("Не удалось загрузить модель")
                    logger.info(
                        f"Попробуйте загрузить вручную: python -m spacy download {model_name}"
                    )
            else:
                logger.info("Загрузка отменена")

    # Итоговая информация
    logger.info(f"\n{'=' * 50}")
    logger.info("ИТОГИ УСТАНОВКИ:")
    logger.info(f"Установлено моделей: {installed_count} из {len(models_to_install)}")

    if installed_count == len(models_to_install):
        logger.success("✅ Все модели успешно установлены!")
    elif installed_count > 0:
        logger.warning("⚠️ Установлены не все модели")
    else:
        logger.error("❌ Ни одна модель не установлена")

    # Дополнительная информация
    logger.info("\n📚 Доступные модели SpaCy:")
    logger.info("Русские:")
    logger.info("  - ru_core_news_sm (маленькая, ~15 МБ)")
    logger.info("  - ru_core_news_md (средняя, ~45 МБ)")
    logger.info("  - ru_core_news_lg (большая, ~500 МБ)")
    logger.info("\nАнглийские:")
    logger.info("  - en_core_web_sm (маленькая, ~15 МБ)")
    logger.info("  - en_core_web_md (средняя, ~40 МБ)")
    logger.info("  - en_core_web_lg (большая, ~400 МБ)")
    logger.info("  - en_core_web_trf (transformer, ~450 МБ)")


if __name__ == "__main__":
    main()
