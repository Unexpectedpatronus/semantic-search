"""Скрипт для установки и проверки SpaCy модели"""

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
    "--model",
    default="ru_core_news_sm",
    help="Название модели SpaCy для загрузки",
)
def main(model: str):
    """Установка и проверка SpaCy модели"""

    logger.info("Проверка установки SpaCy...")

    try:
        import spacy

        logger.info(f"SpaCy версии {spacy.__version__} установлен")
    except ImportError:
        logger.error("SpaCy не установлен!")
        logger.info("Установите SpaCy командой: pip install spacy")
        sys.exit(1)

    logger.info(f"Проверка наличия модели {model}...")

    if check_spacy_model(model):
        logger.success(f"Модель {model} уже установлена!")

        # Проверяем работу модели
        try:
            nlp = spacy.load(model)
            doc = nlp("Это тестовое предложение для проверки.")
            logger.info(f"Модель работает корректно. Токенов: {len(doc)}")

            # Показываем информацию о модели
            logger.info(f"Язык: {nlp.lang}")
            logger.info(f"Компоненты: {nlp.pipe_names}")

        except Exception as e:
            logger.error(f"Ошибка при тестировании модели: {e}")

    else:
        logger.warning(f"Модель {model} не найдена")

        response = click.confirm("Хотите загрузить модель?", default=True)
        if response:
            if download_spacy_model(model):
                logger.success(f"Модель {model} успешно загружена!")

                # Проверяем после загрузки
                if check_spacy_model(model):
                    logger.success("Модель готова к использованию!")
                else:
                    logger.error("Модель загружена, но не может быть загружена")
            else:
                logger.error("Не удалось загрузить модель")
                logger.info(
                    f"Попробуйте загрузить вручную: python -m spacy download {model}"
                )
        else:
            logger.info("Загрузка отменена")

    # Дополнительная информация
    logger.info("\nДоступные русские модели SpaCy:")
    logger.info("- ru_core_news_sm (маленькая, ~15 МБ)")
    logger.info("- ru_core_news_md (средняя, ~45 МБ)")
    logger.info("- ru_core_news_lg (большая, ~550 МБ)")


if __name__ == "__main__":
    main()
