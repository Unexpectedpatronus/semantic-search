"""Скрипт для установки SpaCy модели"""

import subprocess
import sys

from loguru import logger


def install_spacy_model():
    """Установка русской модели SpaCy"""
    try:
        logger.info("Начинаем установку SpaCy модели ru_core_news_sm...")

        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", "ru_core_news_sm"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info("✅ SpaCy модель успешно установлена")
            print("✅ SpaCy модель ru_core_news_sm установлена успешно!")
            return True
        else:
            logger.error(f"❌ Ошибка установки: {result.stderr}")
            print(f"❌ Ошибка установки SpaCy модели: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        print(f"❌ Критическая ошибка при установке: {e}")
        return False


if __name__ == "__main__":
    install_spacy_model()
