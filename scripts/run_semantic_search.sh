#!/bin/bash
# Скрипт для запуска Semantic Search на Linux/Mac

echo "===================================="
echo "    Semantic Search Launcher"
echo "===================================="
echo

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Проверяем наличие Poetry
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Poetry не найден!"
    echo
    echo "Установите Poetry: https://python-poetry.org/docs/#installation"
    echo
    exit 1
fi

# Проверяем/создаем виртуальное окружение
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}[INFO]${NC} Создание виртуального окружения..."
    poetry install
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${NC} Ошибка установки зависимостей!"
        exit 1
    fi
fi

# Функция для показа меню
show_menu() {
    echo
    echo "Выберите режим запуска:"
    echo
    echo "1. GUI приложение"
    echo "2. CLI интерфейс"
    echo "3. Проверка зависимостей"
    echo "4. Список моделей"
    echo "5. Очистка кэша"
    echo "6. Выход"
    echo
}

# Основной цикл
while true; do
    show_menu
    read -p "Введите номер (1-6): " choice
    
    case $choice in
        1)
            echo
            echo -e "${GREEN}Запуск GUI приложения...${NC}"
            poetry run semantic-search
            break
            ;;
        2)
            echo
            echo -e "${GREEN}Запуск CLI интерфейса...${NC}"
            echo
            poetry run semantic-search-cli --help
            echo
            read -p "Нажмите Enter для продолжения..."
            ;;
        3)
            echo
            echo -e "${GREEN}Проверка зависимостей...${NC}"
            echo
            poetry run python scripts/check_dependencies.py
            echo
            read -p "Нажмите Enter для продолжения..."
            ;;
        4)
            echo
            echo -e "${GREEN}Список доступных моделей...${NC}"
            echo
            poetry run python scripts/list_models.py
            echo
            read -p "Нажмите Enter для продолжения..."
            ;;
        5)
            echo
            echo -e "${GREEN}Очистка кэша...${NC}"
            echo
            poetry run python scripts/cache_clear.py
            echo
            read -p "Нажмите Enter для продолжения..."
            ;;
        6)
            echo
            echo -e "${GREEN}Спасибо за использование Semantic Search!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Неверный выбор!${NC}"
            ;;
    esac
done