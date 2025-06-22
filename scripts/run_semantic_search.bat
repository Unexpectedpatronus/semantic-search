@echo off
REM Батник для запуска Semantic Search на Windows

echo ====================================
echo    Semantic Search Launcher
echo ====================================
echo.

REM Проверяем наличие Poetry
where poetry >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Poetry не найден в PATH!
    echo.
    echo Установите Poetry: https://python-poetry.org/docs/#installation
    echo.
    pause
    exit /b 1
)

REM Проверяем наличие виртуального окружения
if not exist ".venv" (
    echo [INFO] Создание виртуального окружения...
    poetry install
    if %errorlevel% neq 0 (
        echo [ERROR] Ошибка установки зависимостей!
        pause
        exit /b 1
    )
)

REM Меню выбора
:menu
echo Выберите режим запуска:
echo.
echo 1. GUI приложение
echo 2. CLI интерфейс
echo 3. Проверка зависимостей
echo 4. Список моделей
echo 5. Очистка кэша
echo 6. Выход
echo.

set /p choice="Введите номер (1-6): "

if "%choice%"=="1" goto gui
if "%choice%"=="2" goto cli
if "%choice%"=="3" goto check
if "%choice%"=="4" goto models
if "%choice%"=="5" goto cache
if "%choice%"=="6" goto end

echo Неверный выбор!
echo.
goto menu

:gui
echo.
echo Запуск GUI приложения...
poetry run semantic-search
goto end

:cli
echo.
echo Запуск CLI интерфейса...
echo.
poetry run semantic-search-cli --help
echo.
pause
goto menu

:check
echo.
echo Проверка зависимостей...
echo.
poetry run python scripts/check_dependencies.py
echo.
pause
goto menu

:models
echo.
echo Список доступных моделей...
echo.
poetry run python scripts/list_models.py
echo.
pause
goto menu

:cache
echo.
echo Очистка кэша...
echo.
poetry run python scripts/cache_clear.py
echo.
pause
goto menu

:end
echo.
echo Спасибо за использование Semantic Search!
timeout /t 3 >nul