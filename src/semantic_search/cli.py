# cli.py — логика командной строки
import click

from semantic_search.core.text_summarizer import TextSummarizer
from semantic_search.utils.notification_system import NotificationManager
from semantic_search.utils.task_manager import TaskManager

cli = click.Group()
notifier = NotificationManager()
task_manager = TaskManager()


@cli.command()
@click.argument("text")
def summarize_text(text):
    """Суммаризация по тексту"""
    result = TextSummarizer().summarize(text)
    click.echo(result)


@cli.command()
@click.argument("file_path")
def summarize_file(file_path):
    """Суммаризация по файлу"""
    result = TextSummarizer().summarize_file(file_path)
    click.echo(result)


@cli.command()
@click.argument("folder_path")
def summarize_batch(folder_path):
    """Пакетная суммаризация по папке"""
    results = TextSummarizer().summarize_folder(folder_path)
    for res in results:
        click.echo(res)


@cli.command()
def status():
    """Показать статус текущих задач"""
    click.echo(task_manager.get_status())


@cli.command()
def cancel():
    """Отменить текущие задачи"""
    task_manager.cancel_all()
    click.echo("Все задачи отменены.")


@cli.command()
def cleanup():
    """Очистить временные файлы или кэш"""
    task_manager.cleanup()
    click.echo("Кэш очищен.")


@cli.command()
def system_info():
    """Вывести информацию о системе и зависимостях"""
    from platform import platform, python_version

    click.echo(f"Python {python_version()} on {platform()}")
