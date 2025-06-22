"""Скрипт для сборки исполняемого файла с помощью PyInstaller"""

import argparse
import shutil
import sys
from pathlib import Path


def clean_build_dirs():
    """Очистка старых директорий сборки"""
    dirs_to_clean = ["build", "dist"]
    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"✓ Удалена директория {dir_name}")


def create_spec_file(onefile=False, windowed=True, name="SemanticSearch"):
    """Создание spec файла для PyInstaller"""

    # Пути
    src_path = Path(__file__).parent.parent / "src"
    main_script = src_path / "semantic_search" / "main.py"

    # Дополнительные данные
    datas = [
        # Конфигурации
        ("config", "config"),
        # Иконка если есть
        # ("assets/icon.ico", "assets"),
    ]

    # Скрытые импорты
    hidden_imports = [
        "semantic_search",
        "semantic_search.core",
        "semantic_search.gui",
        "semantic_search.utils",
        "semantic_search.evaluation",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "gensim.models.doc2vec",
        "sklearn.metrics.pairwise",
        "spacy",
        "click",
        "loguru",
        "docx",
        "pymupdf",
    ]

    # Исключения
    excludes = [
        "matplotlib",
        "notebook",
        "jupyter",
        "pytest",
        "mypy",
        "ruff",
    ]

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    [r'{main_script}'],
    pathex=[r'{src_path}'],
    binaries=[],
    datas={datas},
    hiddenimports={hidden_imports},
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={excludes},
    noarchive=False,
)

pyz = PYZ(a.pure)

"""

    if onefile:
        spec_content += f"""exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='{name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=not {windowed},
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)"""
    else:
        spec_content += f"""exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=not {windowed},
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{name}',
)"""

    spec_file = Path(f"{name}.spec")
    with open(spec_file, "w", encoding="utf-8") as f:
        f.write(spec_content)

    print(f"✓ Создан spec файл: {spec_file}")
    return spec_file


def build_executable(spec_file, clean=True):
    """Запуск PyInstaller для сборки"""
    import subprocess

    cmd = ["pyinstaller"]

    if clean:
        cmd.append("--clean")

    cmd.extend(["--noconfirm", str(spec_file)])

    print("\n🔨 Запуск PyInstaller...")
    print(f"Команда: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Сборка завершена успешно!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка сборки: {e}")
        return False


def post_build_actions(name="SemanticSearch", onefile=False):
    """Действия после сборки"""

    if onefile:
        exe_path = Path("dist") / f"{name}.exe"
        if exe_path.exists():
            print(f"\n📦 Исполняемый файл создан: {exe_path}")
            print(f"   Размер: {exe_path.stat().st_size / 1024 / 1024:.1f} МБ")
    else:
        dist_dir = Path("dist") / name
        if dist_dir.exists():
            exe_path = dist_dir / f"{name}.exe"
            if exe_path.exists():
                print(f"\n📦 Приложение собрано в: {dist_dir}")
                print(f"   Исполняемый файл: {exe_path}")

                # Создаем директории для данных
                for dir_name in ["data", "logs", "config"]:
                    (dist_dir / dir_name).mkdir(exist_ok=True)
                print("   ✓ Созданы директории для данных")

                # Копируем README
                readme_src = Path("README.md")
                if readme_src.exists():
                    shutil.copy2(readme_src, dist_dir)
                    print("   ✓ Скопирован README.md")


def main():
    parser = argparse.ArgumentParser(
        description="Сборка Semantic Search в исполняемый файл"
    )
    parser.add_argument("--onefile", action="store_true", help="Собрать в один файл")
    parser.add_argument(
        "--windowed", action="store_true", help="Без консоли (GUI режим)"
    )
    parser.add_argument(
        "--name", default="SemanticSearch", help="Имя исполняемого файла"
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="Не очищать старые сборки"
    )

    args = parser.parse_args()

    print("🚀 Semantic Search Builder")
    print("=" * 50)

    # Проверка PyInstaller
    try:
        import PyInstaller

        print(f"✓ PyInstaller {PyInstaller.__version__} установлен")
    except ImportError:
        print("❌ PyInstaller не установлен!")
        print("Установите: poetry add --group dev pyinstaller")
        sys.exit(1)

    # Очистка старых сборок
    if not args.no_clean:
        clean_build_dirs()

    # Создание spec файла
    spec_file = create_spec_file(
        onefile=args.onefile, windowed=args.windowed, name=args.name
    )

    # Сборка
    if build_executable(spec_file, clean=not args.no_clean):
        post_build_actions(args.name, args.onefile)

        print("\n💡 Подсказки:")
        print("   - Для уменьшения размера используйте UPX")
        print("   - Добавьте иконку в assets/icon.ico")
        print("   - Протестируйте на чистой системе")

        if args.onefile:
            print("\n⚠️  Предупреждение: ")
            print("   Режим onefile может работать медленнее")
            print("   и вызывать ложные срабатывания антивирусов")
    else:
        print("\n❌ Сборка не удалась. Проверьте логи выше.")
        sys.exit(1)


if __name__ == "__main__":
    main()
