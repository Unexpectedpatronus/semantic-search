from src.semantic_search.config import CACHE_DIR
from src.semantic_search.utils.cache_manager import CacheManager


def clear_cache():
    """Очистка кэша приложения"""
    cache_manager = CacheManager(CACHE_DIR)
    if cache_manager.clear():
        print("Кэш успешно очищен.")
    else:
        print("Произошла ошибка при очистке кэша.")


if __name__ == "__main__":
    clear_cache()
