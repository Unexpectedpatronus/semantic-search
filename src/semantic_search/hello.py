# import win32com.client

# try:
#     word = win32com.client.Dispatch("Word.Application")
#     word.Visible = True  # Показать интерфейс Word (для отладки)
#     doc = word.D(
#         r"C:\Users\evgen\Evgeny\Dev_projects\Dev_Python\diplom\СТАТЬИ по ГЛОКАЛИЗАЦИИ\Контактная вариантология\Homi Bhabha.docx"
#     )
#     print("Файл успешно открыт в Word!")
#     doc.Close()
#     word.Quit()
# except Exception as e:
#     print(f"Ошибка: {e}")

from pathlib import Path

print("PATH:  " + str(Path(__file__).parent.parent.parent))
