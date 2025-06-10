import win32com.client

try:
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = True  # Показать интерфейс Word (для отладки)
    doc = word.Documents.Open(
        r"C:\Users\evgen\Evgeny\Dev_projects\Dev_Python\diplom\СТАТЬИ по ГЛОКАЛИЗАЦИИ\Контактная вариантология\ReviewofBrajB.doc"
    )
    print("Файл успешно открыт в Word!")
    doc.Close()
    word.Quit()
except Exception as e:
    print(f"Ошибка: {e}")
