# Emotion-Recognizer

Проект для распознавания эмоций с помощью камеры. (Python 3.8+)

---

### 🚀 Возможности

- Отслеживание лица в реальном времени    
- Распознавание шести основных эмоций

    Радость (happy)
    
    Грусть (sad)
    
    Злость (angry)
    
    Удивление (surprise)
    
    Страх (fear)
    
    Отвращение (disgust)

    Нейтральное состояние (neutral)
- Выход из программы по нажатию `q`  

---

### 🧠 Используемые библиотеки

- opencv-python — захват и отображение изображения  
- deepface — распознавание эмоций  
- mtcnn — обнаружение лиц и ключевых точек (глаз, носа, рта)  
- tensorflow — фреймворк, используемый DeepFace и MTCNN
- numpy — численные вычисления  

---

### ⚙️ Запуск проекта

Перед стартом выполнить команды:

```bash
pip install -r requirements.txt
python main.py
```
Скачать exe файл для windows (без установки python и др):
https://github.com/VaNeN1Y/emotion-recognizer/releases/download/emotion-recognizer/emotion-recognizer.exe