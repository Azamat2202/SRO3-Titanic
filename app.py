import gradio as gr
import joblib
import os

# Загружаем модель
model = joblib.load("model.pkl")

# Функция для предсказания
def predict(features):
    try:
        values = [float(i) for i in features.split(",")]
        prediction = model.predict([values])
        return f"Результат предсказания: {prediction[0]}"
    except Exception as e:
        return f"Ошибка: {str(e)}"

# Интерфейс Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Введите данные через запятую (пример: 5.1, 3.5, 1.4, 0.2)"),
    outputs=gr.Textbox(label="Результат"),
    title="🌸 Titanic Predictor"
)

# Получаем порт от Render
port = int(os.getenv("PORT", 10000))

# Запускаем сервер
demo.launch(server_name="0.0.0.0", server_port=port)
