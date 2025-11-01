import gradio as gr
import joblib

# Загружаем модель
model = joblib.load("model.pkl")

# Функция для предсказания
def predict(features):
    prediction = model.predict([features])
    return f"Результат предсказания: {prediction[0]}"

# Интерфейс
inputs = gr.Textbox(label="Введите данные через запятую (пример: 5.1, 3.5, 1.4, 0.2)")
outputs = gr.Textbox(label="Результат")

demo = gr.Interface(fn=lambda x: predict([float(i) for i in x.split(',')]),
                    inputs=inputs,
                    outputs=outputs,
                    title="🌸 Titanic Predictor")

demo.launch()
