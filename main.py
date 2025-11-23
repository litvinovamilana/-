from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Инициализация моделей
sentiment_analazer = pipeline("sentiment-analysis", 
                              model="blanchefort/rubert-base-cased-sentiment")

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")

app = Flask(__name__)

def generate_recomendation(mood):
    prompt = f"Посоветуй ОДИН фильм для человека с настроением: {mood}. Фильм должен быть 2010-2025 года. Кратко объясни почему этот фильм подходит и укажи возрастное ограничение. Фильм:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_length=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Извлекаем только сгенерированную часть (после промпта)
    response = generated_text[len(prompt):].strip()
    
    # Очищаем ответ от возможных повторений
    if "Фильм:" in response:
        response = response.split("Фильм:")[-1].strip()
    
    return response if response else "К сожалению, не удалось подобрать фильм. Попробуйте еще раз!"

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendation = ""
    user_text = ""
    ai_result = ""
    
    if request.method == "POST":
        user_text = request.form["message"]
        
        try:
            result = sentiment_analazer(user_text)[0]
            label = result['label']
            
            if label.upper() == "POSITIVE":
                recommendation = "доволен😊"
            elif label.upper() == "NEGATIVE":
                recommendation = "недоволен😔"
            else:
                recommendation = "вообще недоволен"
            
            # Генерация рекомендации
            ai_text = generate_recomendation(recommendation)
            ai_result = ai_text           
            
        except Exception as e:
            recommendation = "Ошибка анализа настроения"
            ai_result = "Попробуйте еще раз!"
            print(f"Error: {e}")
    
    return render_template('rec.html', 
                         recommendation=recommendation, 
                         user_text=user_text, 
                         ai_result=ai_result)

if __name__ == '__main__':
    app.run(debug=True)