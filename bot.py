import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the Flan-T5 model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
chatbot_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chatbot_model.to(device)

# Dictionary of text files corresponding to each food class
food_info_files = {
    "pizza": "clean_pizza_info.txt",
    "pasta": "clean_pasta_info.txt",
    "pancake": "clean_pancake_info.txt",
    "French fries": "clean_french_fries_info.txt",
    "donut": "clean_donut_info.txt"
}

# Specific nutritional data per food item, ensure this only extracts the relevant portion
def load_nutritional_info(predicted_food):
    file_path = food_info_files.get(predicted_food)
    if file_path:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            nutritional_info = ""
            for line in lines:
                # Extracting relevant nutritional data like calories, protein, fats etc.
                if any(key in line.lower() for key in ["calories", "protein", "total fat", "saturated fat"]):
                    nutritional_info += line.strip() + "\n"
            return nutritional_info if nutritional_info else "Nutritional information not found."
    else:
        return "No information available."

# Function to generate chatbot responses
def chatbot_response(question, food_context, predicted_food):
    if not question.strip():
        return f"Can you ask something specific about {predicted_food}? 😄"

    if predicted_food.lower() not in question.lower():
        return f"Oops! I can only help you with questions about {predicted_food}. Please try asking something else! 😊"

    # Prepare the prompt for the Flan-T5 model
    prompt = f"Question: {question}\nContext: {food_context}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Generate a response
    output = chatbot_model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.99, temperature=0.4)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return answer.strip() + " 🍽️"
