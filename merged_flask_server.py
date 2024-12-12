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
    output = chatbot_model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.95, temperature=0.7)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return answer.strip() + " 🍽️"


from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load Flan-T5 model for chatbot
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
chatbot_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chatbot_model.to(device)

# Load food classification model
model_path = 'final_food_model.pth'

class FoodClassificationModel(nn.Module):
    def __init__(self):
        super(FoodClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 5)  # Assuming 5 output classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load saved model
model = FoodClassificationModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Nutritional data for each food
nutritional_data = {
    "pancake": {
        "calories": 227, "totalFat": 9, "saturatedFat": 3, "protein": 5
    },
    "pizza": {
        "calories": 266, "totalFat": 10, "saturatedFat": 4, "protein": 11
    },
    "pasta": {
        "calories": 131, "totalFat": 1, "saturatedFat": 0.2, "protein": 5
    },
    "French fries": {
        "calories": 300, "totalFat": 15, "saturatedFat": 2.5, "protein": 3
    },
    "donut": {
        "calories": 452, "totalFat": 25, "saturatedFat": 10, "protein": 4
    }
}

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Store the last predicted food globally
predicted_food_global = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global predicted_food_global

    try:
        file = request.files['file']
        if file:
            image = Image.open(file.stream).convert('RGB')
            processed_image = preprocess_image(image)

            # Predict using the PyTorch model
            with torch.no_grad():
                output = model(processed_image)
                _, predicted_class = torch.max(output, 1)

            class_labels = ['pancake', 'pizza', 'pasta', 'French fries', 'donut']
            predicted_food = class_labels[predicted_class.item()]
            predicted_food_global = predicted_food  # Store predicted food

            # Retrieve nutritional information
            nutrition_info = nutritional_data[predicted_food]

            return jsonify({
                'prediction': predicted_food,
                'calories': nutrition_info['calories'],
                'totalFat': nutrition_info['totalFat'],
                'saturatedFat': nutrition_info['saturatedFat'],
                'protein': nutrition_info['protein']
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot():
    global predicted_food_global

    if predicted_food_global is None:
        return jsonify({'response': 'Please predict the food before asking questions.'})

    data = request.get_json()
    question = data['question']

    if predicted_food_global.lower() not in question.lower():
        return jsonify({
            'response': f"I can only answer questions related to {predicted_food_global}. Please ask something about {predicted_food_global}."
        })

    context = f"Information about {predicted_food_global}"

    # Tokenize the input and create prompt
    prompt = f"Question: {question}\nContext: {context}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Generate response from the chatbot model
    output = chatbot_model.generate(**inputs, max_new_tokens=50, do_sample=True, top_p=0.95, temperature=0.7)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(debug=True)
