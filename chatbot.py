import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer directly
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def generate_response(user_input):
    # Encode the input and generate a response
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

class SolarCalculator:
    def __init__(self, daily_energy_kwh, sunlight_hours, panel_cost):
        self.daily_energy_kwh = daily_energy_kwh
        self.sunlight_hours = sunlight_hours
        self.panel_cost = panel_cost
        self.panel_wattage = 250  # Hardcoded panel wattage
        self.panel_efficiency = 25  # Hardcoded panel efficiency in percentage
        self.panel_area = 1.7  # Hardcoded panel area in square meters

    def calculate_number_of_panels(self):
        daily_energy_wh = self.daily_energy_kwh * 1000  # convert kWh to Wh
        daily_panel_output = self.panel_wattage * self.sunlight_hours * (self.panel_efficiency / 100)
        number_of_panels = daily_energy_wh / daily_panel_output
        return number_of_panels

    def calculate_area_required(self):
        number_of_panels = self.calculate_number_of_panels()
        total_area = number_of_panels * self.panel_area
        return total_area

    def calculate_total_cost(self):
        number_of_panels = self.calculate_number_of_panels()
        total_cost = number_of_panels * self.panel_cost
        return total_cost

def main():
    st.title("Solar Calculator Chatbot")
    st.write("I can help you calculate the number of solar panels you need, the area required, and the total cost.")
    
    daily_energy_kwh = st.number_input("Enter your daily energy consumption in kWh", min_value=0.0, step=0.1)
    sunlight_hours = st.number_input("Enter average daily sunlight hours", min_value=0.0, step=0.1)
    panel_cost = st.number_input("Enter cost per solar panel (in USD)", min_value=0.0, step=0.1)
    
    calculator = SolarCalculator(daily_energy_kwh, sunlight_hours, panel_cost)

    # Buttons for calculations
    if st.button("Calculate Number of Panels"):
        number_of_panels = calculator.calculate_number_of_panels()
        st.write(f"Bot: You need approximately {number_of_panels:.2f} solar panels.")

    if st.button("Calculate Area Required"):
        total_area = calculator.calculate_area_required()
        st.write(f"Bot: You need approximately {total_area:.2f} square meters of area for the solar panels.")
    
    if st.button("Calculate Total Cost"):
        total_cost = calculator.calculate_total_cost()
        st.write(f"Bot: The total cost for the solar panels is approximately ${total_cost:.2f}.")

    # Text input for conversational queries
    user_input = st.text_input("You:", "")
    
    if user_input:
        response = generate_response(user_input)
        st.write(f"Bot: {response}")

if __name__ == "__main__":
    main()
