import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re
def fetch_website_content(url):
    print("Fetching website content...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
            content = "\n".join([el.get_text() for el in elements if el.get_text().strip()])
            if content:
                print("Website content fetched successfully!")
                return content
            else:
                return "Error: No readable content found on the website."
        else:
            return f"Failed to fetch data. Status Code: {response.status_code}"
    except Exception as e:
        return f"Error fetching website: {e}"
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def setup_qa_model():
    print("Loading question answering model...")
    model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    print("Model is ready!")
    return model
def get_predefined_answer(question):
    predefined_answers = {
        "hey": "Hey, how can I help you?",
        "hello": "Hello! What would you like to know?",
        "what are project specifications": "Project specifications are detailed descriptions of the requirements, scope, and objectives of a project.",
        "what are the key features": "Some of the key features include live chat, unified inbox, analytics, and mobile app integration.",
        "who are the partners": "The partners include top-tier global companies in various sectors such as technology, retail, and healthcare.",
        "what is botpenguin": "BotPenguin is a platform offering AI chatbots for businesses to automate customer support, marketing, and lead generation."
    }
    question = re.sub(r'[^\w\s]', '', question).lower().strip()
    return predefined_answers.get(question, None)

def get_model_answer(model, question, context_chunks):
    for chunk in context_chunks:
        answer = model(question=question, context=chunk)
        if answer['score'] > 0.4:  
            return answer['answer']
    return "I'm sorry, I couldn't find a relevant answer."
def chatbot_console(url):
    website_data = fetch_website_content(url)
    if "Failed" in website_data or "Error" in website_data:
        print(website_data)
        return

    context_chunks = chunk_text(website_data)
    model = setup_qa_model()
    print("\nYou can now chat with the bot! Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        
        predefined_answer = get_predefined_answer(user_input)
        if predefined_answer:
            print(f"Chatbot: {predefined_answer}")
        else:
            print("Chatbot: Thinking...")
            model_answer = get_model_answer(model, user_input, context_chunks)
            print(f"Chatbot: {model_answer}")

if __name__ == "__main__":
    website_url = input("Enter a website URL to scrape content from: ").strip()
    chatbot_console(website_url)
