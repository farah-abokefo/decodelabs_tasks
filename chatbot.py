"""
PROJECT 1: RULE-BASED AI CHATBOT
Enhanced with Personality & Chat History
DecodeLabs - Batch 2026
"""

import re
import time
import random
import json
from datetime import datetime
import os

# ============================================
# 1. KNOWLEDGE BASE (Hash Map)
# ============================================

RESPONSES = {
    # Greetings
    "hello": "Hi there! Welcome to DecodeLabs. How can I help you today?",
    "hi": "Hello! I'm your AI assistant. What brings you here?",
    "hey": "Hey! Ready to explore the world of AI?",
    "good morning": "Good morning! A beautiful day to learn AI!",
    "good evening": "Good evening! Let's make this productive.",
    
    # Questions about AI
    "what is ai": "AI is the simulation of human intelligence in machines. I'm a rule-based example!",
    "what is machine learning": "ML is a subset of AI where systems learn from data. But I'm deterministic!",
    "tell me about decode labs": "DecodeLabs is an AI training hub in Greater Lucknow, India.",
    "what is python": "Python is a programming language. That's what I'm written in!",
    
    # Project specific
    "what is project 1": "Project 1 is the Rule-Based AI Chatbot - your foundation phase!",
    "what is an ipo model": "IPO = Input → Process → Output. It's the fundamental blueprint for any system.",
    
    # Personal
    "what is your name": "I'm DecodeBot, your rule-based AI assistant!",
    "who are you": "I'm a deterministic AI chatbot built for DecodeLabs Project 1.",
    
    # Help
    "help": """🤖 I can help you with:
    • Greetings (hello, hi, good morning)
    • AI questions (what is AI, machine learning)
    • About DecodeLabs
    • Python programming
    • Project 1 questions
    
    💡 Try: 'hello', 'what is AI', 'help', 'bye'""",
    "what can you do": "I'm a rule-based bot! I respond to predefined inputs using a dictionary. I can greet you, answer about AI, or just chat!",
    
    # Exit commands
    "bye": "Goodbye! Remember: An LLM without rules is a hallucination engine! 👋",
    "goodbye": "Farewell! Keep building that deterministic skeleton! 🎯",
    "exit": "Exiting now. Your journey to becoming an AI engineer has begun! 🚀",
    "quit": "Quitting... But your learning never stops! See you next time!",
    "see you": "See you! Come back anytime to practice your AI skills!"
}

FALLBACK = "I don't understand. I'm rule-based - try 'hello', 'help', or ask about AI!"

# ============================================
# 2. PERSONALITY (Option A)
# ============================================

PERSONALITY = {
    "catchphrases": [
        "Build that skeleton first! 🚀",
        "Traceability is key! 🔍",
        "Zero hallucinations here! 💡",
        "Deterministic > Probabilistic! 🎯",
        "White box all the way! 📦",
        "Control flow is everything! ⚡",
        "Master the logic engine! 🔧",
        "No mystery, just pure logic! ✨"
    ],
    "emojis": ["🤖", "✨", "💡", "🚀", "🎯", "🔧", "📦", "⚡"],
    "greetings_variations": [
        "Hey there!", "Hiya!", "Hello!", "Greetings!", "Howdy!"
    ],
    "farewell_variations": [
        "Catch you later!", "Until next time!", "See ya!", "Take care!"
    ],
    "encouragements": [
        "You're doing great!", "Keep it up!", "Awesome question!",
        "Great thinking!", "You're on the right track!"
    ]
}

def add_personality(response, user_input=None):
    """
    Add personality to responses based on context
    """
    # Don't add personality to help or exit messages
    if response == "EXIT" or response.startswith("🤖 I can help"):
        return response
    
    # Randomly add personality (30% chance)
    if random.random() < 0.3:
        # Choose different personality elements
        personality_type = random.choice([
            "catchphrase", "encouragement", "emojis", "none"
        ])
        
        if personality_type == "catchphrase":
            return f"{response} {random.choice(PERSONALITY['catchphrases'])}"
        
        elif personality_type == "encouragement" and len(user_input or "") > 3:
            return f"{response} {random.choice(PERSONALITY['encouragements'])}"
        
        elif personality_type == "emojis":
            emoji = random.choice(PERSONALITY['emojis'])
            return f"{response} {emoji}"
    
    return response

# ============================================
# 3. CHAT HISTORY (Option B)
# ============================================

class ChatHistory:
    """Manages conversation history with save/load capabilities"""
    
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.history = []
        self.session_start = datetime.now()
        self.exchange_count = 0
        
        # Create chat_logs directory if it doesn't exist
        if not os.path.exists('chat_logs'):
            os.makedirs('chat_logs')
    
    def add_exchange(self, user_input, bot_response):
        """Add a single exchange to history"""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "bot": bot_response,
            "exchange_id": self.exchange_count + 1
        }
        self.history.append(exchange)
        self.exchange_count += 1
        
        # Trim history if too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_recent_exchanges(self, n=5):
        """Get the n most recent exchanges"""
        return self.history[-n:] if self.history else []
    
    def get_session_stats(self):
        """Get statistics about the current session"""
        duration = datetime.now() - self.session_start
        minutes = int(duration.total_seconds() / 60)
        seconds = int(duration.total_seconds() % 60)
        
        # Count unique words used by user
        all_user_inputs = " ".join([h['user'] for h in self.history])
        unique_words = len(set(all_user_inputs.lower().split())) if all_user_inputs else 0
        
        return {
            "total_exchanges": self.exchange_count,
            "session_duration": f"{minutes}m {seconds}s",
            "unique_words_used": unique_words,
            "total_history": len(self.history)
        }
    
    def save_to_file(self, filename=None):
        """Save conversation history to a JSON file"""
        if not filename:
            timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
            filename = f"chat_logs/chat_{timestamp}.json"
        
        # Prepare data for saving
        data = {
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_exchanges": self.exchange_count,
            "history": self.history
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return filename
        except Exception as e:
            print(f"⚠️ Error saving chat: {e}")
            return None
    
    def load_from_file(self, filename):
        """Load conversation history from a JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.history = data.get('history', [])
            self.exchange_count = len(self.history)
            self.session_start = datetime.fromisoformat(data.get('session_start', datetime.now().isoformat()))
            return True
        except Exception as e:
            print(f"⚠️ Error loading chat: {e}")
            return False
    
    def display_history(self, n=None):
        """Display recent conversation history"""
        if not self.history:
            print("📝 No conversation history yet.")
            return
        
        exchanges = self.get_recent_exchanges(n) if n else self.history
        
        print("\n" + "="*50)
        print("📜 CONVERSATION HISTORY")
        print("="*50)
        for exchange in exchanges:
            timestamp = exchange.get('timestamp', '')
            if timestamp:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%H:%M:%S")
                print(f"[{time_str}] 👤 {exchange['user']}")
                print(f"         🤖 {exchange['bot']}")
                print("-"*50)
        print(f"📊 Total exchanges: {self.exchange_count}")
        print("="*50 + "\n")

# ============================================
# 4. KEYWORD PATTERN MATCHING
# ============================================

KEYWORD_PATTERNS = {
    r"hello|hi|hey|howdy|greetings": "hello",
    r"name|called|yourself|who are you": "what is your name",
    r"ai|artificial intelligence": "what is ai",
    r"ml|machine learning": "what is machine learning",
    r"decode|lab": "tell me about decode labs",
    r"python": "what is python",
    r"project 1|project one": "what is project 1",
    r"ipo": "what is an ipo model",
    r"help|support|assist": "help",
    r"bye|exit|quit|goodbye|see you|farewell": "bye"
}

# ============================================
# 5. FUNCTIONS
# ============================================

def sanitize_input(text):
    """Clean the user input"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def match_with_keywords(text):
    """Match input against keyword patterns"""
    for pattern, intent in KEYWORD_PATTERNS.items():
        if re.search(pattern, text):
            return intent
    return None

def get_response(text, chat_history=None):
    """
    Main processing function - The ENGINE
    Now with chat history integration
    """
    # Step 1: Sanitize
    clean_text = sanitize_input(text)
    
    if not clean_text:
        return "I didn't catch that. Please say something!"
    
    # Step 2: Check exit commands
    if clean_text in ["bye", "goodbye", "exit", "quit", "see you", "farewell"]:
        return "EXIT"  # Special signal to break the loop
    
    # Step 3: Try exact match (O(1) hash map lookup)
    if clean_text in RESPONSES:
        response = RESPONSES[clean_text]
        # Add personality
        response = add_personality(response, clean_text)
        return response
    
    # Step 4: Try keyword matching
    matched_intent = match_with_keywords(clean_text)
    if matched_intent and matched_intent in RESPONSES:
        response = RESPONSES[matched_intent]
        response = add_personality(response, clean_text)
        return response
    
    # Step 5: Check for follow-up questions (context awareness)
    if chat_history and chat_history.exchange_count > 0:
        recent = chat_history.get_recent_exchanges(1)
        if recent:
            last_user_input = recent[0]['user']
            # If user said "tell me more" or similar
            if re.search(r"tell me more|more about|elaborate", clean_text):
                return "I'd love to tell you more! But I'm rule-based - try asking a specific question like 'what is AI' or 'tell me about DecodeLabs'!"
    
    # Step 6: Fallback with personality
    fallback_responses = [
        "I don't quite understand that. I'm rule-based - try asking about AI, Python, or DecodeLabs!",
        "Hmm, that's new to me! I'm deterministic - I only know what I'm programmed to know!",
        "I'm not sure about that. Try 'hello', 'help', or ask about AI!",
        f"{random.choice(PERSONALITY['catchphrases'])} But I don't understand that specific question."
    ]
    
    response = random.choice(fallback_responses)
    # Add personality to fallback too
    response = add_personality(response, clean_text)
    return response

def display_welcome():
    """Display welcome message with personality"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║    🤖  WELCOME TO DECODELABS AI PROJECT 1                    ║
    ║                                                               ║
    ║    Rule-Based Chatbot | Batch 2026                           ║
    ║    "The White Box: Fully Traceable, 0 Hallucination"         ║
    ║                                                               ║
    ║    💡 Type 'help' for available commands                    ║
    ║    🚪 Type 'bye' or 'exit' to quit                          ║
    ║    📝 Type 'history' to see conversation history            ║
    ║    💾 Type 'save' to save chat history                      ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Random greeting
    greeting = random.choice(PERSONALITY["greetings_variations"])
    print(f"🤖 DecodeBot: {greeting} I'm your deterministic AI assistant!")
    print(f"💡 Tip: Try asking me about AI, Python, or DecodeLabs!\n")

# ============================================
# 6. MAIN CHATBOT ENGINE (with Personality + History)
# ============================================

def run_chatbot():
    """THE HEARTBEAT: The Infinite Loop with Personality & History"""
    
    # Initialize chat history
    chat_history = ChatHistory()
    save_on_exit = True
    
    # Display welcome
    display_welcome()
    
    running = True
    
    while running:
        # === INPUT PHASE ===
        user_input = input("👤 You: ")
        
        # Check for special commands
        if user_input.lower() in ["history", "show history"]:
            chat_history.display_history()
            continue
        
        if user_input.lower() in ["save", "save chat", "save history"]:
            filename = chat_history.save_to_file()
            if filename:
                print(f"💾 Chat history saved to: {filename}")
            continue
        
        if user_input.lower() in ["stats", "statistics"]:
            stats = chat_history.get_session_stats()
            print(f"""
            📊 SESSION STATISTICS:
            ├── Total exchanges: {stats['total_exchanges']}
            ├── Session duration: {stats['session_duration']}
            ├── Unique words used: {stats['unique_words_used']}
            └── History entries: {stats['total_history']}
            """)
            continue
        
        if user_input.lower() in ["clear", "clear history"]:
            chat_history.history = []
            chat_history.exchange_count = 0
            print("🗑️ History cleared!")
            continue
        
        # === PROCESS PHASE ===
        response = get_response(user_input, chat_history)
        
        # Check if exit signal
        if response == "EXIT":
            farewell = random.choice(PERSONALITY["farewell_variations"])
            print(f"🤖 DecodeBot: {farewell} Goodbye! Remember: Build that skeleton first! 👋")
            running = False
            
            # Save conversation on exit
            if save_on_exit and chat_history.exchange_count > 0:
                filename = chat_history.save_to_file()
                if filename:
                    print(f"💾 Conversation saved to: {filename}")
            continue
        
        # === OUTPUT PHASE ===
        print(f"🤖 DecodeBot: {response}\n")
        
        # Record in history
        chat_history.add_exchange(user_input, response)
        
        # Small delay for natural feel
        time.sleep(0.1)
    
    # Final message
    print(f"\n📊 Session completed: {chat_history.exchange_count} exchanges")
    print("🏁 Chatbot stopped successfully!")

# ============================================
# 7. ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        run_chatbot()
    except KeyboardInterrupt:
        print("\n\n🤖 DecodeBot: Interrupted! Remember: Control flow is everything in deterministic AI!")
        print("🏁 Chatbot stopped by user.")
    except Exception as e:
        print(f"\n⚠️ Unexpected error: {e}")
        print("🔄 Please restart the chatbot.")

# ============================================
# 8. ADDITIONAL UTILITIES
# ============================================

def view_saved_chats():
    """Utility to view all saved chat files"""
    if not os.path.exists('chat_logs'):
        print("📁 No chat logs directory found.")
        return
    
    files = [f for f in os.listdir('chat_logs') if f.endswith('.json')]
    
    if not files:
        print("📁 No saved chat files found.")
        return
    
    print("\n📂 SAVED CHAT FILES:")
    print("="*40)
    for i, file in enumerate(files, 1):
        filepath = os.path.join('chat_logs', file)
        size = os.path.getsize(filepath)
        print(f"{i}. {file} ({size} bytes)")
    print("="*40 + "\n")
    
    # Option to load a specific file
    choice = input("Enter file number to load (or press Enter to cancel): ")
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        filename = os.path.join('chat_logs', files[int(choice)-1])
        history = ChatHistory()
        if history.load_from_file(filename):
            history.display_history()
        return history
    return None

# Optional: If you want to run the utility
# if __name__ == "__main__":
#     view_saved_chats()