import tkinter as tk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pretrained model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

class ChatBotGUI:
    def __init__(self, root):  # Corrected __init__ method
        self.root = root
        self.root.title("Transformers ChatBot")
        self.root.geometry("500x650")
        self.chat_history_ids = None

        # Chat log window
        self.chat_log = tk.Text(self.root, bd=1, bg="white", font=("Arial", 12), wrap=tk.WORD)
        self.chat_log.config(state=tk.DISABLED)
        self.chat_log.place(x=6, y=6, height=500, width=480)

        # Scrollbar
        self.scrollbar = tk.Scrollbar(self.chat_log)
        self.chat_log['yscrollcommand'] = self.scrollbar.set
        self.scrollbar.place(x=480, y=6, height=500)

        # Input box
        self.entry_box = tk.Text(self.root, bd=1, bg="lightgrey", font=("Arial", 12), wrap=tk.WORD)
        self.entry_box.place(x=6, y=510, height=90, width=380)

        # Send button
        self.send_button = tk.Button(self.root, text="Send", font=("Arial", 12, 'bold'), width=12, height=5,
                                     bd=0, bg="#008080", activebackground="#008080", fg="#ffffff",
                                     command=self.send_message)
        self.send_button.place(x=390, y=510, height=90)

    def display_message(self, message, sender="You"):
        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_log.config(state=tk.DISABLED)
        self.chat_log.yview(tk.END)

    def send_message(self):
        user_input = self.entry_box.get("1.0", 'end-1c').strip()
        self.entry_box.delete("0.0", tk.END)

        if user_input != "":
            self.display_message(user_input, "You")

            # Encode the user input and append it to the chat history
            new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

            # Generate a response using the model
            if self.chat_history_ids is not None:
                bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
            else:
                bot_input_ids = new_input_ids

            # Get model response
            self.chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

            self.display_message(response, "Bot")

# Run the GUI application
root = tk.Tk()
chat_interface = ChatBotGUI(root)
root.mainloop()