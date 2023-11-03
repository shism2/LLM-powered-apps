class SimpleListChatMemory:
    def __init__(self):
        self.chat_history = list()

    def __len__(self):
        return len(self.chat_history)

    def __call__(self, question, answer):
        self.chat_history.append((question, answer))

    def clear_memory(self):
        self.chat_history = list()
        return self.chat_history