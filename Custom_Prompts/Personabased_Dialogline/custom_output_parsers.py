from langchain.schema.output_parser import BaseOutputParser
from pydantic import Field

class LastSentenceOutputParser(BaseOutputParser):
    last_sentence: str = Field()

    def parse(self, text: str) -> str:
        sentences = text.split(self.last_sentence)
        return sentences[-1]

