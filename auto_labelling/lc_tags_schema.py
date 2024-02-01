schema = {
    "properties": {
        ".due_date": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "description": "they contain information or inquiries related to payment due dates, \
                payment requirements, or any mention of specific due dates in the context of financial \
                transactions or bills. This includes questions about when a payment will be required, \
                inquiries about billing dates, requests for information about the next due date, \
                and mentions of specific due dates in monetary contexts. ",
        },
        "balance_amount": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "description": "describes how aggressive the statement is, the higher the number the more aggressive",
        },
        "aggressiveness": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "describes how aggressive the statement is, the higher the number the more aggressive",
        },
        "language": {
            "type": "string",
            "enum": ["spanish", "english", "french", "german", "italian"],
        },
    },
    "required": [".due_date", "balance_amount", ....],
}


def build_schema(label_defs, scale=10):
    label_dict = {}
    for label, d in label_defs.items():
        label_dict[label]["description"] = d
        label_dict[label]["enum"] = [i for i in range(1, scale+1)]
        label_dict[label]["type"] = "integer"
    schema = {}
    schema["properties"] = label_dict
    schema["required"] = list(label_dict.keys())
    return schema
    

from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.chat_models import ChatOpenAI

# LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key="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")
chain = create_tagging_chain(schema, llm)
inp = "What is my balance?"
chain.run(inp)