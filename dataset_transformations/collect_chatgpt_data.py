import asyncio
import json
import time
import traceback

import openai
from tqdm.auto import tqdm

openai.api_key = ""
openai.organization = ""

BATCH_SIZE = 10  # how many parallel requests to send to gpt
GPT_MODEL = "gpt-3.5-turbo"
MAX_ITER = 1000  # how many failures we tolerate
OFFSET = 0  # how many samples from the beginning we skip
TIMEOUT = 40  # max duration of request before we throw a timeout error
SAVE_PATH = (
    "test_gpt"  # path to save responses; each batch is saved in a separate json file
)
INITIAL_DATA_PATH = "test_data_unprocessed.json" # change to appropriate path
SLEEP = 0  # time in sec to sleep after the error before making a new request
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_TOKENS = 512
DELIMITER = "<delimiter!>"

responses = {}

# sample data: list of passages
dataset = json.load(open(INITIAL_DATA_PATH))


async def gather_with_concurrency(max_concurrency: int, *tasks: asyncio.Task):
    """
    Function that limits the # of concurrent tasks executed and executes a batch.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _semaphore_task(task: asyncio.Task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(_semaphore_task(task) for task in tasks))


def generate_prompt(text):
    """
    Generate prompt for ChatGPT
    """

    text, question, options, answer = text.split(DELIMITER)
    return [
        {
            "role": "user",
            "content": f"""There is a passage, a multiple choice question, and answer options.

            Passage:
            {text}

            Question:
            {question}

            Answer options:
            {options}

            Correct answer is {answer}. Provide an evidence snippet from the passage that helps to get to this correct answer.

            Example output:
            "EVIDENCE: [DIRECT QUOTE from the passage]"
            """,
        },  
    ]


def run_query(concept):
    prompt = generate_prompt(concept)
    return asyncio.ensure_future(
        openai.ChatCompletion.acreate(
            model=GPT_MODEL,
            messages=prompt,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['"""'],
        )
    )

    return response


async def main(dataset: list[str]) -> list[str]:
    global OFFSET
    global responses
    for _ in range(MAX_ITER):
        for i in tqdm(range(OFFSET, len(dataset), BATCH_SIZE)):
            print(i, len(dataset))
            batch = [
                item
                for item in dataset[i : i + BATCH_SIZE]
                if (item not in responses or len(responses[item]) == 0)
            ]
            tasks = [run_query(sample) for sample in batch]
            try:
                batch_response = await asyncio.wait_for(
                    gather_with_concurrency(BATCH_SIZE, *tasks), timeout=TIMEOUT
                )
            except:
                if SLEEP > 0:
                    time.sleep(SLEEP)
                print(traceback.format_exc())
                break
            for sample, response in zip(batch, batch_response):
                results = []
                for result in response["choices"][0]["message"]["content"].split("\n"):
                    results += [result]
                responses[sample] = results
            json.dump(
                responses,
                open(f"{SAVE_PATH}/responses_{i}.json", "w"),
            )
            del responses
            responses = {}
            OFFSET = i + BATCH_SIZE


loop = asyncio.get_event_loop()
loop.run_until_complete(main(dataset))
