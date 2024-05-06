import os
import json
import unittest
import openai
import logging
import time
from dotenv import load_dotenv
import argparse

load_dotenv()

root_logger = logging.getLogger(__name__)
ROOT_LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=ROOT_LOG_FORMAT, filemode="w", force=True)
OPENAI_ENGINE = 'code-davinci-002' # change this for using other engine
OPENAI_PROMPT_LINE_TOKEN = '#'
OPENAI_PROMPT_HEADER = f'{OPENAI_PROMPT_LINE_TOKEN} Translate the following SQL query into natural language question:'
OPENAI_SQL_QUERY_TITLE = f'{OPENAI_PROMPT_LINE_TOKEN} SQL query:'
OPENAI_NL_QUESTION_TITLE = f'{OPENAI_PROMPT_LINE_TOKEN} Natural language questions:'
OPENAI_PROMPT_STOP_SEQ = '\n'

def load_args():
    parser = argparse.ArgumentParser(description="Build GPT3 prompter, request the API and log the results")
    parser.add_argument('--query_only', action='store_true', help='if added: build prompter only with query, else: prompt text must include original nl question')
    parser.add_argument('--dataset_name', type=str, defalut='spider', help='')
    parser.add_argument('--json_file', type=str, required=True, help='')
    parser.add_argument('--output_file', type=str, required=True, help='')
    return parser

# data is a list of dicts that contains element d, which has following attr.
def prompt_builder(d, prompt_header=OPENAI_PROMPT_HEADER, sql_query_title=OPENAI_SQL_QUERY_TITLE, nl_question_title=OPENAI_NL_QUESTION_TITLE, promp_line_token=OPENAI_PROMPT_LINE_TOKEN):
    original_sql = d['original_query']
    original_nl = d['question']
    generated_sql = d['generated_query']
    prompt_txt = []
    prompt_txt.append(prompt_header)
    prompt_txt.append(f'{sql_query_title} {original_sql}')
    prompt_txt.append(f'{promp_line_token}')
    prompt_txt.append(f'{nl_question_title}')
    prompt_txt.append(f'{promp_line_token}(1)')
    res = '\n'.join(prompt_txt)
    return res


def prompt_builder_with_query_only(d, i, prompt_header=OPENAI_PROMPT_HEADER, sql_query_title=OPENAI_SQL_QUERY_TITLE, nl_question_title=OPENAI_NL_QUESTION_TITLE, promp_line_token=OPENAI_PROMPT_LINE_TOKEN, prompt_txt=[]):
    if i == 0:
        generated_sql = d['query']
        prompt_txt.append(prompt_header)
        # prompt_txt.append(f'{promp_line_token}')
        prompt_txt.append(f'{sql_query_title}')
        prompt_txt.append(generated_sql)
        prompt_txt.append(f'{nl_question_title} ')
    else:
        prompt_txt = prompt_txt.strip().split('\n')
    prompt_txt.append(f'{promp_line_token} ({i}) ')
    res = '\n'.join(prompt_txt)
    return res


def prompt_request(prompt_txt, engine=OPENAI_ENGINE, sec=3):
    openai.api_key = OPENAI_API_KEY
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    response = None
    while response is None:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt_txt,
                temperature=0.1,
                max_tokens=100,
                top_p=1.0,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                stop=["#"]
            )
            time.sleep(sec)
            root_logger.info(f"sleep {sec} sec...") # use sleep to fit the throttling of API
        except Exception as e:
            root_logger.error(f"{e}")
            sec += sec
            time.sleep(sec)
            pass
    return response


def parse_gpt3_response(prompt_txt, response, logger, verbose=False):
    results = []
    """Example of response
    {
        "choices": [
            {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "text": "How many heads of the departments are younger than 56 ?\n"
            }
        ],
        "created": 1645008276,
        "id": "cmpl-4cC5UlueiVrnR6KldMS6zK3WE9FRp",
        "model": "code-davinci:001",
        "object": "text_completion"
    }
    """
    if any(response['choices']):
        for choice in response['choices']:
            if len(choice['text']) > 0:
                # log file
                if verbose:
                    logger.info(prompt_txt + choice['text'] + '\n')
                results.append(choice['text'])
        # print(response)
        return results
    else:
        raise Exception(f"Empty response: {response}")


def gpt3_iter_responses(d, logger, iter_n):
    prompt_txt = prompt_builder_with_query_only(d, i=0, prompt_txt=[])
    responses = []
    for i in range(1, iter_n+1):
        response = prompt_request(prompt_txt)
        generated_nls = parse_gpt3_response(prompt_txt, response, logger)
        generated_nl = generated_nls[0].strip().replace('\n', ' ')
        responses.append(generated_nl)
        prompt_txt = prompt_builder_with_query_only(
            d, i, prompt_txt=prompt_txt+generated_nl)
    logger.info('\n'.join(prompt_txt.split('\n')[:-1]))
    return responses


def set_logger_output(logger, log_file, fmt='%(asctime)s - %(name)s - %(levelname)s\n%(message)s'):
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    # remove all existed FileHandler
    for hdlr in logger.handlers[:]:
        if isinstance(hdlr, logging.FileHandler):
            logger.removeHandler(hdlr)
    # generate folders if not existed
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    file_h = logging.FileHandler(log_file, mode='w')
    logger.addHandler(file_h)
    formatter = logging.Formatter(fmt)
    file_h.setFormatter(formatter)


def init_logger(logger):
    logger.setLevel(logging.DEBUG)


def generate_queries(data, dataset_name):
    res = []
    dir_path = os.path.dirname(__file__)
    logger_name = 'gpt-3-prompt'
    logger = logging.getLogger(logger_name)
    init_logger(logger)
    cnt = 0
    for i, d in enumerate(data):
        log_file = str(i+1)+'.txt'
        log_file = os.path.join(dir_path, 'log', dataset_name, log_file)
        set_logger_output(logger, log_file)
        prompt_txt = prompt_builder(d)
        response = prompt_request(prompt_txt)
        generated_nls = parse_gpt3_response(prompt_txt, response, logger, verbose=True)
        original_sql = d['original_query']
        original_nl = d['question']
        generated_sql = d['generated_query']
        db_id = d['db_id']
        for generated_nl in generated_nls:
            temp = {
                'db_id': db_id,
                'original_id': i,
                'original_query': original_sql,
                'original_nl': original_nl,
                'id': cnt,
                'query': generated_sql,
                'nl': generated_nl
            }
            cnt += 1
            res.append(temp)
    return res


def generate_multiple_questions(data, dataset_name, dir_path, iter_n):
    res = []
    for i, d in enumerate(data):
        logger_name = 'gpt-3-prompt'
        logger = logging.getLogger(logger_name)
        init_logger(logger)
        log_file = str(i)+'.txt'
        dataset_name = 'cordis'
        log_file = os.path.join(dir_path, 'log', dataset_name, log_file)
        set_logger_output(logger, log_file)
        responses = gpt3_iter_responses(d, logger, iter_n)
        db_id = d['db_id']
        temp = {}
        temp['db_id'] = db_id
        temp['query'] = d['query']
        temp['id'] = i
        temp['hops'] = d['hops']
        temp['questions'] = responses
        res.append(temp)
    return res

# for spider
def run_gpt3_prompter_with_original(dataset_name, json_file, output_file):
    
    args = parser.parse_args()
    dir_path = os.path.dirname(__file__)
    par_path = os.path.dirname(dir_path)
    root_path = os.path.dirname(par_path)
    data_path = os.path.join(root_path, 'data', dataset_name)
    json_file = os.path.join(data_path, json_file)
    with open(json_file, 'r') as f_in:
        data = json.load(f_in)
    output = generate_queries(data, dateset_name)
    output_file = ''
    output_file = os.path.join(par_path, output_file)
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f_out:
            json.dump(output, f_out, indent=2, sort_keys=True)


# for cordis
def run_gpt3_prompter_without_original(dataset_name, json_file, output_file):
    
    
    dir_path = os.path.dirname(__file__)
    par_path = os.path.dirname(dir_path)
    root_path = os.path.dirname(par_path)
    data_path = os.path.join(root_path, 'data', dataset_name)
    json_file = os.path.join(data_path, json_file)
    with open(json_file, 'r') as f_in:
        data = json.load(f_in)
    # data = data[:2]
    output = generate_multiple_questions(
        data, dataset_name=dateset_name, dir_path=dir_path, iter_n=4)
    output_file = os.path.join(root_path, output_file)
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f_out:
            json.dump(output, f_out, indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = load_parser()
    args = parser.parse_args()
    if args.query_only:
        run_gpt3_prompter_without_original(dataset_name=args.dataset_name, json_file=args.output_file, output_file=args.output_file)
    else:
        run_gpt3_prompter_with_original(dataset_name=args.dataset_name, json_file=args.output_file, output_file=args.output_file)
    