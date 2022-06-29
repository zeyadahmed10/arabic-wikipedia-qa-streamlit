import logging
import os
import re
from functools import lru_cache
from urllib.parse import unquote

import streamlit as st
import wikipedia
from codetiming import Timer
from fuzzysearch import find_near_matches
from googleapi import google
from transformers import AutoTokenizer, pipeline

from annotator import annotated_text
from preprocess import ArabertPreprocessor

logger = logging.getLogger(__name__)

wikipedia.set_lang("ar")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

preprocessor = ArabertPreprocessor("wissamantoun/araelectra-base-artydiqa")
logger.info("Loading Pipeline...")
tokenizer = AutoTokenizer.from_pretrained("ZeyadAhmed/AraElectra-ASQuADv2-CLS", do_lower_case = False)
qa_pipe = pipeline("question-answering", model="ZeyadAhmed/AraElectra-ASQuADv2-QA")
tokenizer_kwargs = {'truncation':True,'max_length':512}
cls_pipe = pipeline('text-classification', model="ZeyadAhmed/AraElectra-ASQuADv2-CLS")
#cls_pip = pipeline("")
logger.info("Finished loading Pipeline...")
def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
def find_unanswered_questions(cls_result, threshold):
    indcies = list()
    for i in range(len(cls_result)):
        conf = None
        if cls_result[i]['label'] == 'LABEL_0':
            conf = 1-cls_result[i]['conf']
            cls_result[i]['conf'] = conf 
        else:
            conf = cls_result[i]['conf']
        print(conf)
        if conf >threshold:
            indcies.append(i)
    return indcies
def concatenate_dict(results, cls_results):
    for i in range(len(cls_results)):
        conf = cls_results[i]['score']
        del cls_results[i]['score']
        cls_results[i]['conf'] = conf
        results[i].update(cls_results[i])
    return results      
@lru_cache(maxsize=100)
def get_results(question):
    logger.info("\n=================================================================")
    logger.info(f"Question: {question}")

    if "زياد احمد" in question or "زياد أحمد" in question or "zeyad ahmed" in question.lower():
        return {
            "title": "Creator",
            "results": [
                {
                    "score": 1.0,
                    "new_start": 0,
                    "new_end": 12,
                    "new_answer": "My Creator 😜",
                    "original": "My Creator 😜",
                    "link": "https://www.linkedin.com/in/zeyadahmed1/",
                }
            ],
        }
    search_timer = Timer(
        "search and wiki", text="Search and Wikipedia Time: {:.2f}", logger=logging.info
    )
    try:
        search_timer.start()
        search_results = google.search(
            question + " site:ar.wikipedia.org", lang="ar", area="ar"
        )
        if len(search_results) == 0:
            return {}

        page_name = search_results[0].link.split("wiki/")[-1]
        wiki_page = wikipedia.page(unquote(page_name))
        wiki_page_content = wiki_page.content
        search_timer.stop()
    except:
        return {}

    sections = []
    for section in re.split("== .+ ==[^=]", wiki_page_content):
        if not section.isspace():
            prep_section = tokenizer.tokenize(preprocessor.preprocess(section))
            if len(prep_section) > 500:
                subsections = []
                for subsection in re.split("=== .+ ===", section):
                    if subsection.isspace():
                        continue
                    prep_subsection = tokenizer.tokenize(
                        preprocessor.preprocess(subsection)
                    )
                    subsections.append(subsection)
                    # logger.info(f"Subsection found with length: {len(prep_subsection)}")
                sections.extend(subsections)
            else:
                # logger.info(f"Regular Section with length: {len(prep_section)}")
                sections.append(section)

    full_len_sections = []
    temp_section = ""
    for section in sections:
        if (
            len(tokenizer.tokenize(preprocessor.preprocess(temp_section)))
            + len(tokenizer.tokenize(preprocessor.preprocess(section)))
            > 384
        ):
            if temp_section == "":
                temp_section = section
                continue
            full_len_sections.append(temp_section)
            # logger.info(
            #     f"full section length: {len(tokenizer.tokenize(preprocessor.preprocess(temp_section)))}"
            # )
            temp_section = ""
        else:
            temp_section += " " + section + " "
    if temp_section != "":
        full_len_sections.append(temp_section)

    reader_time = Timer("electra", text="Reader Time: {:.2f}", logger=logging.info)
    reader_time.start()
    questions=[preprocessor.preprocess(question)] * len(full_len_sections)
    contexts=[preprocessor.preprocess(x) for x in full_len_sections]
    results = qa_pipe(
        question = questions,
        context = contexts
    )
    cls_results = cls_pipe(
        [{'text':x, 'text_pair':y} for x, y in zip(questions, contexts)],**tokenizer_kwargs
    )
    if not isinstance(cls_results, list):
        cls_results = [cls_results]
    if not isinstance(results, list):
        results = [results]

    logger.info(f"Wiki Title: {unquote(page_name)}")
    logger.info(f"Total Sections: {len(sections)}")
    logger.info(f"Total Full Sections: {len(full_len_sections)}")

    for result, section in zip(results, full_len_sections):
        result["original"] = section
        answer_match = find_near_matches(
            " " + preprocessor.unpreprocess(result["answer"]) + " ",
            result["original"],
            max_l_dist=min(5, len(preprocessor.unpreprocess(result["answer"])) // 2),
            max_deletions=0,
        )
        try:
            result["new_start"] = answer_match[0].start
            result["new_end"] = answer_match[0].end
            result["new_answer"] = answer_match[0].matched
            result["link"] = (
                search_results[0].link + "#:~:text=" + result["new_answer"].strip()
            )
        except:
            result["new_start"] = result["start"]
            result["new_end"] = result["end"]
            result["new_answer"] = result["answer"]
            result["original"] = preprocessor.preprocess(result["original"])
            result["link"] = search_results[0].link
        logger.info(f"Answers: {preprocessor.preprocess(result['new_answer'])}")
    results = concatenate_dict(results, cls_results)
    print(len(results))
    indcies = find_unanswered_questions(results, 0.5)
    delete_multiple_element(results, indcies)
    print(len(results),len(indcies))
    print('alo')
    sorted_results = sorted(results, reverse=True, key=lambda x: x["score"])

    return_dict = {}
    return_dict["title"] = unquote(page_name)
    return_dict["results"] = sorted_results

    reader_time.stop()
    logger.info(f"Total time spent: {reader_time.last + search_timer.last}")
    return return_dict
def splitter(question, text, tokenizer, split_size, overlap_size):
  samples = []
  start = 0
  text_splited = text.split(" ")
  print(len(tokenizer(text)['input_ids']))
  if len(tokenizer(question, text)['input_ids'])<384:
    return [text]
  while(start< len(text_splited)):
    curr_section  = " ".join(text_splited[start:start+split_size])
    end = start+split_size
    while(len(tokenizer(question, curr_section)['input_ids'])>384):
        end = end -10
        curr_section  = " ".join(text_splited[start:end])
        flag = True
    if flag == True:
        start = end-overlap_size
    else:
        start = start +split_size - overlap_size
    samples.append(curr_section)
  return samples  
@lru_cache(100)
def get_offline_results(question, doc):
    if "زياد احمد" in question or "زياد أحمد" in question or "zeyad ahmed" in question.lower():
        return {
            "title": "Creator",
            "results": [
                {
                    "score": 1.0,
                    "new_start": 0,
                    "new_end": 12,
                    "new_answer": "My Creator 😜",
                    "original": "My Creator 😜",
                    "link": "https://www.linkedin.com/in/zeyadahmed1/",
                }
            ],
        }
    
    max_length = 384 # The maximum length of a feature (question and context)
    doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
    samples = splitter(question, doc,tokenizer, max_length, doc_stride)
    questions=[preprocessor.preprocess(question)] * len(samples)
    contexts=[preprocessor.preprocess(x) for x in samples]
    results = qa_pipe(
        question = questions,
        context = contexts
    )
    cls_results = cls_pipe(
        [{'text':x, 'text_pair':y} for x, y in zip(questions, contexts)],**tokenizer_kwargs
    )
    if not isinstance(cls_results, list):
        cls_results = [cls_results]
    if not isinstance(results, list):
        results = [results]

    for result, section in zip(results, samples):
        result["original"] = section
        answer_match = find_near_matches(
            " " + preprocessor.unpreprocess(result["answer"]) + " ",
            result["original"],
            max_l_dist=min(5, len(preprocessor.unpreprocess(result["answer"])) // 2),
            max_deletions=0,
        )
        try:
            result["new_start"] = answer_match[0].start
            result["new_end"] = answer_match[0].end
            result["new_answer"] = answer_match[0].matched
        except:
            result["new_start"] = result["start"]
            result["new_end"] = result["end"]
            result["new_answer"] = result["answer"]
            result["original"] = preprocessor.preprocess(result["original"])
        logger.info(f"Answers: {preprocessor.preprocess(result['new_answer'])}")
    results = concatenate_dict(results, cls_results)
    indcies = find_unanswered_questions(results, 0.5)
    delete_multiple_element(results, indcies)
    sorted_results = sorted(results, reverse=True, key=lambda x: x["score"])

    return_dict = {}
    return_dict["results"] = sorted_results
    return return_dict

def shorten_text(text, n, reverse=False):
    if text.isspace() or text == "":
        return text
    if reverse:
        text = text[::-1]
    words = iter(text.split())
    lines, current = [], next(words)
    for word in words:
        if len(current) + 1 + len(word) > n:
            break
            lines.append(current)
            current = word
        else:
            current += " " + word
    lines.append(current)
    if reverse:
        return lines[0][::-1]
    return lines[0]


def annotate_answer(result):
    annotated_text(
        shorten_text(
            result["original"][: result["new_start"]],
            500,
            reverse=True,
        ),
        (result["new_answer"], "جواب", "#8ef"),
        shorten_text(result["original"][result["new_end"] :], 500) + " ...... إلخ",
    )


if __name__ == "__main__":
    results_dict = get_results("ما هو نظام لبنان؟")
    for result in results_dict["results"]:
        annotate_answer(result)
        f"[**المصدر**](<{result['link']}>)"
