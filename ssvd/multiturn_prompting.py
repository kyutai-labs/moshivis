"""Main pipeline for generating dialogues"""

import json
from copy import copy
from random import random
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import rich
import torch
from multiturn_instruct import MTCInstruct
from transformers import Pipeline
from utils import (
    compile_pattern,
    get_replace_pattern,
    get_strings_for_logging,
    maybe_shorten_caption,
)


def list_to_prompt(
    convo_list: List[str],
    img_caption: str,
    pipe: Pipeline,
    setting: str,
) -> List[Dict]:
    """
    Converts a conversation list into a prompt for chat-based language models.

    :param convo_list: A list of strings representing the conversation.
    :param img_caption: The caption for the image associated with the conversation.

    :return: A list of dictionaries representing the chat prompt, where each dictionary
    contains the role (speaker) and content of a message in the conversation.

    Example:
        convo_list = ["Hello!", "How are you?", "I'm good, thanks!"]
        img_caption = "A beautiful sunset"
        prompt = list_to_prompt(convo_list, img_caption)
        print(prompt)
        # Output: [{'role': 'system', 'content': 'A beautiful sunset'},
        #          {'role': 'user', 'content': 'Speaker 2:\nHello!'},
        #          {'role': 'assistant', 'content': 'Speaker 1:\nHow are you?'},
        #          {'role': 'user', 'content': 'Speaker 2:\nI'm good, thanks!'}]
    """
    try:
        setting_obj = MTCInstruct(setting)
        system_template, speaker1_template, speaker2_template, start_conv = (
            setting_obj.get_method(len(convo_list))()
        )
    except ValueError as e:
        raise NotImplementedError("Unknown MTCInstruct setting", setting) from e

    convo_list = copy(convo_list)
    if len(convo_list) % 2 == 0:
        convo_list = [
            system_template.format(
                ROLE_SPECIFIC_TEXT=speaker1_template.format(caption=img_caption),
                caption=img_caption,
            ),
            start_conv,
        ] + convo_list
    else:
        convo_list = [
            system_template.format(
                ROLE_SPECIFIC_TEXT=speaker2_template.format(caption=img_caption),
                caption=img_caption,
            )
        ] + convo_list

    def speaker_iter() -> Iterator:
        yield "system"
        while True:
            yield "user"
            yield "assistant"

    def prefix_iter() -> Iterator:
        yield ""
        while True:
            yield "Question: "
            yield "Answer: "

    chat = [
        {"role": speaker, "content": prefix + c}
        for c, speaker, prefix in zip(convo_list, speaker_iter(), prefix_iter())
    ]
    tok = pipe.tokenizer
    return tok.apply_chat_template(
        chat,
        tokenize=False,
        continue_final_message=False,
    )


def postprocess_mtc(
    s: str,
    drop_probs: Optional[Dict[str, Dict]] = None,
    default_prob: float = 0.8,
    setting: Optional[str] = None,
) -> str:
    """Post-process to remove some unwanted patterns:
    - remove expression referring to the image caption/description
    - remove references to the LLM role
    - reduce probability of very common LLM phrases e.g. "it's quite striking, isn't it ?"
    """
    pattern = get_replace_pattern()
    s = pattern.sub("", s)
    if drop_probs is None:
        drop_probs = {
            r"Wow,\s": dict(p=default_prob, replace_by=""),
            r", isn't it[?]": dict(p=default_prob, replace_by="."),
            r"Well,\s": dict(p=default_prob, replace_by=""),
            r"quite striking": dict(p=0.5, replace_by="impressive"),
            r"quite": dict(p=0.3, replace_by=""),
            r"I'm not (entirely )?sure( about that)?\.": dict(
                p=default_prob, replace_by=""
            ),
            # hardcoded replacement
            r"Teacher:": dict(p=1.0, replace_by=""),
            r"Assistant": dict(p=1.0, replace_by=""),
            r"You:": dict(p=1.0, replace_by=""),
            r"Teacher :": dict(p=1.0, replace_by=""),
            r"You :": dict(p=1.0, replace_by=""),
            r"Speaker1": dict(p=1.0, replace_by=""),
            r"Speaker2": dict(p=1.0, replace_by=""),
            r"Speaker ": dict(p=1.0, replace_by=""),
            r"image description": dict(p=1.0, replace_by="image"),
            r"he image doesn't specify": dict(
                p=1.0, replace_by="he image doesn't depict"
            ),
            r"doesn't mention": dict(p=1.0, replace_by="doesn't show"),
            r"not mention": dict(p=1.0, replace_by="not depict"),
            r"mentions": dict(p=1.0, replace_by="depicts"),
            r"mentioned": dict(p=1.0, replace_by="visible"),
            r"no mention": dict(p=1.0, replace_by="no sign"),
            r"No mention": dict(p=1.0, replace_by="No sign"),
            r"described": dict(p=1.0, replace_by="shown"),
            r"describing": dict(p=1.0, replace_by="showing"),
            r"isn't specified": dict(p=1.0, replace_by="isn't visible"),
        }
    if setting is not None and setting not in {"cap", "cap2", "rnd"}:
        drop_probs[r"description"] = dict(p=1.0, replace_by="image")
    for drop_s, drop_kwargs in drop_probs.items():
        pattern = compile_pattern(drop_s)
        p = drop_kwargs["p"]
        r = drop_kwargs["replace_by"]
        if random() < p:
            s = pattern.sub(r, s).strip()
            try:
                if drop_s[0].isupper():
                    s = s[0].upper() + s[1:]
            except IndexError:
                pass
    s = s.strip()
    if not s.startswith('"'):
        s = '"' + s
    if not s.endswith('"'):
        s += '"'
    return s


class ConvoIter:
    """Conversation builder"""

    def __init__(
        self,
        convo_length: int = 4,
        batch_size: int = 64,
        pipe: Optional[Pipeline] = None,
        setting: str = "mtc",
    ) -> None:
        """Init object to store the ongoing conversation"""
        self.convos: Dict[str, List[str]] = {}
        self.convo_length = convo_length
        self.batch_size = batch_size
        self.pipe = pipe
        self.setting = setting
        self.last_updated: Optional[List[str]] = None

    def add_to_convos(self, uid: str, answer: str) -> None:
        """Add next turn to the dialogue for the image `uid`"""
        if not uid in self.convos:
            self.convos[uid] = []
        self.convos[uid].append(answer)
        self.last_updated = self.convos[uid]

    def make_iter(self, captions: Sequence[str], img_ids: Sequence[str]) -> Iterator:
        """Main iterator for the dialogue"""
        convo_ids_within_loop = []
        captions_within_loop = []
        for count, (uid, img_caption) in enumerate(zip(img_ids, captions)):
            img_caption = maybe_shorten_caption(img_caption, max_cap_len=1000)
            convo_ids_within_loop.append(uid)
            captions_within_loop.append(img_caption)
            return_value = list_to_prompt(
                convo_list=[],
                img_caption=img_caption,
                pipe=self.pipe,
                setting=self.setting,
            )
            yield return_value
            if ((count + 1) % self.batch_size) == 0:
                for _ in range(self.convo_length - 1):
                    for uid, img_caption in zip(
                        convo_ids_within_loop, captions_within_loop
                    ):
                        return_value = list_to_prompt(
                            self.convos[uid],
                            img_caption=img_caption,
                            pipe=self.pipe,
                            setting=self.setting,
                        )
                        yield return_value

                convo_ids_within_loop = []
                captions_within_loop = []


@torch.no_grad()
def run_multiturn_pipeline(
    pipe: Pipeline,
    captions: Sequence[str],
    img_ids: Sequence[str],
    out_file: str,
    batch_size: int = 64,
    convo_length: int = 6,
    setting: str = "mtc",
    temperature: float = 0.0,
    max_new_tokens: int = 150,
) -> None:
    """Main pipeline for generating multi-turn conversations (back and forth between LLMs)

    :param pipe: transformers pipeline
    :param captions: List of captions
    :param img_ids: List of associated image ids
    :param out_file: Output files to dump the captions in
    :param batch_size: Batch size
    :param setting: Which MTCInstruct to use
    :param temperature: Sampling temperature for the pipeline
    :param max_new_tokens: Maximum number of tokens per turn
    """
    assert len(captions) == len(img_ids)
    count = 0

    def uid_iter(img_ids: Sequence[str]) -> Iterator:
        """UID iter with groups of size `batch_size`"""
        nonlocal convo_length
        ids = np.array(
            list(img_ids) + [None] * (batch_size - len(img_ids) % batch_size)
        ).reshape(-1, batch_size)

        for batch_ids in ids:
            for _ in range(convo_length):
                yield from batch_ids

    convo_iter = ConvoIter(
        convo_length=convo_length, batch_size=batch_size, pipe=pipe, setting=setting
    )
    data_iter = convo_iter.make_iter(captions, img_ids)
    total = len(captions) * convo_length
    try:
        for uid, out in zip(
            uid_iter(img_ids),
            pipe(
                data_iter,
                max_new_tokens=max_new_tokens,
                return_full_text=False,
                add_special_tokens=False,
                batch_size=batch_size,
                do_sample=temperature > 0,
                temperature=temperature,
            ),
        ):
            answer = postprocess_mtc(out[0]["generated_text"], setting=setting)
            convo_iter.add_to_convos(uid=uid, answer=answer)
            count += 1
            if (count % (batch_size * convo_length)) == 0:
                try:
                    assert convo_iter.last_updated is not None
                    q, a = get_strings_for_logging(
                        [
                            dict(
                                zip(
                                    ["question", "answer"], convo_iter.last_updated[-2:]
                                )
                            )
                        ]
                    )
                    print(
                        f"{count+1:>8d}/{total:8d} ({100*(count+1)/total:6.2f}%)\tQ: {q} \tA: {a}",
                        flush=True,
                    )

                except Exception as e:  # pylint: disable=W0718
                    rich.print(
                        "[red]WARNING:[/red] Something went wrong when reading the result.",
                        flush=True,
                    )
                    print(f"Result: {convo_iter.convos[uid]}", flush=True)
                    print(e, flush=True)

    except Exception as e:  # pylint: disable = W0718
        rich.print(
            "[red]WARNING:[/red] Something went wrong when running the pipeline."
            " Saving existing results and then terminating.",
            flush=True,
        )
        print(e, flush=True)

    print(flush=True)
    with open(out_file, "w") as f:
        for uid, res in convo_iter.convos.items():
            json.dump({"uid": uid, "res": res}, f)
            f.write("\n")
