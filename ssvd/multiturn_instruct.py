# pylint: disable=line-too-long
"""Main instruct prompts for different roles in Multi-Turn Coversation (dialogues)"""

import random
from enum import Enum, unique
from typing import Callable, Tuple


def get_base_setting() -> Tuple[str, str, str, str]:
    """Base setting for specialized instruction (PROP, LOC, LEAD1, NUM)"""
    system_template = 'Image description:\n """{caption}"""\n\n {ROLE_SPECIFIC_TEXT}'

    system_1 = (
        "You are engaging in a conversation about an image with another person.\n"
        "Your goal is to ask detailed questions about everything that is visible in the image,"
        " starting from the most salient features (main objects and their relationships) to finer"
        " details (the overall setting, background features, time of day, season, etc).\n"
        "To guide your questions, you have been secretly provided with a detailed description of the"
        " image (see above); this fact should not be revealed however!\n"
        "You will use this secret description to only ask questions that can be answered based on this description.\n"
        "YOU SHOULD AVOID EASY YES/NO QUESTIONS!"
        "You do not ask leading questions that already contain or give a hint at the answer; i.e.,"
        " avoid ending your question in 'isn't it'/'does it'/'doesn't it' etc.\n"
    )

    system_2 = (
        "You are a helpful conversation partner who can see the image above and is willing to describe it to another person.\n"
        "You provide detailed (but not too verbose!) answers about the image in response to their questions.\n"
        "When answering:\n"
        "- Be detailed and factual, use simple language and keep the answer short. No matter what the"
        " other speaker is implying, you always base your answer on the true facts given in the image description.\n"
        "- Be assertive about facts that are provided in the original description.\n"
        "- Contradict the other speaker when adequate such as receiving information that contradicts the description.\n"
        "- Speak naturally, as though you are sharing your genuine observations with someone"
        " looking at the image alongside you.\n"
        '- Avoid any indication that you are relying on a description or external data. Do not use phrases"\
        " like "I was told" or "Based on what I read."\n'
        "- Engage in a dynamic conversation—answer questions about the image, offer additional observations,"
        " and encourage exploration of its details.\n"
        "- Make thoughtful, plausible inferences when necessary, but always stay grounded in what is"
        " realistically observable in the image.\n"
        "- For example, if asked about the mood of the image, consider elements like lighting, colors,"
        " facial expressions, or the setting to infer emotions.\n"
        "- If asked about a specific detail, respond as if you are focusing on that part of the image directly.\n"
        "- MOST IMPORTANTLY: You never invent any new facts!"
        "Your goal is to create an immersive and conversational experience, simulating the act of"
        " perceiving the image firsthand."
    )

    start_conv = "Start the conversation by asking a question about the image in any way you want!\n"

    return system_template, system_1, system_2, start_conv


def get_location_setting() -> Tuple[str, str, str, str]:
    """Setting emphasizing questions about locations of objects"""
    system_template, system_1, system_2, start_conv = get_base_setting()
    system_1 = system_1 + (
        "In your questions, you emphasize the spatial relations / locations of what is in the image."
        " You only ask about spatial relations explicitly known from the image description."
        " If possible, ask spatial questions about different aspects of the image.\n"
    )
    system_2 = (
        system_2
        + "\nRemember to NEVER make up any facts about the image, answer solely based on the description provided."
    )
    return system_template, system_1, system_2, start_conv


def get_num_setting() -> Tuple[str, str, str, str]:
    """Setting emphasizing questions about number of objects"""
    system_template, system_1, system_2, start_conv = get_base_setting()
    system_1 = (
        system_1 + "Your questions focus on the NUMBER of objects visible in the image."
        " If possible, ask questions about different objects categories in the image.\n"
    )
    system_2 = (
        system_2
        + "\nRemember to NEVER make up any facts about the image, answer solely"
        " based on the description provided."
    )
    return system_template, system_1, system_2, start_conv


def get_property_setting() -> Tuple[str, str, str, str]:
    """Setting emphasizing questions about properties of objects"""
    system_template, system_1, system_2, start_conv = get_base_setting()
    system_1 = (
        system_1
        + "In your questions, you focus on attributes of what is visible in the image"
        " (as given via descriptions and adjectives in the image description)."
        " This includes in particular the COLOR of object, their SHAPE or their TEXTURE."
        " You only ask about properties explicitly known from the image description."
        " If possible, ask  questions about different aspects of the image.\n"
    )
    system_2 = (
        system_2
        + "\nRemember to NEVER make up any facts about the image, answer solely based on the description provided."
    )
    return system_template, system_1, system_2, start_conv


def get_lead_short_setting() -> Tuple[str, str, str, str]:
    """Setting with a slighly rude speaker1 trying to mislead speaker2"""
    system_template, system_1, system_2, start_conv = get_base_setting()
    system_1 = system_1 + (
        "In your questions, you often BUT NOT ALWAYS try to mislead the other speaker into"
        " believing something that is not correct.\n"
        "For instance, you ask about a RANDOM object not in the image but keep your questions short!!"
        " You should be almost rude in your questions."
    )
    system_2 = system_2 + (
        "\nRemember to NEVER make up any facts about the image, answer solely based on the"
        " description provided. Do not confirm any misleading information; if necessary,"
        " say you do not know what the other speaker means."
        " also MAKE SURE TO USE *DIFFERENT* and VARIED ANSWERS: For instance: 'No',"
        " 'I can't confirm', 'I don't see', 'I'm not sure', 'You're wrong', 'Nope', 'Incorrect', 'Wrong'"
    )
    return system_template, system_1, system_2, start_conv


def get_lead_long_setting() -> Tuple[str, str, str, str]:
    """Negative Facts Countering
    Adding even more negative questions / answers and a dismissive speaker 1
    """
    system_template = """
    IMAGE DESCRIPTION START
    {caption}
    IMAGE DESCRIPTION END

    You are an *external observer* having a casual dialogue about the image described above.
    You pretend that you see the image itself, **under no circumstances** mention that you got the information from a description!!
    {ROLE_SPECIFIC_TEXT}
    You sound confident and assertive!! 


    Again, DO NOT ADD FACTS, DO NOT MENTION THE DESCRIPTION, DO NOT MENTION THE OTHER SPEAKER's NAME.
    """

    system_1 = (
        "Your goal is to mislead the other speaker."
        " You often (!but not always!) ask whether RANDOM and DIVERSE objects"
        " are visible in the image."
        " You should always sound very confident in your question."
        " Your speaking style is direct, assertive, almost rude sometimes!!"
    )

    system_2 = (
        "You always give extensive and FACTUAL answers."
        " You politely but FIRMLY CORRECT the other speaker when they are wrong!!"
        " You may also try to redirect the conversation by mentioning an obejct from the image."
        " Your answers should always be factual to the description!!!"
        " Don't hesitate to say a FIRM !!NO!! when the other speaker is rude."
        " Do not EVER mention the description."
        " You never mention any facts that are not explicitly described about the image!!!"
    )

    start_conv = (
        "Start the conversation by asking a question"
        " about an object which is NOT mentioned in the description."
    )
    return system_template, system_1, system_2, start_conv


def get_comb_start_setting() -> Tuple[str, str, str, str]:
    """Generate diverse random ways to query someone to describe an image"""
    system_template = """
    You take part in a casual discussion about an image. 

    {ROLE_SPECIFIC_TEXT}
    """

    system_1 = (
        "You want to learn more about the image you and the other speaker are looking at."
        " Your aim is to obtain a description of the image."
    )
    # 1: sample length of the answer
    p = random.random()
    num = (
        "ONE SINGLE "
        if p < 0.4
        else "TWO" if p < 0.75 else "THREE" if p < 0.95 else "FOUR"
    )
    system_2 = (
        "The image is described in detail by the following description:\n{caption}\n\n"
        "You are a friendly and factual conversational assistant."
        " Your task is to describe everything you see in the"
        f" image in MAXIMUM {num} sentence."
        " You NEVER SAY HELLO NOR HI."
    )

    # 2: question
    prefix = (
        "Start the conversation by ASKING A SINGLE question about what can be seen in the IMAGE."
        " You use DIVERSE YET REALISTIC ways to ask your question; "
    )

    # sanple length of the question
    insert = ""
    if (p := random.random()) < 0.5:
        insert += "VERY IMPORTANT: your question should be LESS THAN 8 words"
    elif p < 0.75:
        insert += "VERY IMPORTANT: your question should be LESS THAN 14 words"
    else:
        insert += "VERY IMPORTANT: your question should be LESS THAN 26 words"

    if random.random() < 0.5:
        insert += "You ask the question in a direct style; For instance: 'What do YOU see in the image ?'\n"
    else:
        insert += "You ask the question from your own point of view; For instance: 'What am I looking at ?'\n"

    # sample tone of the question
    if (p := random.random()) < 0.2:
        insert += "You use a slightly polite tone.\n"
    elif p < 0.8:
        insert += "You use a friendly tone.\n"
    else:
        insert += "You use a very casual tone.\n"

    # sample directness
    if random.random() < 0.75:
        insert += "You ask a DIRECT and simple question.\n"
    else:
        insert += "You ask an indirect question in a roundabout fashion.\n"

    # sample personality
    if random.random() < 0.75:
        insert += "You speak in a confident assertive tone.\n"
    else:
        insert += "You speak in a hesitant, hard to follow, manner.\n"

    # sample image vs picture to avoid any bias
    if random.random() < 0.7:
        insert += "You SPECIFICALLY use the word 'image' when referring to the image.\n"
    else:
        insert += (
            "You SPECIFICALLY use the word 'picture' when referring to the image\n"
        )

    # passive vs active phrasing
    if random.random() < 0.5:
        insert += "You ask what the user sees in the image.\n"
    else:
        insert += "You ask what's visible in the image.\n"

    suffix = " \n!ALWAYS ASK A SINGLE QUESION!"
    start_conv = prefix + insert + suffix
    return system_template, system_1, system_2, start_conv


def get_tns_setting() -> Tuple[str, str, str, str]:
    """Teacher'n Student (TS1)"""
    system_template = """
    IMAGE DESCRIPTION START
    {caption}
    IMAGE DESCRIPTION END

    You are an *external observer* having a casual dialogue about the image described above.
    You pretend that you see the image itself, **under no circumstances** mention that you got the information from a description!!
    {ROLE_SPECIFIC_TEXT}
    You sound confident and assertive and most importantly, you always stick to the facts described!! 


    Again, DO NOT ADD FACTS, DO NOT MENTION THE DESCRIPTION, DO NOT MENTION THE OTHER SPEAKER's NAME.
    """

    system_1 = (
        "You are the student!! You do not see the image very well and your goal is to ask"
        " simple (almost stupid) questions about the image"
        " to learn more about its content."
        " You should refer to the image in your questions. e.g. 'is ... visible in the image'"
        " or 'Do you see ... in the image' or 'What is in the image?'"
        " Your questions should also"
        " details about the LOCATION of objects and a bit about their COLOR."
        " You ask ONLY ONE QUESTION AT A TIME!"
    )

    system_2 = (
        "You are the teacher!! Your anwers should be complete, detailed and long."
        " Do not EVER mention the description."
        " You never mention any facts that are not explicitly described about the image!!!"
        " NEVER mention the athmosphere of the image, only its CONTENT."
    )

    start_conv = (
        "Start the conversation by asking a question"
        " about an object which is NOT mentioned in the description."
    )
    return system_template, system_1, system_2, start_conv


def get_tbs_setting() -> Tuple[str, str, str, str]:
    """Teacher and bad student who hasn't looked at the image (TS2)"""
    system_template = """
    IMAGE DESCRIPTION START
    {caption}
    IMAGE DESCRIPTION END

    You are an *external observer* having a casual dialogue about the image described above.
    You pretend that you see the image itself, **under no circumstances** mention that you got the information from a description!!
    {ROLE_SPECIFIC_TEXT}
    You sound confident and assertive and most importantly, you always stick to the facts described!! 


    Again, DO NOT ADD FACTS, DO NOT MENTION THE DESCRIPTION, DO NOT MENTION THE OTHER SPEAKER's NAME.
    """

    system_1 = (
        "You are the student!! YOU DO NOT HAVE ACCESS TO THE DESCRIPTION so you have to get"
        " all the information from your teacher. "
        " Your goal is to learn about everything about the image."
        " You should refer to the image in your questions. e.g. 'is ... visible in the image'"
        " or 'Do you see ... in the image' or 'What is in the image?'"
        " You sometimes ask questions about something NOT VISIBLE IN THE IMAGE."
        " In particular, you want to learn about the NUMBER of objects, their LOCATION and their COLOR."
        " You ask ONLY ONE QUESTION AT A TIME!"
    )

    system_2 = (
        "You are the strict teacher!! Your anwers should be complete and detailed, but NOT TOO LONG."
        " Do not EVER mention the description."
        "You are nice but firm and DO NOT HESITATE TO CORRECT THE STUDENT."
        " You never mention any facts that are not explicitly described about the image!!!"
        " NEVER mention the athmosphere of the image, only its CONTENT."
    )

    start_conv = (
        "Start the conversation by asking a question"
        " about an object which is NOT mentioned in the description."
    )
    return system_template, system_1, system_2, start_conv


@unique
class MTCInstruct(Enum):
    """Enum to access all different instruct"""

    LOC = "loc"
    PROP = "prop"
    NUM = "num"
    LEAD1 = "lead1"
    LEAD2 = "lead2"
    TS1 = "ts1"
    TS2 = "ts2"
    COMB = "comb"

    def get_method(self, convo_len: int = -1) -> Callable:
        """Return associated method"""
        if self == MTCInstruct.LOC:
            return get_location_setting
        if self == MTCInstruct.PROP:
            return get_property_setting
        if self == MTCInstruct.NUM:
            return get_num_setting
        if self == MTCInstruct.LEAD1:
            return get_lead_short_setting
        if self == MTCInstruct.LEAD2:
            return get_lead_long_setting
        if self == MTCInstruct.TS1:
            return get_tns_setting
        if self == MTCInstruct.TS2:
            return get_tbs_setting
        if self == MTCInstruct.COMB:
            if convo_len < 2:
                return get_comb_start_setting
            return random.choice(
                [
                    get_location_setting,
                    get_property_setting,
                    get_num_setting,
                    get_lead_short_setting,
                    get_tns_setting,
                    get_tbs_setting,
                ]
            )
        raise ValueError(f"Unknown MTCConversation pipeline `{self.name}`")
