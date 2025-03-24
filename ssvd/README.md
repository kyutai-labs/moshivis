# Synthetic visual dialogues pipeline

This directory contains the data generation pipeline for synthetic visual dialogues described in Section 3.3 and Appendix E of [our preprint](https://arxiv.org/abs/2503.15633).

We generate synthetic visual dialogues in text-only form with a variety of instruction prompting, using an image captioning dataset as source of information (here, DOCCI, PixMo or PixelProse). We use two LLMs (here, both based on Mistral-NeMo) to represent the user and assistant respectively, each taking a turn in the conversation one after the other, and characterized by their respective instruct prompt.


For more information, please refer to Section 3.3 and Appendix E of [our preprint](https://arxiv.org/abs/2503.15633).


### Quickstart

As for the rest of the codebase, we recommend using [uv](https://docs.astral.sh/uv/) to setup and run the code,
as it will manage all dependencies for you transparently. `uv` is provided as a lightweight binary and can be installed as:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To generate a single visual dialogue for the first sample in DOCCI train split and display the output 
with "LEAD1" instruct, run:

```bash
cd ssvd;
uv run generate.py run --task lead1 --dataset docci --verbose True --end_idx 1
```

*Example output: (train_00000 of DOCCI)*
>  - First things first, what's with the extra limbs? Is that a human-like creature or some sort of insect?
>  - I'd say it's a unique blend. It has four arms and two legs like a human, but those antennae and the insect-like features on its head suggest something else. It's quite the fusion!
>  - Four arms, two legs, antennae... I see. So, is it also wearing a suit and tie?
>  - No, it's not wearing a suit or tie. It has purple skin and is wearing a green cap with orange lines. No signs of formal clothing here.
>  - Yeah, I see the cap. It says 'PAN' on it. Is that a musical instrument on its head?
>  - No, that's not an instrument on its head. It's a green cap with orange lines, and 'PAN' is written in black on the front.
>  - Oh, so it's just a cap? What's up with the orange shoes then?
>  - The character is indeed wearing orange shoes. They have a green tongue and white, loose laces. Quite a vibrant pair!


To generate synthetic visual dialogues for each of the image in the DOCCI train split for the 
most general "COMB" instruct, run the following command. 
The output will be generated in the database `./synthetic_visual_dialogues/docci_ssvd.db`

```bash
uv run generate.py run --task comb --dataset docci --batch_size 64
```

If you want to generate one more dialogue for each of the image next, simply re-run the command in resume mode:

```bash
uv run generate.py run --task comb --dataset docci --batch_size 64 --overwrite resume
```


For further options of the `run` command:

```bash
uv run generate.py run --help
```

To view a sample of the generated database (e.g., 10th sample), you can use:

```bash
uv run generate.py watch --task comb --dataset docci --idx 10
```

*Example output: (train_00012 of DOCCI)*
> USER: So, what's going on in this image then?
> MOSHIVIS: A stone brick wall, slightly bowed, occupies the center, with limestone-colored bricks and brown mortar. Above it, a long, wide shadow rises towards the top right, resembling a crack in the smooth, light tan cave ceiling, dotted with white splotches and lit by a spotlight.
> USER: What color are the bricks and mortar of the wall?
> MOSHIVIS: Limestone-colored bricks with brown mortar.
> USER: How many bricks can you count in the wall?
> MOSHIVIS: I'm afraid the image doesn't provide a clear count of individual bricks. The wall appears to be made up of limestone-colored bricks with brown mortar, but the perspective and lighting make it difficult to distinguish each brick.
> USER: I see. How many distinct sections of stone brick wall can you make out in the image?
> MOSHIVIS: There are three distinct sections of stone brick wall in the image. The largest one is bowed and takes up the central part, with two smaller sections in the top left and bottom right corners.



### Setup
We currently support three captioning datasets out of the box: [DOCCI](https://huggingface.co/datasets/google/docci), [PixMo-Cap](https://huggingface.co/datasets/allenai/pixmo-cap) and [PixelProse](https://huggingface.co/datasets/tomg-group-umd/pixelprose). In all cases, the dataset will be downloaded fron HuggingFace when first running the code, if it not already found in your HuggingFace cache.
  


### Description of instruct prompts
We design several instructs / tasks to capture several types of dialogues about images:

  * `PROP`: **Prop**erties of objects (color, texture etc.)
  * `LOC`: **Loc**ation of objects
  * `NUM`: **Num**ber of objects
  * `LEAD1`: Mis**lead**ing questions about objects in the image
  * `LEAD2`: More emphasis on mis**lead**ing questions
  * `TNS1`: Conversation between an informative **t**eacher (assistant) a**n**d a **s**tudent wanting to learn about the image(user)
  * `TNS2`: Same as **TNS**1 but the student now has less information about the input image.
  * `COMB`: Combination of the previous instructs; It starts with a generic question (e.g. what is in the image), then randomly samples one of the aforementioned instructs for the subsequent conversation turns.

Please refer to Appendix E of [our preprint](https://arxiv.org/abs/2503.15633) or to `multiturn_instruct.py` for a more detailed description.
