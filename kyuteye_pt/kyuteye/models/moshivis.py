"""Moshi the little AI"""

from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from kyuteye.config.kyuteye_config import KyuteyeConfig
from kyuteye.models.helium import Helium
from kyuteye.modules.streaming_utils import StreamingModule
from kyuteye.modules.transformer import Transformer
from kyuteye.modules.utils import ClampedEmbedding
from moshi.utils.sampling import sample_token


class MoshiVis(StreamingModule):
    """Moshi model derived from Audiocraft with extra stuff for vision conditioninign"""

    # Class attributes; extra special tokens
    end_of_text_padding_id = 0
    zero_token_id = -1
    ungenerated_token_id = -2

    def __init__(
        self,
        hidden_scale: float = 4.125,
        norm: str = "real_rms_norm_f32",
        gating: bool = True,
        activation: str = "silu",
        n_q: int = 8,
        dep_q: Optional[int] = None,
        audio_card: int = 1024,
        audio_context: Optional[int] = None,
        depformer: bool = False,
        depformer_multi_linear: bool = False,
        depformer_pos_emb: Optional[Literal["none", "sin", "rope", "sin_rope"]] = None,
        depformer_dim: Optional[int] = None,
        depformer_dim_feedforward: Optional[int] = None,
        depformer_num_layers: Optional[int] = None,
        depformer_num_heads: Optional[int] = None,
        depformer_weights_per_step: bool = False,
        depformer_context: Optional[int] = 8,
        depformer_gating: Optional[bool] = None,
        depformer_activation: Optional[str] = None,
        delays: Optional[List[int]] = None,
        text_card: int = 32000,
        text_context: Optional[int] = None,
        padding_token_id: int = 3,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a MoshiVis model"""
        super().__init__()
        # Set parameter for generation/preprocessing
        self.text_card = text_card
        self.audio_card = audio_card
        self.text_context = text_context
        self.text_padding_token_id = padding_token_id
        self.audio_context = audio_context
        self.n_q = n_q
        self.dep_q = dep_q or self.n_q
        assert delays is not None and len(delays) > 0, "Delays must be non empty"
        assert len(delays) <= self.num_codebooks, "Too many delays"
        if len(delays) < self.num_codebooks:
            delays = delays + [delays[-1]] * (self.num_codebooks - len(delays))
        self.delays = delays

        embeddings_factory = partial(
            ClampedEmbedding, device=device, dtype=dtype, zero_idx=self.zero_token_id
        )

        # LLM backbone (includes text embedding + text linear projection)
        self.llm = Helium(
            hidden_scale=hidden_scale,
            card=text_card
            + 1,  # Add an initial token in the embedding but not in the text linear
            output_card=text_card + int(padding_token_id is None),
            padding_token_id=padding_token_id,
            device=device,
            dtype=dtype,
            zero_token_id=self.zero_token_id,
            **kwargs,
        )

        # Audio input embeddings
        self.audio_emb = torch.nn.ModuleList(
            [
                embeddings_factory(audio_card + 1, self.llm.dim)
                for _ in range(self.num_audio_codebooks_in)
            ]
        )

        # Depformer
        self.depformer: Optional[torch.nn.Module] = None
        self.depformer_multi_linear = depformer_multi_linear
        if depformer:
            assert depformer_dim is not None
            assert depformer_num_heads is not None
            assert depformer_num_layers is not None
            assert depformer_pos_emb is not None
            if depformer_dim_feedforward is None:
                depformer_dim_feedforward = int(hidden_scale * depformer_dim)
            assert depformer_dim_feedforward is not None

            self.depformer_in = torch.nn.ModuleList(
                [
                    torch.nn.Linear(self.llm.dim, depformer_dim, bias=False)
                    for _ in range(
                        self.num_audio_codebooks_out if depformer_multi_linear else 1
                    )
                ]
            )
            # Text and audio input embeddings for the depformer
            self.depformer_emb = torch.nn.ModuleList(
                [
                    embeddings_factory(audio_card + 1, depformer_dim)
                    for _ in range(self.num_audio_codebooks_out - 1)
                ]
            )
            self.depformer_text_emb = embeddings_factory(text_card + 1, depformer_dim)

            self.depformer = Transformer(
                d_model=depformer_dim,
                dim_feedforward=depformer_dim_feedforward,
                positional_embedding=depformer_pos_emb,
                num_heads=depformer_num_heads,
                num_layers=depformer_num_layers,
                norm=norm,
                device=device,
                dtype=dtype,
                causal=True,
                cross_attention=False,
                context=depformer_context,
                gating=depformer_gating or gating,
                activation=depformer_activation or activation,
                weights_per_step=dep_q if depformer_weights_per_step else None,
            )
            # Output projection
            self.audio_linears = torch.nn.ModuleList(
                [
                    torch.nn.Linear(depformer_dim, audio_card, bias=False)
                    for _ in range(self.num_audio_codebooks_out)
                ]
            )

    @property
    def cross_attention(self) -> bool:
        """Shortcut for checking whether cross_attention i sused"""
        return self.llm.cross_attention

    @property
    def num_audio_codebooks_in(self) -> int:
        """Number of audio codebooks to model as input"""
        return self.n_q

    @property
    def num_audio_codebooks_out(self) -> int:
        """Number of audio codebooks to model in the depformer"""
        return self.dep_q

    @property
    def num_codebooks(self) -> int:
        """Number codebooks including text"""
        return self.num_audio_codebooks_in + 1

    @property
    def initial_audio_token_id(self) -> int:
        """Initial token for the audio codebooks"""
        return self.audio_card

    @property
    def initial_text_token_id(self) -> int:
        """Initial token for the text; takes into account the "fake/proxy"
        tokens for beginning and end of image if they have been set"""
        return self.text_card

    @property
    def audio_offset(self) -> int:
        """Offset in the audio codebook. Returns 1 because we always generate with text"""
        return 1

    def forward_text(
        self,
        input_ids: torch.Tensor,
        cross_attention_src: Optional[
            Tuple[torch.Tensor, torch.Tensor] | torch.Tensor
        ] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Forward pass for Moshi

        :param input_ids: Text + audio tokens of shape (batch, codebooks, seq length)
        :param cross_attention_src: Conditioning (image) tokens that can be
            cross-attended to through the cross attention module
        :param cross_attention_mask: Additional mask for the cross_attention_src.
            This is necessary mainly for Pixtral models, as the cross-attended images
            might be of different sizes and therefore padded.
        :param attention_mask: Optional attention mask on input_ids (e.g. used at
            generation for batched inference with left padding)
        :return: A tuple containing the
          * text logits (None if `text_or_audio` is audio)
          * audio logits (None if `text_or_audio` is text)
        """
        # Embed tokens
        inputs_embeds = torch.zeros((), device=input_ids.device)
        if self.audio_offset > 0:
            inputs_embeds = self.llm.text_emb(input_ids[:, 0, :])

        for cb_index in range(self.num_audio_codebooks_in):
            update = self.audio_emb[cb_index](
                input_ids[:, cb_index + self.audio_offset, :]
            )
            inputs_embeds += update

        # Pass through Helium
        transformer_out, gate_weight = self.llm(
            inputs_embeds=inputs_embeds,
            cross_attention_src=cross_attention_src,
            cross_attention_mask=cross_attention_mask,
            attention_mask=attention_mask,
            return_features=True,
        )

        # Output proj
        text_logits = self.llm.text_linear(transformer_out)[:, None]
        return transformer_out, text_logits, gate_weight

    def forward_depformer(
        self,
        depformer_cb_index: int,
        input_ids: torch.Tensor,
        depformer_input: torch.Tensor,
    ) -> torch.Tensor:
        """Forward one depformer step"""
        _, num_codes, seq_len = input_ids.shape
        assert self.depformer is not None
        assert (
            num_codes == 1
        ), f"Codebooks for Depformer streaming should be passed 1 by 1, got {num_codes}."
        assert (
            seq_len == 1
        ), f"Steps for Depformer streaming should be passed 1 by 1, got {seq_len}."
        assert (
            depformer_input.shape[1] == 1
        ), "Transformer output should be a for a single step."

        # project transformer out
        depformer_input = self.depformer_in[
            depformer_cb_index if self.depformer_multi_linear else 0
        ](depformer_input)

        # project input ids
        if depformer_cb_index == 0:
            depformer_input += self.depformer_text_emb(input_ids[:, 0])
        else:
            depformer_input += self.depformer_emb[depformer_cb_index - 1](
                input_ids[:, 0]
            )

        # depformer_input is [B, 1, depformer_dim].
        # The streaming state of the depformer ensures that the proper layer is run.
        dep_output, _ = self.depformer(depformer_input)
        logits = self.audio_linears[depformer_cb_index](dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits

    @property
    def device(self) -> torch.device:
        """Torch device"""
        return next(iter(self.parameters())).device

    def get_initial_token(self) -> torch.Tensor:
        """Returns the initial token that will be fed to the model to predict the
        very first timestep. This is akin to a beginning of sentence tokens but
        to handle potentially delayed codebooks

        :param text_or_audio: Whether we are predicting for text, audio, or both

        :return: A Tensor fo shape (B, K, 1)
        """
        zero = torch.full(
            [1, 1, 1], MoshiVis.zero_token_id, device=self.device, dtype=torch.long
        )
        audio_token = torch.full_like(
            zero, self.initial_audio_token_id or MoshiVis.zero_token_id
        )
        text_token = torch.full_like(
            zero, self.initial_text_token_id or MoshiVis.zero_token_id
        )

        return torch.cat(
            [text_token, audio_token.expand(-1, self.num_audio_codebooks_in, -1)], dim=1
        )


class MoshiVisGen(StreamingModule):
    """MoshiVis for autoregressive generation at inference"""

    def __init__(
        self,
        moshi_vis: MoshiVis,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        check: bool = False,
    ):
        assert not moshi_vis.training, "generation shouldn't be used in training mode."
        super().__init__()

        self.lm_model = moshi_vis
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.check = check
        self.max_delay = max(
            moshi_vis.delays
        )  # with delays, we need to generate a few more time steps.
        self.delays_cuda = torch.tensor(
            moshi_vis.delays, device=self.lm_model.device, dtype=torch.long
        )
        self.initial_token = self.lm_model.get_initial_token()

    def update_gen_kwargs(
        self,
        temp: Optional[float] = None,
        temp_text: Optional[float] = None,
        top_k: Optional[int] = None,
        top_k_text: Optional[int] = None,
    ) -> None:
        """update params for sampling during generation"""
        self.temp = temp or self.temp
        self.temp_text = temp_text or self.temp_text
        self.top_k = top_k or self.top_k
        self.top_k_text = top_k_text or self.top_k_text

    @property
    def model_dim(self) -> int:
        """Return dimension of the tokens in the model"""
        return self.lm_model.llm.dim

    @property
    def num_audio_codebooks_out(self) -> int:
        """Number of audio codebooks generated by the model"""
        return self.lm_model.num_audio_codebooks_out

    @classmethod
    def from_config(
        cls,
        kyuteye_config: KyuteyeConfig,
        moshi_weight: Optional[Dict[str, Any]] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        **gen_kwargs: Any,
    ) -> "MoshiVisGen":
        """Instantiate model from a config

        :param base config:
        :param moshi_weight
        """
        moshivis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)
        if moshi_weight is not None:
            missing_keys, _ = moshivis.load_state_dict(moshi_weight, strict=False)
            # cross-attention MHSA is shared across layers
            missing_keys = [
                k
                for k in missing_keys
                if ("cross_attention.mha" not in k or "layers.0" in k)
            ]
            assert len(missing_keys) == 0

        return MoshiVisGen(moshi_vis=moshivis.eval().to(device), **gen_kwargs)

    @torch.no_grad()
    def precompte_ca_kv(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompte kv proj for cross-attention"""
        ca_layer = self.lm_model.llm.transformer.layers[0].cross_attention.mha
        if hasattr(ca_layer, "in_proj_weight_kv"):
            splits = torch.chunk(ca_layer.in_proj_weight_kv, 2)
        else:
            splits = torch.chunk(ca_layer.in_proj_weight, 3)

        k = torch.nn.functional.linear(  # pylint: disable=not-callable
            embeddings, splits[-2]
        )
        v = torch.nn.functional.linear(  # pylint: disable=not-callable
            embeddings, splits[-1]  # type: ignore
        )
        return k, v

    @torch.no_grad()
    def step(
        self,
        input_tokens: torch.Tensor,
        ca_src: Optional[Tuple[torch.Tensor, torch.Tensor] | torch.Tensor] = None,
    ) -> Tuple[torch.Tensor | None, float]:
        """One step of generation"""
        state = self._streaming_state
        if state is None:
            raise RuntimeError(
                "You should wrap those calls with a `with lm_gen.streaming(): ...`."
            )
        lm_model = self.lm_model

        assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
        batch_size, num_codes, seq_len = input_tokens.shape
        assert seq_len == 1, "Only support being given steps one by one."
        needed_tokens = lm_model.num_codebooks - lm_model.num_audio_codebooks_out - 1
        assert (
            num_codes == needed_tokens
        ), f"We expect {needed_tokens} tokens from the user stream, got {num_codes}."

        current_input_cache = self.get_streaming_attribute(
            "cache",
            torch.full(
                (batch_size, self.lm_model.num_codebooks, self.max_delay + 2),
                self.lm_model.ungenerated_token_id,
                device=self.lm_model.device,
                dtype=torch.long,
            ),
        )
        current_offset = self.get_streaming_attribute("offset", 0)
        dcache_len = current_input_cache.shape[2]

        # write input_tokens (sent from Mimi) in OTHER codebooks
        for q_other in range(input_tokens.shape[1]):
            k = lm_model.num_audio_codebooks_out + lm_model.audio_offset + q_other
            write_position = (current_offset + lm_model.delays[k]) % dcache_len
            current_input_cache[:, k, write_position : write_position + 1] = (
                input_tokens[:, q_other]
            )

        # Only for the very beginning, we extend the initial token for the acoustic
        # token that are delayed, and thus have no good value to take.
        position = current_offset % dcache_len
        for k, delay in enumerate(lm_model.delays):
            if current_offset <= delay:
                current_input_cache[:, k, position] = self.initial_token[:, k, 0]

        # Transformer forward
        input_ = current_input_cache[:, :, position : position + 1]

        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                current_offset,
                input_,
            )
            assert (
                input_[:, lm_model.audio_offset :] <= lm_model.audio_card
            ).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()

        transformer_out, text_logits, gate_weight = self.lm_model.forward_text(
            input_, cross_attention_src=ca_src
        )

        # Sample text tokens
        # Shape of text_logits should be [B, K_text=1, T=1, Card_text]
        text_token = sample_token(
            text_logits.float(),
            self.use_sampling,
            self.temp_text,
            self.top_k_text,
        )
        assert text_token.dim() == 3, text_token.shape
        assert text_token.shape[2] == 1
        assert text_token.shape[1] == 1, "Only one text stream supported."
        text_token = text_token[:, 0, 0]  # shape is [B]

        # Generate and sample audio tokens
        audio_tokens = self.depformer_step(text_token, transformer_out)

        # Write generated tokens
        current_offset += 1
        position = current_offset % dcache_len
        current_input_cache[:, 0, position] = text_token
        current_input_cache[
            :,
            lm_model.audio_offset : lm_model.num_audio_codebooks_out
            + lm_model.audio_offset,
            position,
        ] = audio_tokens

        # if <= max_delay, we continue partial-generation
        # until removing all ungenerated tokens
        if current_offset <= self.max_delay:
            self.add_streaming_attribute("cache", current_input_cache)
            self.add_streaming_attribute("offset", current_offset)
            return None, 0.0

        # otherwise, retrieve tokens with the correct delay
        gen_delays_cuda = self.delays_cuda[
            : lm_model.num_audio_codebooks_out + lm_model.audio_offset
        ]
        index = (
            ((current_offset - self.max_delay + gen_delays_cuda) % dcache_len)
            .view(1, -1, 1)
            .expand(current_input_cache.shape[0], -1, 1)
        )
        out = current_input_cache.gather(dim=2, index=index)
        self.add_streaming_attribute("offset", current_offset)
        self.add_streaming_attribute("cache", current_input_cache)
        return out, gate_weight

    def depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """A step of the depformer"""
        batch_size = text_token.shape[0]
        depformer_tokens: list[torch.Tensor] = []
        assert self.lm_model.depformer is not None

        with self.lm_model.depformer.streaming():
            next_token = text_token[:, None, None]

            for cb_index in range(self.lm_model.num_audio_codebooks_out):
                logits = self.lm_model.forward_depformer(
                    cb_index, next_token, transformer_out
                )
                next_token = sample_token(
                    logits.float(),
                    self.use_sampling,
                    self.temp,
                    self.top_k,
                )
                assert next_token.shape == (batch_size, 1, 1)
                depformer_tokens.append(next_token[:, 0, 0])
        out = torch.stack(depformer_tokens, dim=1)
        return out
