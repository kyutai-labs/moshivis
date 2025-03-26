// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
use crate::nn::{linear, MaybeQuantizedEmbedding, MaybeQuantizedLinear, MaybeQuantizedVarBuilder};
use crate::{
    transformer::{self, CaSrc},
    NormType,
};
use candle::{DType, Device, IndexOp, Module, Result, Tensor};

thread_local! {
    pub static VERBOSE: bool = {
        match std::env::var("MIMI_VERBOSE") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DepFormerConfig {
    pub transformer: transformer::Config,
    pub num_slices: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub transformer: transformer::Config,
    pub depformer: Option<DepFormerConfig>,
    pub text_in_vocab_size: usize,
    pub text_out_vocab_size: usize,
    pub audio_vocab_size: usize,
    pub audio_codebooks: usize,
}

impl Config {
    // /lustre/scwpod02/client/kyutai/alex/mimi_exp/xps/af78657c/outputs/hyperparams.json
    // Update 2024-03-19: Sin embeddings -> None, RmsNorm fix, scale factor 4.125
    // Update 2024-05-02: split text_vocab_size into text_in_vocab_size and text_out_vocab_size.
    // embeddings.
    pub fn v0_1() -> Self {
        let lm_cfg = transformer::Config {
            d_model: 4096,
            num_heads: 32,
            num_layers: 32,
            dim_feedforward: 4096 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 3000,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
        };
        let depformer_cfg = transformer::Config {
            d_model: 1024,
            num_heads: 16,
            num_layers: 6,
            dim_feedforward: 1024 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 8,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::None,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
        };
        let depformer_cfg = DepFormerConfig {
            num_slices: 8,
            transformer: depformer_cfg,
        };
        Self {
            transformer: lm_cfg,
            depformer: Some(depformer_cfg),
            audio_vocab_size: 2049,
            text_in_vocab_size: 32001,
            text_out_vocab_size: 32000,
            audio_codebooks: 8,
        }
    }

    pub fn v0_1_vision() -> Self {
        let lm_cfg = transformer::Config {
            d_model: 4096,
            num_heads: 32,
            num_layers: 32,
            dim_feedforward: 4096 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 3000,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: Some((
                transformer::CrossAttentionGating::ConditionalGatedSigmoid,
                NormType::RmsNorm,
                None,
            )),
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
        };
        let depformer_cfg = transformer::Config {
            d_model: 1024,
            num_heads: 16,
            num_layers: 6,
            dim_feedforward: 1024 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 8,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: crate::NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::None,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
        };
        let depformer_cfg = DepFormerConfig {
            num_slices: 8,
            transformer: depformer_cfg,
        };
        Self {
            transformer: lm_cfg,
            depformer: Some(depformer_cfg),
            audio_vocab_size: 2049,
            text_in_vocab_size: 32001,
            text_out_vocab_size: 32000,
            audio_codebooks: 8,
        }
    }

    pub fn v0_1_vision_streaming(num_slices: usize) -> Self {
        let mut s = Self::v0_1_vision();
        s.audio_codebooks = 16;
        if let Some(depformer) = s.depformer.as_mut() {
            depformer.num_slices = num_slices;
            depformer.transformer.context = num_slices;
        }
        s
    }

    pub fn v0_1_streaming(num_slices: usize) -> Self {
        let mut s = Self::v0_1();
        s.audio_codebooks = 16;
        if let Some(depformer) = s.depformer.as_mut() {
            depformer.num_slices = num_slices;
            depformer.transformer.context = num_slices;
        }
        s
    }
}

#[derive(Debug, Clone)]
struct DepFormerSlice {
    transformer: transformer::StreamingTransformer,
    // Note that the embedding for the first slice does not have the same dimension as the
    // embedding for the other slices as it takes a text token as input rather than an audio token.
    emb: MaybeQuantizedEmbedding,
    linear_in: MaybeQuantizedLinear,  // depformer_in.{idx}
    linear_out: MaybeQuantizedLinear, // linears.{idx}
}

impl DepFormerSlice {
    fn new(
        in_vocab_size: usize,
        out_vocab_size: usize,
        main_transformer_dim: usize,
        cfg: &transformer::Config,
        vb: MaybeQuantizedVarBuilder,
    ) -> Result<Self> {
        let dim = cfg.d_model;
        let transformer = transformer::StreamingTransformer::new(cfg, vb.pp("transformer"))?;
        let emb = MaybeQuantizedEmbedding::new(in_vocab_size, dim, vb.pp("emb"))?;
        let linear_in = linear(main_transformer_dim, dim, false, vb.pp("linear_in"))?;
        let linear_out = linear(dim, out_vocab_size, false, vb.pp("linear_out"))?;
        Ok(Self {
            transformer,
            emb,
            linear_in,
            linear_out,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DepFormer {
    slices: Vec<DepFormerSlice>,
}

impl DepFormer {
    pub fn new(
        text_vocab_size: usize,
        audio_vocab_size: usize,
        main_transformer_dim: usize,
        cfg: &DepFormerConfig,
        vb: MaybeQuantizedVarBuilder,
    ) -> Result<Self> {
        let mut slices = Vec::with_capacity(cfg.num_slices);
        for slice_idx in 0..cfg.num_slices {
            let in_vs = if slice_idx == 0 {
                text_vocab_size
            } else {
                audio_vocab_size
            };
            // The depformer cannot predict the audio padding token.
            let slice = DepFormerSlice::new(
                in_vs,
                audio_vocab_size - 1, // The depformer cannot emit an audio padding token.
                main_transformer_dim,
                &cfg.transformer,
                vb.pp(slice_idx),
            )?;
            slices.push(slice)
        }
        Ok(Self { slices })
    }

    /// Run a transformer sampling step, getting a token id per codebook.
    /// - `xs` is the previous layer hidden state.
    pub fn sample(
        &mut self,
        xs: &Tensor,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Vec<u32>> {
        use crate::streaming::StreamingModule;
        let dev = xs.device();
        let mut tokens = Vec::with_capacity(self.slices.len());
        let mut last_token = text_token;
        for slice_idx in 0..self.slices.len() {
            if slice_idx == 0 {
                self.slices[slice_idx].transformer.reset_state();
            } else {
                let (lhs, rhs) = self.slices.split_at_mut(slice_idx);
                rhs[0]
                    .transformer
                    .copy_state(&lhs[slice_idx - 1].transformer)?
            }
            let slice = &mut self.slices[slice_idx];
            let xs = slice.linear_in.forward(xs)?;
            let xs = match last_token {
                Some(last_token) => {
                    let token_id = Tensor::from_vec(vec![last_token], (1, 1), dev)?;
                    let token_emb = slice.emb.forward(&token_id)?;
                    xs.broadcast_add(&token_emb)?
                }
                None => xs,
            };
            let xs = slice.transformer.forward(&xs)?;
            let logits = xs.apply(&slice.linear_out)?;
            let logits = match logits.dim(0)? {
                1 => logits.i((0, 0))?,
                b_size => candle::bail!("unexpected batch size {b_size}"),
            };
            let token = lp.sample(&logits)?;
            if VERBOSE.with(|v| *v) {
                println!("sampled {token} logits {slice_idx}:\n{logits}");
            }
            tokens.push(token);
            let token_for_next_layer = forced_audio_tokens
                .get(slice_idx)
                .copied()
                .flatten()
                .unwrap_or(token);
            last_token = Some(token_for_next_layer);
        }
        Ok(tokens)
    }

    // Sampling with classifier free guidance.
    pub fn sample_cfg(
        &mut self,
        xs: &Tensor,
        cfg_alpha: f64,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Vec<u32>> {
        use crate::streaming::StreamingModule;
        let dev = xs.device();
        let mut tokens = Vec::with_capacity(self.slices.len());
        let mut last_token = text_token;
        for slice_idx in 0..self.slices.len() {
            if slice_idx == 0 {
                self.slices[slice_idx].transformer.reset_state();
            } else {
                let (lhs, rhs) = self.slices.split_at_mut(slice_idx);
                rhs[0]
                    .transformer
                    .copy_state(&lhs[slice_idx - 1].transformer)?
            }
            let slice = &mut self.slices[slice_idx];
            let xs = slice.linear_in.forward(xs)?;
            let xs = match last_token {
                Some(last_token) => {
                    let token_id = Tensor::from_vec(vec![last_token], (1, 1), dev)?;
                    let token_emb = slice.emb.forward(&token_id)?;
                    xs.broadcast_add(&token_emb)?
                }
                None => xs,
            };
            let xs = slice.transformer.forward(&xs)?;
            let logits = xs.apply(&slice.linear_out)?;
            let logits = match logits.dim(0)? {
                2 => ((logits.i((0, 0))? * cfg_alpha)? - (logits.i((1, 0))? * (cfg_alpha - 1.))?)?,
                b_size => candle::bail!("unexpected batch size {b_size}"),
            };
            let token = lp.sample(&logits)?;
            tokens.push(token);
            let token_for_next_layer = forced_audio_tokens
                .get(slice_idx)
                .copied()
                .flatten()
                .unwrap_or(token);
            last_token = Some(token_for_next_layer);
        }
        Ok(tokens)
    }
}

#[derive(Debug, Clone)]
pub struct LmModel {
    pub transformer: transformer::StreamingTransformer,
    pub text_emb: MaybeQuantizedEmbedding,
    pub audio_embs: Vec<MaybeQuantizedEmbedding>,
    pub text_linear: MaybeQuantizedLinear,
    pub out_norm: transformer::Norm,
    pub depformer: Option<DepFormer>,
    pub audio_vocab_size: usize,
    pub text_in_vocab_size: usize,
    pub dtype: DType,
}

impl LmModel {
    pub fn new(cfg: &Config, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        let d_model = cfg.transformer.d_model;
        let depformer = match &cfg.depformer {
            None => None,
            Some(depformer_cfg) => {
                let depformer = DepFormer::new(
                    cfg.text_in_vocab_size,
                    cfg.audio_vocab_size,
                    d_model,
                    depformer_cfg,
                    vb.pp("depformer"),
                )?;
                Some(depformer)
            }
        };
        let text_emb =
            MaybeQuantizedEmbedding::new(cfg.text_in_vocab_size, d_model, vb.pp("text_emb"))?;
        let out_norm = transformer::Norm::new(d_model, &cfg.transformer, vb.pp("out_norm"))?;
        let text_linear = linear(
            d_model,
            cfg.text_out_vocab_size,
            false,
            vb.pp("text_linear"),
        )?;
        let transformer =
            transformer::StreamingTransformer::new(&cfg.transformer, vb.pp("transformer"))?;
        let vb_e = vb.pp("emb");
        let mut audio_embs = Vec::with_capacity(cfg.audio_codebooks);
        for i in 0..cfg.audio_codebooks {
            let emb = MaybeQuantizedEmbedding::new(cfg.audio_vocab_size, d_model, vb_e.pp(i))?;
            audio_embs.push(emb)
        }
        let dtype = match vb {
            MaybeQuantizedVarBuilder::Real(weights) => weights.dtype(),
            MaybeQuantizedVarBuilder::Quantized(_) => DType::F32,
        };
        Ok(Self {
            transformer,
            text_emb,
            text_linear,
            audio_embs,
            out_norm,
            depformer,
            text_in_vocab_size: cfg.text_in_vocab_size,
            audio_vocab_size: cfg.audio_vocab_size,
            dtype,
        })
    }

    pub fn reset_state(&mut self) {
        use crate::streaming::StreamingModule;
        self.transformer.reset_state()
    }

    pub fn in_audio_codebooks(&self) -> usize {
        self.audio_embs.len()
    }

    pub fn audio_pad_token(&self) -> u32 {
        self.audio_vocab_size as u32 - 1
    }

    pub fn text_start_token(&self) -> u32 {
        self.text_in_vocab_size as u32 - 1
    }

    pub fn generated_audio_codebooks(&self) -> usize {
        self.depformer.as_ref().map_or(0, |v| v.slices.len())
    }

    pub fn is_quantized(&self) -> bool {
        match self.text_linear {
            MaybeQuantizedLinear::Quantized(_) => true,
            MaybeQuantizedLinear::Real(_) => false,
        }
    }

    pub fn device(&self) -> &Device {
        self.text_emb.embeddings().device()
    }

    pub fn forward(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
    ) -> candle::Result<(Tensor, Tensor)> {
        if VERBOSE.with(|v| *v) {
            print!("text_ids ");
            if let Some(text_ids) = text_ids.as_ref() {
                let text_ids = text_ids.flatten_all()?.to_vec1::<u32>()?;
                println!("{text_ids:?}");
            } else {
                println!("none")
            }
            print!("audio_ids ");
            for audio_id in audio_ids.iter() {
                if let Some(audio_id) = audio_id {
                    let audio_id = audio_id.flatten_all()?.to_vec1::<u32>()?;
                    print!(" {audio_id:?}");
                } else {
                    print!(" none")
                }
            }
            println!();
        }
        let mut emb = match text_ids.as_ref() {
            Some(text_ids) => text_ids.apply(&self.text_emb)?,
            None => {
                let device = self.text_emb.embeddings().device();
                Tensor::zeros((1, 1, self.text_emb.hidden_size()?), self.dtype, device)?
            }
        };

        for (audio_emb, audio_ids) in self.audio_embs.iter().zip(audio_ids.iter()) {
            if let Some(audio_ids) = audio_ids {
                let e = audio_ids.apply(audio_emb)?;
                emb = (emb + e)?
            }
        }
        let ys = self.transformer.forward(&emb)?;
        let ys = ys.apply(&self.out_norm)?;
        let logits = ys.apply(&self.text_linear)?;
        if VERBOSE.with(|v| *v) {
            println!("logits:\n{logits}");
        }
        Ok((logits, ys))
    }

    pub fn maybe_precompute_ca_kv(&self, ca_src: Option<CaSrc>) -> Result<Option<CaSrc>> {
        let ca_src = match ca_src {
            None => None,
            z => self.transformer.maybe_precompute_ca_kv(z)?,
        };
        Ok(ca_src)
    }

    pub fn forward_ca(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        ca_src: &CaSrc,
    ) -> candle::Result<(Tensor, Tensor)> {
        let (logits, ys, _) = self.forward_with_gate_weight(text_ids, audio_ids, ca_src)?;
        Ok((logits, ys))
    }

    pub fn forward_with_gate_weight(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        ca_src: &CaSrc,
    ) -> candle::Result<(Tensor, Tensor, Tensor)> {
        if VERBOSE.with(|v| *v) {
            print!("text_ids ");
            if let Some(text_ids) = text_ids.as_ref() {
                let text_ids = text_ids.flatten_all()?.to_vec1::<u32>()?;
                println!("{text_ids:?}");
            } else {
                println!("none")
            }
            print!("audio_ids ");
            for audio_id in audio_ids.iter() {
                if let Some(audio_id) = audio_id {
                    let audio_id = audio_id.flatten_all()?.to_vec1::<u32>()?;
                    print!(" {audio_id:?}");
                } else {
                    print!(" none")
                }
            }
            println!();
        }
        let b_size = match ca_src {
            CaSrc::KeysValues((cak, _)) => cak.dim(0)?,
            CaSrc::Tokens(catoks) => catoks.dim(0)?,
        };
        let mut emb = match text_ids {
            Some(text_ids) => text_ids.apply(&self.text_emb)?,
            None => {
                let device = self.text_emb.embeddings().device();
                Tensor::zeros(
                    (b_size, 1, self.text_emb.hidden_size()?),
                    self.dtype,
                    device,
                )?
            }
        };
        for (audio_emb, audio_ids) in self.audio_embs.iter().zip(audio_ids.iter()) {
            if let Some(audio_ids) = audio_ids {
                let e = audio_ids.apply(audio_emb)?;
                emb = emb.broadcast_add(&e)?
            }
        }
        let (ys, alpha) = self
            .transformer
            .forward_with_gate_weight(&emb, Some(ca_src))?;
        let ys = ys.apply(&self.out_norm)?;
        let logits = ys.apply(&self.text_linear)?;
        Ok((logits, ys, alpha))
    }

    pub fn depformer_sample(
        &mut self,
        xs: &Tensor,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>> {
        let sample = match &mut self.depformer {
            None => None,
            Some(m) => {
                let sample = m.sample(xs, text_token, forced_audio_tokens, lp)?;
                Some(sample)
            }
        };
        Ok(sample)
    }
}

pub fn load_lm_model<P: AsRef<std::path::Path>>(
    cfg: Config,
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let quantized = model_file.as_ref().extension().is_some_and(|v| v == "gguf");
    let vb = if quantized {
        MaybeQuantizedVarBuilder::Quantized(
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(model_file, dev)?,
        )
    } else {
        unsafe {
            MaybeQuantizedVarBuilder::Real(candle_nn::VarBuilder::from_mmaped_safetensors(
                &[model_file],
                dtype,
                dev,
            )?)
        }
    };
    let model = LmModel::new(&cfg, vb)?;
    Ok(model)
}

pub fn load<P: AsRef<std::path::Path>>(
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let cfg = Config::v0_1();
    load_lm_model(cfg, model_file, dtype, dev)
}

pub fn load_streaming<P: AsRef<std::path::Path>>(
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let cfg = Config::v0_1_streaming(8);
    load_lm_model(cfg, model_file, dtype, dev)
}

pub fn load_streaming_both_ways<P: AsRef<std::path::Path>>(
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let cfg = Config::v0_1_streaming(16);
    load_lm_model(cfg, model_file, dtype, dev)
}

pub fn load_vision<P: AsRef<std::path::Path>>(
    model_file: P,
    override_cross_attention_gating: Option<transformer::CrossAttentionGating>,
    override_cross_attention_in_dim: Option<usize>,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    // load_vision allows for overriding some hyperparams of the lm from the main config file
    let mut cfg = Config::v0_1_vision_streaming(8);
    cfg.transformer.cross_attention = override_cross_attention_gating
        .map(|v| (v, cfg.transformer.norm, override_cross_attention_in_dim));
    load_lm_model(cfg, model_file, dtype, dev)
}

pub struct ForcedAudioTokens {
    acoustic_delay: usize,
    // Tokens that are teacher forced before the acoustic delay.
    pre_delay_tokens: Vec<Option<u32>>,
}

impl ForcedAudioTokens {
    pub fn new(acoustic_delay: usize, audio_pad_token: u32, stream_codebooks: &[usize]) -> Self {
        let mut pre_delay_tokens = vec![];
        for codebooks in stream_codebooks.iter() {
            for c in 0..*codebooks {
                let token = if c == 0 { None } else { Some(audio_pad_token) };
                pre_delay_tokens.push(token);
            }
        }
        Self {
            acoustic_delay,
            pre_delay_tokens,
        }
    }

    pub fn forced_tokens(&self, step_idx: usize) -> &[Option<u32>] {
        if step_idx < self.acoustic_delay {
            &self.pre_delay_tokens
        } else {
            &[]
        }
    }
}
