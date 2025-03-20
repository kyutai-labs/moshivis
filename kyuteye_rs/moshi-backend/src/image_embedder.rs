use anyhow::Result;
use candle::{Device, Tensor};
use candle_nn::{linear, Linear, VarBuilder};
use candle_transformers::models::fastvit as MobileClip;
use candle_transformers::models::pixtral::vision_model as Pixtral;
use candle_transformers::models::siglip as Siglip;
use moshi::transformer::{CaSrc, Norm};
use moshi::NormType;

fn load_image(
    bytes: &[u8],
    max_size: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
    center_crop: bool,
    preserve_aspect_ratio: bool,
) -> candle::Result<Tensor> {
    // Load RGB image and resize such that longest side is equal to `max_size`
    // if crop_to_square is True, the resize also crops the image to square
    let mut img = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()?
        .decode()
        .map_err(candle::Error::wrap)?;

    // if not center crop, we just resize to the max size
    let (img, width, height) = if !center_crop {
        if preserve_aspect_ratio {
            let (width, height) = (img.width(), img.height());
            let (new_width, new_height) = if width < height {
                (((width * max_size as u32) / height) as usize, max_size)
            } else {
                (max_size, ((height * max_size as u32) / width) as usize)
            };
            let img = img.resize_exact(width, height, image::imageops::FilterType::CatmullRom);
            (img, new_width, new_height)
        } else {
            (
                img.resize_exact(
                    max_size as u32,
                    max_size as u32,
                    image::imageops::FilterType::CatmullRom,
                ),
                max_size,
                max_size,
            )
        }
    }
    // otherwise, we first center crop to a square of (max_size, maz_size) then resize
    else {
        let (width, height) = (img.width(), img.height());
        let min_dim = if width > height { height } else { width };
        let x = (width - min_dim) / 2;
        let y = (height - min_dim) / 2;
        //print!("center crop: {} {} {} {} {}\n", x, y, min_dim, width, height);
        let img = img.crop(x, y, min_dim, min_dim);
        let img = img.resize_exact(
            max_size as u32,
            max_size as u32,
            image::imageops::FilterType::CatmullRom,
        );
        (img, max_size, max_size)
    };

    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (height, width, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    tracing::info!(data = ?data.shape(), "image loaded");

    let mean = Tensor::new(mean, &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(std, &Device::Cpu)?.reshape((3, 1, 1))?;
    (data.to_dtype(candle::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

// Image Encoder for vision-conditioned models
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ImageEncoder {
    Siglip224,
    Siglip448,
    Siglip896,
    MobileclipS1,
    MobileclipS2,
    Pixtral,
}

#[derive(Debug, Clone)]
pub enum ImageEncoderModel {
    Siglip(Siglip::VisionModel),
    Mobileclip(candle_nn::Func<'static>),
    Pixtral(Pixtral::Model),
}

fn init_output_proj(in_dims: usize, out_dims: usize, vb: VarBuilder) -> Result<Option<Linear>> {
    let proj = if vb.contains_tensor("proj_xa.weight") {
        Some(linear(in_dims, out_dims, vb.pp("proj_xa"))?)
    } else {
        None
    };
    Ok(proj)
}

#[derive(Debug, Clone)]
pub struct ImageEmbedder {
    model: ImageEncoderModel,
    // optional output proj
    proj: Option<Linear>,
    // output norm
    norm: Norm,
    //image loading param
    mean: [f32; 3],
    std: [f32; 3],
    // max image size
    // corresponds to maximum allowed image dimension for the model
    // For Pixtral, this is the maximum side
    // For Siglip and Mobileclip, images are fixed to a square size for now
    max_image_size: usize,
    patch_size: usize,
    // whether the lm model will be quantized (need F32 inputs)
    quantized_lm_model: bool,
}

impl ImageEmbedder {
    pub fn new(
        model_file: &str,
        image_prefix_backbone: Option<ImageEncoder>,
        image_prefix_rmsnorm: bool,
        cross_attention_in_dims: Option<usize>,
        dev: &Device,
    ) -> Result<Self> {
        let out_dims = cross_attention_in_dims.unwrap_or(4096);
        let quantized_lm_model = model_file.ends_with(".gguf");
        let vb = if quantized_lm_model {
            // model_file should have format XXX.quant_format.gguf
            // and the unquantized vision tower weights should be in XXX_vision_tower_unquant.safetensors
            let base_path = model_file.rsplitn(3, '.').nth(2);
            let out_path = match base_path {
                None => anyhow::bail!(".gguf file does not have a corresponding vision tower"),
                Some(p) => format!("{}_vision_tower_unquant.safetensors", p),
            };
            tracing::info!(?out_path, "Loading unquantized vision encoder from");
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[out_path.as_str()], candle::DType::F32, dev)?
            }
        } else {
            // if safetensors, we assume the file contains *all* model's tensors
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], candle::DType::F32, dev)? }
        };

        let vb = vb.pp("image_prefix");
        let norm_typ = match image_prefix_rmsnorm {
            true => NormType::RmsNorm,
            false => NormType::LayerNorm,
        };

        // output norm
        let norm = Norm::new_shortcut(
            out_dims,
            norm_typ,
            moshi::nn::MaybeQuantizedVarBuilder::Real(vb.pp("norm_xa")),
        )?;

        let max_image_size = match image_prefix_backbone {
            Some(
                ImageEncoder::Siglip224 | ImageEncoder::MobileclipS1 | ImageEncoder::MobileclipS2,
            ) => 224,
            Some(ImageEncoder::Siglip448) => 448,
            Some(ImageEncoder::Siglip896) => 896,
            Some(ImageEncoder::Pixtral) => 1024,
            None => anyhow::bail!("Image encoder type not specified in config"),
        };

        // Main backbone
        match image_prefix_backbone {
            Some(ImageEncoder::Siglip224 | ImageEncoder::Siglip448 | ImageEncoder::Siglip896) => {
                // Siglip
                // https://huggingface.co/google/paligemma-3b-pt-224/blob/main/config.json
                let siglip_cfg = match image_prefix_backbone {
                    Some(ImageEncoder::Siglip224) => Siglip::VisionConfig::paligemma_3b_224(),
                    Some(ImageEncoder::Siglip448) => Siglip::VisionConfig::paligemma_3b_448(),
                    Some(ImageEncoder::Siglip896) => Siglip::VisionConfig::paligemma_3b_896(),
                    _ => anyhow::bail!("Impossible match arm. Congrats on reaching there"),
                };
                let model = Siglip::VisionModel::new(&siglip_cfg, false, vb.pp("enc.model"))?;
                let proj = init_output_proj(siglip_cfg.hidden_size, out_dims, vb)?;
                Ok(Self {
                    model: ImageEncoderModel::Siglip(model),
                    proj,
                    norm,
                    mean: [0.5, 0.5, 0.5],
                    std: [0.5, 0.5, 0.5],
                    max_image_size,
                    patch_size: siglip_cfg.patch_size,
                    quantized_lm_model,
                })
            }
            Some(ImageEncoder::MobileclipS1) | Some(ImageEncoder::MobileclipS2) => {
                // mobileclip (mci1/2 from candle)
                let vit_cfg = match image_prefix_backbone {
                    Some(ImageEncoder::MobileclipS1) => MobileClip::Config::mci1(),
                    Some(ImageEncoder::MobileclipS2) => MobileClip::Config::mci2(),
                    _ => anyhow::bail!("No image backbone specified for cross-attention layers"),
                };
                let model = MobileClip::fastvit_no_final_layer(&vit_cfg, vb.pp("enc.model"))?;
                let proj = init_output_proj(vit_cfg.in_channels * 16, out_dims, vb)?;
                Ok(Self {
                    model: ImageEncoderModel::Mobileclip(model),
                    proj,
                    norm,
                    mean: [0., 0., 0.],
                    std: [1., 1., 1.],
                    max_image_size,
                    patch_size: 1,
                    quantized_lm_model,
                })
            }
            Some(ImageEncoder::Pixtral) => {
                let pixtral_cfg = Pixtral::Config::pixtral_12b_2409();
                let model = Pixtral::Model::new(&pixtral_cfg, vb.pp("enc.model"))?;
                let proj = init_output_proj(pixtral_cfg.hidden_size, out_dims, vb)?;
                Ok(Self {
                    model: ImageEncoderModel::Pixtral(model),
                    proj,
                    norm,
                    mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
                    std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
                    max_image_size,
                    patch_size: pixtral_cfg.patch_size,
                    quantized_lm_model,
                })
            }
            None => {
                anyhow::bail!("No image backbone specified for cross-attention layers")
            }
        }
    }

    pub fn output_proj(&self, img_features: Tensor, dev: &Device) -> Result<Tensor> {
        // Output linear + normalization
        let img_features = match &self.proj {
            None => img_features.apply(&self.norm)?,
            Some(module) => img_features.apply(module)?.apply(&self.norm)?,
        };

        tracing::info!(feat = ?img_features.shape(), "image features generated");
        // Type conversion
        let dtype = if dev.is_cuda() && !self.quantized_lm_model {
            candle::DType::BF16
        } else {
            candle::DType::F32
        };
        Ok(img_features.to_dtype(dtype)?)
    }

    pub fn embed(
        &self,
        img_bytes: &[u8],
        image_size: usize,
        center_crop: bool,
        dev: &Device,
    ) -> Result<CaSrc> {
        // load Uint image as tensor then embed
        // to avoid any issue with the position embeddings interpolations, resize
        // images to the closest multiple of the model's patch size
        let image_size = if image_size % self.patch_size > self.patch_size / 2 {
            image_size - image_size % self.patch_size + self.patch_size
        } else {
            image_size - image_size % self.patch_size
        };

        // too small image sizes are very out of distributions so we clamp
        // the input image size to something reasonable
        let min_image_size = self.patch_size * 10;
        let image_size = if !(min_image_size..=self.max_image_size).contains(&image_size) {
            tracing::info!(
                "Limiting image size up to be in [{}, {}]",
                min_image_size,
                self.max_image_size
            );
            image_size.clamp(min_image_size, self.max_image_size)
        } else {
            image_size
        };

        let img_features = match &self.model {
            // Pixtral handles dynamic image size + non-square ratios by nature
            ImageEncoderModel::Pixtral(m) => load_image(
                img_bytes,
                image_size,
                &self.mean,
                &self.std,
                center_crop,
                true,
            )?
            .to_device(dev)?
            .unsqueeze(0)?
            .apply(m)?,
            // Siglip now also handles dynamic size/ratios with positional embedding interpolation
            ImageEncoderModel::Siglip(m) => load_image(
                img_bytes,
                image_size,
                &self.mean,
                &self.std,
                center_crop,
                false,
            )?
            .to_device(dev)?
            .unsqueeze(0)?
            .apply(m)?,
            // But MobileClip always resize to its own fixed (and square) image size
            ImageEncoderModel::Mobileclip(m) => load_image(
                img_bytes,
                self.max_image_size,
                &self.mean,
                &self.std,
                center_crop,
                false,
            )?
            .to_device(dev)?
            .unsqueeze(0)?
            .apply(m)?
            .flatten_from(2)?
            .transpose(1, 2)?,
        };
        Ok(CaSrc::Tokens(self.output_proj(img_features, dev)?))
    }

    pub fn embed_from_tensor(&self, img: Tensor, dev: &Device) -> Result<CaSrc> {
        // embed image preloaded as safetensors
        let img_features = match &self.model {
            // Currently siglip and MobileClip only supports fixed size (from config)
            ImageEncoderModel::Siglip(m) => img.apply(m)?,
            ImageEncoderModel::Mobileclip(m) => img.apply(m)?.flatten_from(2)?.transpose(1, 2)?,
            // Pixtral supports any size, but we may need to update the positional embeddings
            ImageEncoderModel::Pixtral(m) => img.apply(m)?,
        };
        Ok(CaSrc::Tokens(self.output_proj(img_features, dev)?))
    }
}
