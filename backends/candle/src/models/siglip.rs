use crate::layers::{get_cublas_lt_wrapper, HiddenAct, LayerNorm, Linear};
use crate::models::Model;
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

// To be compatible with the original google repository
// handle the full config but we only care about the text part
#[derive(Debug, Clone, Deserialize)]
pub struct SiglipConfig {
    pub text_config: SiglipTextConfig,
}

fn default_text_vocab_size() -> usize {
    32000
}

fn default_text_hidden_size() -> usize {
    768
}

fn default_text_intermediate_size() -> usize {
    3072
}

fn default_text_num_hidden_layers() -> usize {
    12
}

fn default_text_num_attention_heads() -> usize {
    12
}

fn default_text_max_position_embeddings() -> usize {
    64
}

fn default_text_layer_norm_eps() -> f64 {
    1e-6
}

fn default_text_pad_token_id() -> u32 {
    1
}

fn default_text_bos_token_id() -> u32 {
    49406
}

fn default_text_eos_token_id() -> u32 {
    49407
}

fn default_text_hidden_act() -> HiddenAct {
    // TEI version of GeluPytorchTanh
    HiddenAct::Gelu
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/configuration_siglip.py#L27
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SiglipTextConfig {
    #[serde(default = "default_text_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_text_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_text_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_text_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_text_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_text_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_text_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_text_pad_token_id")]
    pub pad_token_id: u32,
    #[serde(default = "default_text_bos_token_id")]
    pub bos_token_id: u32,
    #[serde(default = "default_text_eos_token_id")]
    pub eos_token_id: u32,
    #[serde(default = "default_text_hidden_act")]
    pub hidden_act: HiddenAct,
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new(cfg: &SiglipTextConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size();

        let query_weight = vb.pp("q_proj").get((embed_dim, embed_dim), "weight")?;
        let query_bias = vb.pp("q_proj").get(embed_dim, "bias")?;
        let q_proj = Linear::new(query_weight, Some(query_bias), None);

        let key_weight = vb.pp("k_proj").get((embed_dim, embed_dim), "weight")?;
        let key_bias = vb.pp("k_proj").get(embed_dim, "bias")?;
        let k_proj = Linear::new(key_weight, Some(key_bias), None);

        let value_weight = vb.pp("v_proj").get((embed_dim, embed_dim), "weight")?;
        let value_bias = vb.pp("v_proj").get(embed_dim, "bias")?;
        let v_proj = Linear::new(value_weight, Some(value_bias), None);

        let out_weight = vb.pp("out_proj").get((embed_dim, embed_dim), "weight")?;
        let out_bias = vb.pp("out_proj").get(embed_dim, "bias")?;
        let out_proj = Linear::new(out_weight, Some(out_bias), None);

        let num_heads = cfg.num_attention_heads();
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, q_len, _) = xs.dims3()?;
        let query_states = xs.apply(&self.q_proj)?;
        let key_states = xs.apply(&self.k_proj)?;
        let value_states = xs.apply(&self.v_proj)?;

        let shape = (batch_size, q_len, self.num_heads, self.head_dim);
        let query_states = query_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let key_states = key_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let value_states = value_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let attn_weights = (query_states.matmul(&key_states.t()?)? * self.scale)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        // The original implementation upcasts to f32 but candle_nn::ops::softmax should handle this properly.
        let attn_scores = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_outputs = attn_scores
            .matmul(&value_states)?
            .transpose(1, 2)?
            .reshape((batch_size, q_len, ()))?
            .apply(&self.out_proj)?;
        Ok(attn_outputs)
    }
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/modeling_siglip.py#L599
#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
    activation_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &SiglipTextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size();
        let intermediate_size = cfg.intermediate_size();
        let fc1_weight = vb.pp("fc1").get((intermediate_size, hidden_size), "weight")?;
        let fc1_bias = vb.pp("fc1").get(intermediate_size, "bias")?;
        let fc1 = Linear::new(fc1_weight, Some(fc1_bias), None);
        let fc2_weight = vb.pp("fc2").get((intermediate_size, hidden_size), "weight")?;
        let fc2_bias = vb.pp("fc2").get(hidden_size, "bias")?;
        let fc2 = Linear::new(fc2_weight, Some(fc2_bias), None);
        Ok(Self {
            fc1,
            fc2,
            activation_fn: cfg.hidden_act(),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &candle::Tensor) -> Result<candle::Tensor> {
        xs.apply(&self.fc1)?
            .apply(&self.activation_fn)?
            .apply(&self.fc2)
    }
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/modeling_siglip.py#L614
#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: Mlp,
    layer_norm2: LayerNorm,
}

impl EncoderLayer {
    fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size();
        let layer_norm_eps = cfg.layer_norm_eps();
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let layer_norm1 = layer_norm(hidden_size, layer_norm_eps, vb.pp("layer_norm1"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let layer_norm2 = layer_norm(hidden_size, layer_norm_eps, vb.pp("layer_norm2"))?;
        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.layer_norm1)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = xs.apply(&self.layer_norm2)?.apply(&self.mlp)?;
        let xs = (xs + residual)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let mut layers = vec![];
        let vb = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers() {
            let layer = EncoderLayer::new(cfg, vb.pp(layer_idx))?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, attention_mask)?
        }
        Ok(xs)
    }
}

pub struct SiglipTextModel {
    embeddings: SiglipTextEmbeddings,
    //encoder: SiglipTextEncoder,
    //final_layer_norm: LayerNorm,
    //pub head: Linear,
    num_attention_heads: usize,
    pool: Pool,
    device: Device,
    dtype: DType,
    //span: tracing::Span,
}

impl SiglipTextModel {
    pub fn load(vb: VarBuilder, config: &SiglipTextConfig, model_type: ModelType) -> Result<Self> {
        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("SiglipTextModel only supports embedding mode")
            }
            ModelType::Embedding(pool) => pool,
        };
        let embeddings = SiglipTextEmbeddings::new(config, vb.pp("embeddings"))?;
        Ok(Self {
            embeddings: embeddings,
            num_attention_heads: config.num_attention_heads,
            pool: pool,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let batch_size = batch.len();
        let max_length = batch.max_length as usize;
        let shape = (batch_size, max_length);

        println!(
            "SiglipTextModel::forward batch_size: {}, max_length: {}",
            batch_size, max_length
        );
        println!("batch input ids: {:?}", batch.input_ids);
        println!("batch token type ids: {:?}", batch.token_type_ids);
        println!("batch position ids: {:?}", batch.position_ids);
        println!("batch: {:?}", batch);

        let (input_ids, type_ids, position_ids, input_lengths, attention_bias, attention_mask) =
            if batch_size > 1 {
                // padded batch
                let elems = batch_size * max_length;
                let mut input_ids = Vec::with_capacity(elems);
                let mut type_ids = Vec::with_capacity(elems);
                let mut position_ids = Vec::with_capacity(elems);
                let mut attention_mask = Vec::with_capacity(elems);
                let mut attention_bias = Vec::with_capacity(elems);
                let mut input_lengths = Vec::with_capacity(batch_size);
                let mut masking = false;

                for i in 0..batch_size {
                    let start = batch.cumulative_seq_lengths[i] as usize;
                    let end = batch.cumulative_seq_lengths[i + 1] as usize;
                    let seq_length = (end - start) as u32;
                    input_lengths.push(seq_length as f32);

                    for j in start..end {
                        input_ids.push(batch.input_ids[j]);
                        type_ids.push(batch.token_type_ids[j]);
                        position_ids.push(batch.position_ids[j]);
                        attention_mask.push(1.0_f32);
                        attention_bias.push(0.0);
                    }

                    let padding = batch.max_length - seq_length;
                    if padding > 0 {
                        masking = true;
                        for _ in 0..padding {
                            input_ids.push(0);
                            type_ids.push(0);
                            position_ids.push(0);
                            attention_mask.push(0.0_f32);
                            attention_bias.push(f32::NEG_INFINITY);
                        }
                    }
                }

                let (attention_bias, attention_mask) = match masking {
                    true => {
                        // We only need the mask if we use mean pooling
                        // For CLS pooling, the bias is enough
                        let attention_mask = if self.pool == Pool::Mean {
                            let attention_mask = Tensor::from_vec(
                                attention_mask,
                                (batch_size, max_length, 1),
                                &self.device,
                            )?
                            .to_dtype(self.dtype)?;

                            Some(attention_mask)
                        } else {
                            None
                        };

                        let attention_bias = Tensor::from_vec(
                            attention_bias,
                            (batch_size, 1, 1, max_length),
                            &self.device,
                        )?
                        .to_dtype(self.dtype)?;
                        // Broadcast once instead of at every layer
                        let attention_bias = attention_bias
                            .broadcast_as((
                                batch_size,
                                self.num_attention_heads,
                                max_length,
                                max_length,
                            ))?
                            .contiguous()?;
                        (Some(attention_bias), attention_mask)
                    }
                    false => (None, None),
                };

                (
                    input_ids,
                    type_ids,
                    position_ids,
                    input_lengths,
                    attention_bias,
                    attention_mask,
                )
            } else {
                (
                    batch.input_ids,
                    batch.token_type_ids,
                    batch.position_ids,
                    vec![batch.max_length as f32],
                    None,
                    None,
                )
            };

        let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
        let type_ids = Tensor::from_vec(type_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(position_ids, shape, &self.device)?;
        let mut input_lengths =
            Tensor::from_vec(input_lengths, (batch_size, 1), &self.device)?.to_dtype(self.dtype)?;

        let input_ids = self.embeddings.forward(&input_ids)?;
        println!("embeddings: {:?}", input_ids);
        //let input_ids =

        /*/
            let (_bsz, seq_len) = input_ids.dims2()?;
            let input_ids = self.embeddings.forward(input_ids)?;
            let input_ids = self.encoder.forward(&input_ids, None)?;
            let last_hidden_state = self.final_layer_norm.forward(&input_ids)?;
            last_hidden_state
                .i((.., seq_len - 1, ..))?
                .contiguous()?
                .apply(&self.head)
        }
        */

        //let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
        let dummy_embedding = Tensor::zeros(shape, self.dtype, &self.device)?;
        return Ok((Some(dummy_embedding), None));
    }
}

impl Model for SiglipTextModel {
    fn is_padded(&self) -> bool {
        true
        //false
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }

    /*
    fn predict(&self, batch: Batch) -> Result<Tensor> {
        candle::bail!("`predict` is not implemented for this model")
    }
    */
}

#[derive(Debug, Clone)]
struct SiglipTextEmbeddings {
    token_embedding: Embedding,
    position_embedding: Embedding,
    position_ids: Tensor,
}

impl SiglipTextEmbeddings {
    fn new(config: &SiglipTextConfig, vb: VarBuilder) -> Result<Self> {
        let token_embedding = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("token_embedding"),
        )?;
        let position_embedding = candle_nn::embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embedding"),
        )?;
        let position_ids =
            Tensor::arange(0u32, config.max_position_embeddings as u32, vb.device())?
                .unsqueeze(0)?;
        Ok(Self {
            token_embedding,
            position_embedding,
            position_ids,
        })
    }
}

impl Module for SiglipTextEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.dim(D::Minus1)?;
        let inputs_embeds = self.token_embedding.forward(input_ids)?;
        let position_ids = self.position_ids.narrow(1, 0, seq_length)?;
        let position_embedding = self.position_embedding.forward(&position_ids)?;
        inputs_embeds.broadcast_add(&position_embedding)
    }
}
