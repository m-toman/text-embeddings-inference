mod common;

use crate::common::{sort_embeddings, SnapshotEmbeddings};
use anyhow::Result;
use common::{batch, cosine_matcher, download_artifacts, load_tokenizer};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
fn test_siglip() -> Result<()> {
    let (model_root, _) = download_artifacts("google/siglip-base-patch16-224", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    // SigLIP always pools the final position and applies the projection head;
    // the configured pooling mode is ignored by the model.
    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        None,
    )?;

    let input_batch = batch(
        vec![
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
            tokenizer.encode("Deep Learning is...", true).unwrap(),
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
        ],
        [0, 1, 2].to_vec(),
        vec![],
    );

    let matcher = cosine_matcher();

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_batch)?);
    let embeddings_batch = SnapshotEmbeddings::from(pooled_embeddings);
    insta::assert_yaml_snapshot!("siglip_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!("siglip_single", embeddings_single, &matcher);
    // Identical inputs must produce identical embeddings.
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_siglip_all() -> Result<()> {
    let (model_root, _) = download_artifacts("google/siglip-base-patch16-224", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        None,
    )?;

    // Request raw (per-token) embeddings. This exercises the pad-to-64 padding and
    // the padding-drop logic in `SiglipTextModel::forward`, which returns the
    // pre-head hidden states for the real tokens only.
    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        vec![],
        [0].to_vec(),
    );

    let (_, raw_embeddings) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_raw = SnapshotEmbeddings::from(raw_embeddings);

    let matcher = cosine_matcher();
    insta::assert_yaml_snapshot!("siglip_single_raw", embeddings_raw, &matcher);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_siglip_classifier_unsupported() -> Result<()> {
    let (model_root, _) = download_artifacts("google/siglip-base-patch16-224", None, None)?;

    // SigLIP is embedding-only; classification is explicitly rejected at load time.
    let result = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Classifier,
        None,
    );

    assert!(
        result.is_err(),
        "expected SigLIP to reject `ModelType::Classifier`"
    );

    Ok(())
}
