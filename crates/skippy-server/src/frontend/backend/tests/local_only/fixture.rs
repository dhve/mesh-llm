//! Shared real GGUF fixture for collector and stderr privacy regressions.
use super::*;

pub(super) type CapturedTelemetry = Box<dyn FnMut() -> Vec<BTreeMap<String, serde_json::Value>>>;

pub(super) struct RealDecodeFixture {
    pub backend: StageOpenAiBackend,
    pub policy: Arc<RecordingHookPolicy>,
    pub receipt_sink: Arc<CountingReceiptSink>,
    pub proposal_source: Arc<CountingProposalIngress>,
    pub kv: Arc<crate::kv_integration::KvStageIntegration>,
    pub captured_telemetry: Option<CapturedTelemetry>,
    pub request: ChatCompletionRequest,
}

fn load_test_native_runtime(libs_dir: &std::path::Path) {
    if skippy_runtime::native_runtime_loaded() {
        return;
    }
    // Dependencies before the primary `libllama`, which carries the Skippy
    // ABI symbols; `libllama-common` and `libmtmd` depend on `libllama`.
    let order = [
        "libggml-base.dylib",
        "libggml-cpu.dylib",
        "libggml-blas.dylib",
        "libggml-metal.dylib",
        "libggml.dylib",
        "libllama.dylib",
        "libllama-common.dylib",
        "libmtmd.dylib",
    ];
    let paths: Vec<std::path::PathBuf> = order
        .iter()
        .map(|name| libs_dir.join(name))
        .filter(|path| path.is_file())
        .collect();
    assert!(
        paths.iter().any(|path| path.ends_with("libllama.dylib")),
        "no libllama.dylib under {}",
        libs_dir.display()
    );
    unsafe { skippy_runtime::load_native_runtime_libraries(&paths) }
        .expect("load the native runtime libraries for the local-only decode test");
}

pub(super) fn real_decode_fixture(stderr: bool) -> RealDecodeFixture {
    let model_path = std::env::var("SKIPPY_LOCAL_ONLY_TEST_MODEL")
        .expect("SKIPPY_LOCAL_ONLY_TEST_MODEL must point at a GGUF file");
    let libs_dir = std::env::var("SKIPPY_LOCAL_ONLY_TEST_RUNTIME_LIBS")
        .expect("SKIPPY_LOCAL_ONLY_TEST_RUNTIME_LIBS must point at the native runtime lib dir");
    let layer_count: u32 = std::env::var("SKIPPY_LOCAL_ONLY_TEST_MODEL_LAYERS")
        .ok()
        .map(|value| value.parse().expect("layer count"))
        .unwrap_or(24);
    assert!(
        std::path::Path::new(&model_path).is_file(),
        "model file does not exist: {model_path}"
    );
    load_test_native_runtime(std::path::Path::new(&libs_dir));

    let config = skippy_protocol::StageConfig {
        run_id: "local-only-real-decode".to_string(),
        topology_id: "local-only-real-decode".to_string(),
        model_id: "local-only-real-decode".to_string(),
        model_path: Some(model_path),
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: layer_count,
        ctx_size: 512,
        lane_count: 1,
        n_batch: Some(64),
        n_ubatch: Some(64),
        n_gpu_layers: 0,
        mmap: Some(true),
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        kv_cache: Some(skippy_protocol::StageKvCacheConfig {
            mode: skippy_protocol::StageKvCacheMode::LookupRecord,
            payload: skippy_protocol::StageKvCachePayload::Auto,
            max_entries: 8,
            max_bytes: 0,
            min_tokens: 1,
            shared_prefix_stride_tokens: 1,
            shared_prefix_record_limit: 0,
        }),
        native_mtp_enabled: false,
        load_mode: skippy_protocol::LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: None,
        ..skippy_protocol::StageConfig::default()
    };
    let runtime = crate::runtime_state::load_runtime(&config)
        .expect("load runtime")
        .expect("runtime loaded");
    let kv = crate::kv_integration::KvStageIntegration::from_config(
        &config,
        skippy_runtime::ModelStateKind::Dense,
    )
    .expect("kv integration config")
    .map(Arc::new)
    .expect("kv integration enabled for the positive control");
    let (telemetry, captured_telemetry): (_, Option<CapturedTelemetry>) = if stderr {
        (
            crate::telemetry::Telemetry::new(
                None,
                65_536,
                config.clone(),
                crate::telemetry::TelemetryLevel::Off,
            ),
            None,
        )
    } else {
        let (telemetry, capture) = crate::telemetry::Telemetry::capture_for_test(config.clone());
        (telemetry, Some(Box::new(capture)))
    };
    assert!(telemetry.is_debug_enabled());
    let iteration_scheduler =
        IterationScheduler::new(runtime.clone(), &config, 1, true, telemetry.clone())
            .expect("iteration scheduler");
    let policy = Arc::new(RecordingHookPolicy::default());
    let receipt_sink = Arc::new(CountingReceiptSink::default());
    let proposal_source = Arc::new(CountingProposalIngress::default());
    let backend = StageOpenAiBackend {
        runtime: runtime.clone(),
        config: config.clone(),
        telemetry: telemetry.clone(),
        model_id: "local-only-real-decode".to_string(),
        default_max_tokens: 8,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        ctx_size: 512,
        mode: OpenAiBackendMode::LocalRuntime,
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        ngram_max: 0,
        speculative: SpeculativeDecodeConfig::default(),
        generation_limit: Arc::new(GenerationConcurrencyController::fixed(1)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: 1,
        generation_admission_timeout: Duration::from_secs(30),
        generation_service_estimator: Arc::new(GenerationServiceEstimator::new(1)),
        generation_session_locks: Arc::new(Mutex::new(BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(512)),
        hook_policy: Some(policy.clone()),
        generation_receipt: Some(crate::frontend::GenerationReceiptConfig::new(
            receipt_sink.clone(),
        )),
        linear_proposal_ingress: Some(
            crate::frontend::LinearProposalIngressConfig::new(
                proposal_source.clone(),
                Duration::from_millis(20),
                4,
            )
            .expect("proposal ingress"),
        ),
        kv: Some(kv.clone()),
        iteration_scheduler,
        local_only_scope: false,
    };
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-only-real-decode",
        "messages": [
            {"role": "system", "content": "Answer with one word."},
            {"role": "user", "content": "Say PONG."}
        ],
        "max_completion_tokens": 8,
        "temperature": 0
    }))
    .expect("chat request");
    set_chat_mesh_hooks_enabled(&mut request, true);

    RealDecodeFixture {
        backend,
        policy,
        receipt_sink,
        proposal_source,
        kv,
        captured_telemetry,
        request,
    }
}
