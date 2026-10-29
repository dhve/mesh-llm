//! Local-only content observer suppression, including the opt-in real-model control.
use super::*;

mod fixture;
mod stderr;

#[derive(Default)]
struct CountingReceiptSink {
    calls: std::sync::atomic::AtomicUsize,
}

impl crate::frontend::GenerationReceiptSink for CountingReceiptSink {
    fn begin(&self, _start: &crate::frontend::GenerationStart) -> anyhow::Result<()> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn committed(&self, _commit: &crate::frontend::GenerationCommit) -> anyhow::Result<()> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn abort(&self, _abort: &crate::frontend::GenerationAbort) -> anyhow::Result<()> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn record(&self, _receipt: &crate::frontend::GenerationReceipt) -> anyhow::Result<()> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

/// Proposal source that only counts how often Skippy consults it.
#[derive(Default)]
struct CountingProposalIngress {
    calls: std::sync::atomic::AtomicUsize,
}

impl crate::frontend::LinearProposalIngress for CountingProposalIngress {
    fn propose(
        &self,
        _query: crate::frontend::LinearProposalQuery,
    ) -> anyhow::Result<crate::frontend::LinearProposalSourceResponse> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(crate::frontend::LinearProposalSourceResponse::new(None))
    }

    fn report(&self, _receipt: &crate::frontend::LinearProposalReceipt) -> anyhow::Result<()> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

/// `OpenAiRequestContext::with_local_only_policy` must dispatch through a
/// scoped backend that carries no content observer: no hook policy, no
/// generation receipt, no proposal ingress, no prefix cache, and no debug
/// telemetry (which attaches `predicted_token` per decode step). The
/// ordinary path keeps all of them.
#[tokio::test]
async fn local_only_policy_runs_on_a_scoped_backend_without_content_observers() {
    let policy = Arc::new(RecordingHookPolicy {
        deny: true,
        ..RecordingHookPolicy::default()
    });
    let receipt_sink = Arc::new(CountingReceiptSink::default());
    let proposal_source = Arc::new(CountingProposalIngress::default());
    let base = hooks_test_backend(Some(policy.clone()));
    let debug_telemetry = crate::telemetry::Telemetry::new(
        None,
        1,
        base.config.clone(),
        crate::telemetry::TelemetryLevel::Debug,
    );
    let backend = StageOpenAiBackend {
        generation_receipt: Some(crate::frontend::GenerationReceiptConfig::new(
            receipt_sink.clone(),
        )),
        linear_proposal_ingress: Some(
            crate::frontend::LinearProposalIngressConfig::new(
                proposal_source.clone(),
                Duration::from_millis(5),
                4,
            )
            .expect("bounded proposal ingress"),
        ),
        telemetry: debug_telemetry,
        ..base
    };
    assert!(backend.telemetry.is_debug_enabled());

    let scoped = backend.scoped_for_local_only();
    assert!(scoped.local_only_scope);
    assert!(scoped.hook_policy.is_none(), "hook policy cleared");
    assert!(scoped.generation_receipt.is_none(), "receipt sink cleared");
    assert!(
        scoped.linear_proposal_ingress.is_none(),
        "proposal ingress cleared"
    );
    assert!(scoped.kv.is_none(), "prefix cache cleared");
    assert!(
        !scoped.telemetry.is_debug_enabled(),
        "token-level debug telemetry capped"
    );
    assert!(
        scoped.telemetry.is_enabled(),
        "summary telemetry still flows"
    );
    assert!(
        backend.hook_policy.is_some()
            && backend.generation_receipt.is_some()
            && backend.linear_proposal_ingress.is_some()
            && !backend.local_only_scope,
        "the ordinary backend is unchanged"
    );

    // Ordinary dispatch reaches the hook policy, which denies here.
    let ordinary = backend
        .chat_completion_with_context(
            mesh_hooks_request("hooks-test-model"),
            OpenAiRequestContext::new(),
        )
        .await;
    assert!(
        ordinary.is_err(),
        "the denying hook ran on the ordinary path"
    );
    wait_for_hook_terminal(&policy).await;
    assert_eq!(policy.terminals.lock().unwrap().len(), 1);

    // The local-only context never reaches the hook, even though the request
    // asks for mesh hooks, and never touches the receipt sink.
    let _local_only = backend
        .chat_completion_with_context(
            mesh_hooks_request("hooks-test-model"),
            OpenAiRequestContext::new().with_local_only_policy(),
        )
        .await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        policy.terminals.lock().unwrap().len(),
        1,
        "no hook activity for the local-only request"
    );
    assert_eq!(
        receipt_sink
            .calls
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "the receipt sink never observed the local-only request"
    );
    assert_eq!(
        proposal_source
            .calls
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "the proposal source never observed the local-only request"
    );
}

fn kv_resident_entries(kv: &crate::kv_integration::KvStageIntegration) -> u64 {
    kv.attrs()
        .iter()
        .find(|(key, _)| *key == "skippy.kv.resident_entries")
        .and_then(|(_, value)| value.as_u64())
        .unwrap_or(0)
}

/// Real decode on the production stage backend. Run explicitly:
///
/// ```text
/// SKIPPY_LOCAL_ONLY_TEST_MODEL=/path/qwen2.5-0.5b-instruct-q4_k_m.gguf \
/// SKIPPY_LOCAL_ONLY_TEST_RUNTIME_LIBS=/path/meshllm-native-runtime-.../lib \
/// SKIPPY_LOCAL_ONLY_TEST_MODEL_LAYERS=24 \
/// cargo test -p skippy-server --lib local_only_policy_real_decode -- --ignored --nocapture
/// ```
///
/// Ordinary requests with hooks enabled and disabled are positive controls:
/// they generate tokens and drive the hook policy, receipt sink, proposal
/// source, prefix KV cache and debug telemetry. The request under the local-only
/// policy must also generate tokens while every one of those observers stays
/// silent and debug telemetry is capped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs SKIPPY_LOCAL_ONLY_TEST_MODEL and SKIPPY_LOCAL_ONLY_TEST_RUNTIME_LIBS"]
async fn local_only_policy_real_decode_suppresses_observers_that_fire_on_ordinary_requests() {
    let fixture::RealDecodeFixture {
        backend,
        policy,
        receipt_sink,
        proposal_source,
        kv,
        captured_telemetry,
        request,
    } = fixture::real_decode_fixture(false);
    let mut captured_telemetry = captured_telemetry.expect("collector capture configured");

    // Positive control: the ordinary path generates and every observer fires.
    let kv_before = kv_resident_entries(&kv);
    let ordinary = backend
        .chat_completion_with_context(request.clone(), OpenAiRequestContext::new())
        .await
        .expect("ordinary generation succeeds");
    let ordinary_tokens = ordinary.usage.completion_tokens;
    assert!(ordinary_tokens >= 1, "ordinary request generated tokens");
    assert!(
        ordinary.choices[0]
            .message
            .content
            .as_deref()
            .is_some_and(|content| !content.trim().is_empty()),
        "ordinary request produced content"
    );
    wait_for_hook_terminal(&policy).await;
    assert_eq!(
        policy.terminals.lock().unwrap().len(),
        1,
        "hook saw the ordinary request"
    );
    let ordinary_events = captured_telemetry();
    assert!(
        ordinary_events
            .iter()
            .any(|attributes| attributes.contains_key("llama_stage.predicted_token")),
        "ordinary debug telemetry contains generated token IDs"
    );
    // Generation hooks intentionally disable the proposal lane. Exercise that
    // independent observer with hooks disabled instead of a vacuous zero count.
    let mut proposal_request = request.clone();
    set_chat_mesh_hooks_enabled(&mut proposal_request, false);
    let proposal_control = backend
        .chat_completion_with_context(proposal_request, OpenAiRequestContext::new())
        .await
        .expect("proposal-lane ordinary generation succeeds");
    assert!(proposal_control.usage.completion_tokens > 0);
    let receipt_calls_after_ordinary = receipt_sink
        .calls
        .load(std::sync::atomic::Ordering::Relaxed);
    let proposal_calls_after_ordinary = proposal_source
        .calls
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        receipt_calls_after_ordinary > 0,
        "receipt sink observed ordinary generation"
    );
    assert!(
        proposal_calls_after_ordinary > 0,
        "proposal source observed ordinary generation"
    );
    let kv_after_ordinary = kv_resident_entries(&kv);
    assert!(
        kv_after_ordinary > kv_before,
        "prefix KV cache recorded the ordinary request ({kv_before} -> {kv_after_ordinary})"
    );

    // Local-only: same request, real generation, silent observers.
    captured_telemetry();
    let local = backend
        .chat_completion_with_context(
            request.clone(),
            OpenAiRequestContext::new().with_local_only_policy(),
        )
        .await
        .expect("local-only generation succeeds");
    let local_tokens = local.usage.completion_tokens;
    assert!(local_tokens >= 1, "local-only request generated tokens");
    assert!(
        local.choices[0]
            .message
            .content
            .as_deref()
            .is_some_and(|content| !content.trim().is_empty()),
        "local-only request produced content"
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        policy.terminals.lock().unwrap().len(),
        1,
        "the hook policy never saw the local-only request"
    );
    assert_eq!(
        receipt_sink
            .calls
            .load(std::sync::atomic::Ordering::Relaxed),
        receipt_calls_after_ordinary,
        "the receipt sink never saw the local-only request"
    );
    assert_eq!(
        proposal_source
            .calls
            .load(std::sync::atomic::Ordering::Relaxed),
        proposal_calls_after_ordinary,
        "the proposal source never saw the local-only request"
    );
    assert_eq!(
        kv_resident_entries(&kv),
        kv_after_ordinary,
        "the local-only request did not add retained prefix cache entries"
    );
    let local_events = captured_telemetry();
    assert!(
        local_events
            .iter()
            .all(|attributes| !attributes.contains_key("skippy.scheduler.generation_driver")),
        "local-only telemetry must suppress debug records even on the built-in decode path"
    );
    assert!(
        local_events
            .iter()
            .all(|attributes| !attributes.contains_key("llama_stage.predicted_token")),
        "local-only telemetry must never contain generated token IDs"
    );
    assert_eq!(
        local
            .usage
            .prompt_tokens_details
            .as_ref()
            .map(|details| details.cached_tokens)
            .unwrap_or(0),
        0,
        "local-only request must not consume a stored prompt prefix"
    );
}
