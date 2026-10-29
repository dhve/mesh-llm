use super::test_support::{Behaviour, FakeBackend, response};
use super::*;
use openai_frontend::{Usage, chat_mesh_hooks_enabled};
use std::time::Duration;

fn limits(max_completion_tokens: u32) -> LocalCompletionLimits {
    LocalCompletionLimits {
        max_completion_tokens,
        max_prompt_bytes: 4096,
        max_completion_bytes: 4096,
        deadline: None,
        temperature: 0.0,
        seed: None,
    }
}

fn sample_request() -> LocalCompletionRequest {
    LocalCompletionRequest {
        model: LocalModelSelector::InstanceId("embedded-1".into()),
        messages: vec![
            LocalCompletionMessage {
                role: LocalCompletionRole::System,
                content: "Reply with one word.".into(),
            },
            LocalCompletionMessage {
                role: LocalCompletionRole::User,
                content: "ping".into(),
            },
        ],
        limits: limits(16),
        response_format: None,
        request_label: Some("test".into()),
    }
}

fn identity() -> LocalCompletionIdentity {
    LocalCompletionIdentity {
        model_id: "qwen".into(),
        instance_id: "embedded-1".into(),
    }
}

fn make_execution(
    backend: Arc<FakeBackend>,
    request: LocalCompletionRequest,
) -> (LocalCompletionExecution, CancellationToken) {
    let cancellation = CancellationToken::new();
    (
        LocalCompletionExecution {
            backend,
            identity: identity(),
            request,
            cancellation: cancellation.clone(),
            admitted_at: Instant::now(),
        },
        cancellation,
    )
}

#[test]
fn build_chat_request_disables_hooks_tools_cache_and_streaming() {
    let chat = build_chat_request("qwen", &sample_request());
    assert_eq!(chat.model, "qwen");
    assert!(
        !chat_mesh_hooks_enabled(&chat),
        "empty extra keeps mesh hooks off"
    );
    assert!(chat.extra.is_empty());
    assert!(!chat.stream);
    assert_eq!(chat.effective_max_tokens(), Some(16));
    assert_eq!(chat.max_tokens, None);
    assert_eq!(chat.temperature, Some(0.0));
    assert!(chat.tools.is_none() && chat.tool_choice.is_none());
    assert!(chat.prompt_cache_key.is_none() && chat.prompt_cache_retention.is_none());
    assert!(chat.stream_options.is_none() && chat.stop.is_none());
    assert_eq!(chat.messages.len(), 2);
    assert_eq!(chat.messages[0].role, "system");
    assert_eq!(
        chat.messages[1].content,
        Some(MessageContent::Text("ping".into()))
    );
    assert!(chat.messages.iter().all(|message| message.extra.is_empty()));
    chat.validate()
        .expect("the built request is a valid chat request");
}

#[test]
fn validate_rejects_virtual_models_empty_prompts_zero_budgets_and_oversize_prompts() {
    let mut virtual_model = sample_request();
    virtual_model.model = LocalModelSelector::ModelId("auto".into());
    assert!(matches!(
        validate_local_completion_request(&virtual_model),
        Err(LocalCompletionError::InvalidRequest(_))
    ));
    let mut mesh = sample_request();
    mesh.model = LocalModelSelector::ModelId("mesh".into());
    assert!(validate_local_completion_request(&mesh).is_err());

    let mut empty = sample_request();
    empty.messages.clear();
    assert!(validate_local_completion_request(&empty).is_err());

    let mut zero_budget = sample_request();
    zero_budget.limits.max_completion_tokens = 0;
    assert!(validate_local_completion_request(&zero_budget).is_err());

    let mut oversize = sample_request();
    oversize.limits.max_prompt_bytes = 4;
    assert!(validate_local_completion_request(&oversize).is_err());

    let mut hot = sample_request();
    hot.limits.temperature = f32::NAN;
    assert!(validate_local_completion_request(&hot).is_err());

    let mut long_label = sample_request();
    long_label.request_label = Some("x".repeat(MAX_REQUEST_LABEL_BYTES + 1));
    assert!(validate_local_completion_request(&long_label).is_err());

    validate_local_completion_request(&sample_request()).expect("the baseline request is valid");
}

#[test]
fn reduce_rejects_tool_calls_even_with_content_and_finish_stop() {
    let mut with_tools = response("ok", Usage::new(5, 1), FinishReason::Stop);
    with_tools.choices[0].message.tool_calls =
        Some(serde_json::json!([{ "id": "call_1", "type": "function" }]));
    assert_eq!(
        reduce_chat_response(&with_tools, &identity(), &limits(16), Duration::ZERO),
        Err(LocalCompletionError::UnsupportedResponseShape(
            LocalCompletionResponseRejection::ToolCalls
        ))
    );
    // An explicit empty array and an explicit null are not tool calls.
    let mut empty_tools = response("ok", Usage::new(5, 1), FinishReason::Stop);
    empty_tools.choices[0].message.tool_calls = Some(serde_json::json!([]));
    assert!(reduce_chat_response(&empty_tools, &identity(), &limits(16), Duration::ZERO).is_ok());
}

#[test]
fn reduce_rejects_tool_call_and_content_filter_finish_reasons() {
    let tool_finish = response("ok", Usage::new(5, 1), FinishReason::ToolCalls);
    assert_eq!(
        reduce_chat_response(&tool_finish, &identity(), &limits(16), Duration::ZERO),
        Err(LocalCompletionError::UnsupportedResponseShape(
            LocalCompletionResponseRejection::FinishToolCalls
        ))
    );
    let filtered = response("ok", Usage::new(5, 1), FinishReason::ContentFilter);
    assert_eq!(
        reduce_chat_response(&filtered, &identity(), &limits(16), Duration::ZERO),
        Err(LocalCompletionError::UnsupportedResponseShape(
            LocalCompletionResponseRejection::ContentFilter
        ))
    );
}

#[test]
fn reduce_rejects_empty_content_and_bounds_output_without_truncating() {
    let empty = response("", Usage::new(5, 0), FinishReason::Stop);
    assert_eq!(
        reduce_chat_response(&empty, &identity(), &limits(16), Duration::ZERO),
        Err(LocalCompletionError::UnsupportedResponseShape(
            LocalCompletionResponseRejection::EmptyContent
        ))
    );
    let big = response(&"y".repeat(5000), Usage::new(5, 16), FinishReason::Stop);
    assert_eq!(
        reduce_chat_response(&big, &identity(), &limits(16), Duration::ZERO),
        Err(LocalCompletionError::OutputTooLarge { max_bytes: 4096 })
    );
}

#[test]
fn reduce_reports_clamped_budget_as_context_budget_exceeded() {
    // finish=length after only 10 of the requested 16 tokens means the backend
    // clamped the budget to the context: never accepted as a complete reply.
    let clamped = response("partial", Usage::new(5, 10), FinishReason::Length);
    assert_eq!(
        reduce_chat_response(&clamped, &identity(), &limits(16), Duration::ZERO),
        Err(LocalCompletionError::ContextBudgetExceeded)
    );
    let full_budget = response("complete", Usage::new(5, 16), FinishReason::Length);
    let reduced = reduce_chat_response(&full_budget, &identity(), &limits(16), Duration::ZERO)
        .expect("a reply that used exactly its budget is complete");
    assert_eq!(reduced.finish_reason, LocalCompletionFinish::Length);
    assert_eq!(reduced.completion_tokens, 16);
}

#[test]
fn reduce_counts_discarded_reasoning_and_keeps_identity() {
    let mut reasoned = response("answer", Usage::new(7, 3), FinishReason::Stop);
    reasoned.choices[0].message.reasoning_content = Some("thinking...".into());
    let reduced = reduce_chat_response(
        &reasoned,
        &identity(),
        &limits(16),
        Duration::from_millis(5),
    )
    .expect("plain reply");
    assert_eq!(reduced.content, "answer");
    assert_eq!(reduced.reasoning_bytes_discarded, "thinking...".len());
    assert_eq!(reduced.model_id, "qwen");
    assert_eq!(reduced.instance_id, "embedded-1");
    assert_eq!(reduced.prompt_tokens, 7);
    assert_eq!(reduced.finish_reason, LocalCompletionFinish::Stop);
}

#[test]
fn classify_backend_error_never_forwards_messages() {
    let selector = LocalModelSelector::ModelId("qwen".into());
    let secret = OpenAiError::backend("prompt text leaked: top secret payload");
    let classified = classify_backend_error(&secret, &selector, false);
    assert_eq!(
        classified,
        LocalCompletionError::Backend {
            status: 502,
            kind: LocalCompletionBackendErrorKind::ServiceUnavailable,
        }
    );
    assert!(!format!("{classified:?}").contains("secret"));
    assert_eq!(
        classify_backend_error(
            &OpenAiError::context_length_exceeded("too long"),
            &selector,
            false
        ),
        LocalCompletionError::ContextBudgetExceeded
    );
    assert_eq!(
        classify_backend_error(&OpenAiError::timeout("queue"), &selector, false),
        LocalCompletionError::AdmissionTimeout
    );
    assert_eq!(
        classify_backend_error(&OpenAiError::cancelled("x"), &selector, true),
        LocalCompletionError::Cancelled
    );
    assert!(matches!(
        classify_backend_error(&OpenAiError::cancelled("x"), &selector, false),
        LocalCompletionError::Backend {
            kind: LocalCompletionBackendErrorKind::Cancelled,
            ..
        }
    ));
    assert_eq!(
        classify_backend_error(&OpenAiError::model_not_found("qwen"), &selector, false),
        LocalCompletionError::ModelNotLoadedLocally {
            model: "qwen".into()
        }
    );
}

#[test]
fn load_shape_follows_the_backend_topology_not_the_constructor() {
    let whole = LocalCompletionCapability {
        single_stage: true,
        receipts_configured: false,
        kv_cache_configured: false,
        honours_request_context: true,
        local_only_policy_supported: true,
    };
    assert_eq!(
        LocalModelLoadShape::for_capability(&whole),
        LocalModelLoadShape::SingleStage,
        "a whole package model loaded through the stage-zero loader is single stage"
    );
    let staged = LocalCompletionCapability {
        single_stage: false,
        ..whole
    };
    assert_eq!(
        LocalModelLoadShape::for_capability(&staged),
        LocalModelLoadShape::SplitStageZero
    );
}

#[test]
fn validate_bounds_the_selector_that_errors_echo() {
    let mut huge = sample_request();
    huge.model = LocalModelSelector::ModelId("m".repeat(MAX_MODEL_SELECTOR_BYTES + 1));
    let error = validate_local_completion_request(&huge).expect_err("oversize selector");
    assert!(
        !format!("{error}").contains("mmmm"),
        "the selector itself is not echoed"
    );
    let mut fits = sample_request();
    fits.model = LocalModelSelector::ModelId("m".repeat(MAX_MODEL_SELECTOR_BYTES));
    validate_local_completion_request(&fits).expect("selector at the bound is accepted");
}

#[test]
fn eligibility_table() {
    let capability = LocalCompletionCapability {
        single_stage: true,
        receipts_configured: false,
        kv_cache_configured: false,
        honours_request_context: true,
        local_only_policy_supported: true,
    };
    assert_eq!(
        eligibility_for(
            LocalModelLoadShape::SingleStage,
            "skippy",
            "s0",
            &capability
        ),
        LocalCompletionEligibility::Eligible
    );
    assert!(matches!(
        eligibility_for(
            LocalModelLoadShape::SplitStageZero,
            "skippy",
            "s0",
            &capability
        ),
        LocalCompletionEligibility::Ineligible(LocalCompletionIneligibility::SplitStage { .. })
    ));
    assert!(matches!(
        eligibility_for(
            LocalModelLoadShape::SingleStage,
            "llama-server",
            "s0",
            &capability
        ),
        LocalCompletionEligibility::Ineligible(LocalCompletionIneligibility::NotSkippy { .. })
    ));
    // Receipts and a KV cache stay eligible when the backend suppresses them
    // per request, and become ineligible when it cannot.
    let configured = LocalCompletionCapability {
        receipts_configured: true,
        kv_cache_configured: true,
        ..capability
    };
    assert_eq!(
        eligibility_for(
            LocalModelLoadShape::SingleStage,
            "skippy",
            "s0",
            &configured
        ),
        LocalCompletionEligibility::Eligible
    );
    let unsupported = LocalCompletionCapability {
        local_only_policy_supported: false,
        ..configured
    };
    assert_eq!(
        eligibility_for(
            LocalModelLoadShape::SingleStage,
            "skippy",
            "s0",
            &unsupported
        ),
        LocalCompletionEligibility::Ineligible(LocalCompletionIneligibility::ReceiptsEnabled)
    );
    let ignores = LocalCompletionCapability {
        honours_request_context: false,
        ..capability
    };
    assert_eq!(
        eligibility_for(LocalModelLoadShape::SingleStage, "skippy", "s0", &ignores),
        LocalCompletionEligibility::Ineligible(
            LocalCompletionIneligibility::BackendIgnoresCancellation
        )
    );
}

#[tokio::test]
async fn execute_passes_cancellation_token_and_local_only_policy_to_the_backend() {
    let backend = FakeBackend::new(Behaviour::Respond(response(
        "pong",
        Usage::new(9, 1),
        FinishReason::Stop,
    )));
    let (execution, _cancellation) = make_execution(Arc::clone(&backend), sample_request());
    let reduced = execute_local_completion(execution)
        .await
        .expect("completed");
    assert_eq!(reduced.content, "pong");
    assert!(backend.seen_local_only.load(Ordering::Acquire));
    let seen = backend
        .seen_request
        .lock()
        .expect("seen lock")
        .clone()
        .expect("backend saw the request");
    assert!(seen.extra.is_empty());
    assert_eq!(seen.effective_max_tokens(), Some(16));
}

#[tokio::test]
async fn execute_reports_cancelled_only_after_the_backend_returns() {
    let backend = FakeBackend::new(Behaviour::HonourCancellation);
    let (execution, cancellation) = make_execution(Arc::clone(&backend), sample_request());
    let task = tokio::spawn(execute_local_completion(execution));
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!task.is_finished(), "no termination before cancellation");
    cancellation.cancel();
    let outcome = task.await.expect("supervisor task");
    assert_eq!(outcome, Err(LocalCompletionError::Cancelled));
    assert!(backend.returned.load(Ordering::Acquire));
}

#[tokio::test]
async fn execute_waits_for_a_backend_that_ignores_cancellation() {
    let backend = FakeBackend::new(Behaviour::IgnoreCancellationUntilReleased);
    let (execution, cancellation) = make_execution(Arc::clone(&backend), sample_request());
    let task = tokio::spawn(execute_local_completion(execution));
    cancellation.cancel();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !task.is_finished(),
        "cancellation alone must not produce a terminal result while the backend still runs"
    );
    backend.release.notify_one();
    let outcome = task.await.expect("supervisor task");
    // The backend returned content after the caller cancelled: still Cancelled.
    assert_eq!(outcome, Err(LocalCompletionError::Cancelled));
    assert!(backend.returned.load(Ordering::Acquire));
}

#[tokio::test]
async fn execute_deadline_cancels_and_still_waits_for_termination() {
    let honouring = FakeBackend::new(Behaviour::HonourCancellation);
    let mut quick = sample_request();
    quick.limits.deadline = Some(Duration::from_millis(30));
    let (execution, cancellation) = make_execution(Arc::clone(&honouring), quick.clone());
    let outcome = execute_local_completion(execution).await;
    assert_eq!(outcome, Err(LocalCompletionError::DeadlineExceeded));
    assert!(cancellation.is_cancelled());
    assert!(honouring.returned.load(Ordering::Acquire));

    let ignoring = FakeBackend::new(Behaviour::IgnoreCancellationUntilReleased);
    let (execution, _cancellation) = make_execution(Arc::clone(&ignoring), quick);
    let task = tokio::spawn(execute_local_completion(execution));
    tokio::time::sleep(Duration::from_millis(80)).await;
    assert!(
        !task.is_finished(),
        "deadline must not report before termination"
    );
    ignoring.release.notify_one();
    assert_eq!(
        task.await.expect("supervisor task"),
        Err(LocalCompletionError::DeadlineExceeded)
    );
}

#[tokio::test]
async fn execute_classifies_backend_failures_without_messages() {
    let backend = FakeBackend::new(Behaviour::Fail(OpenAiError::context_length_exceeded(
        "prompt of 9000 tokens does not fit",
    )));
    let (execution, _cancellation) = make_execution(backend, sample_request());
    assert_eq!(
        execute_local_completion(execution).await,
        Err(LocalCompletionError::ContextBudgetExceeded)
    );
}

#[tokio::test]
async fn leases_drain_only_when_every_lease_is_released_and_cancel_all_reaches_tokens() {
    let leases = LocalCompletionLeases::new();
    let first_token = CancellationToken::new();
    let second_token = CancellationToken::new();
    let first = leases.begin(first_token.clone());
    let second = leases.begin(second_token.clone());
    assert_eq!(leases.in_flight(), 2);
    leases.cancel_all();
    assert!(first_token.is_cancelled() && second_token.is_cancelled());

    let waiter = {
        let leases = Arc::clone(&leases);
        tokio::spawn(async move { leases.drained().await })
    };
    drop(first);
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!waiter.is_finished(), "one lease still in flight");
    drop(second);
    tokio::time::timeout(Duration::from_secs(1), waiter)
        .await
        .expect("drained after the last release")
        .expect("waiter task");
    assert_eq!(leases.in_flight(), 0);
}

#[tokio::test]
async fn job_wait_until_is_non_consuming_and_wait_returns_the_same_outcome() {
    let (tx, rx) = oneshot::channel();
    let cancellation = CancellationToken::new();
    let mut job = LocalCompletionJob::new(
        cancellation.clone(),
        LocalCompletionAdmitted {
            instance_id: "embedded-1".into(),
            model_id: "qwen".into(),
            outcome: rx,
        },
    );
    assert!(
        job.wait_until(tokio::time::Instant::now() + Duration::from_millis(10))
            .await
            .is_none()
    );
    tx.send(Err(LocalCompletionError::DeadlineExceeded))
        .expect("outcome delivered");
    assert_eq!(
        job.wait_until(tokio::time::Instant::now() + Duration::from_secs(1))
            .await,
        Some(Err(LocalCompletionError::DeadlineExceeded))
    );
    assert_eq!(
        job.wait().await,
        Err(LocalCompletionError::DeadlineExceeded)
    );
}

#[tokio::test]
async fn dropping_the_outcome_sender_is_reported_as_response_dropped() {
    let (tx, rx) = oneshot::channel::<LocalCompletionOutcome>();
    let job = LocalCompletionJob::new(
        CancellationToken::new(),
        LocalCompletionAdmitted {
            instance_id: "embedded-1".into(),
            model_id: "qwen".into(),
            outcome: rx,
        },
    );
    drop(tx);
    assert_eq!(job.wait().await, Err(LocalCompletionError::ResponseDropped));
}
