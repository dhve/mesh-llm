//! Local-only, single-stage chat completion against a model this runtime
//! serves in process.
//!
//! The operation never consults mesh routing targets, the OpenAI ingress,
//! the Mixture-of-Agents gateway, plugin hooks, guardrail or compaction
//! wrappers, or remote peers. It calls the loaded model's inner backend with
//! [`OpenAiRequestContext::with_local_only_policy`], which suppresses
//! generation receipts and prefix-cache use for that one request and makes
//! the completion budget strict. The caller holds a cancellation token, and
//! the reported terminal state is produced only after the backend future has
//! returned, so a dropped caller future is never mistaken for a finished
//! generation.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use openai_frontend::{
    CancellationToken, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, FinishReason,
    MessageContent, OpenAiBackend, OpenAiError, OpenAiErrorKind, OpenAiRequestContext,
};
pub use skippy_server::LocalCompletionCapability;
use tokio::sync::{Notify, oneshot};

/// Upper bound on `request_label` so diagnostics stay small.
pub const MAX_REQUEST_LABEL_BYTES: usize = 128;
/// Upper bound on a supervisor deadline.
pub const MAX_DEADLINE: Duration = Duration::from_secs(60 * 60);
/// Upper bound on a model selector, which errors echo back.
pub const MAX_MODEL_SELECTOR_BYTES: usize = 256;
/// Shutdown and unload wait for in-flight completions; this is how often the
/// blocking backstop logs that it is still waiting.
const DRAIN_WARN_INTERVAL: Duration = Duration::from_secs(5);
const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Which loaded model to run against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalModelSelector {
    /// Preferred. Exact [`LocalCompletionModel::instance_id`].
    InstanceId(String),
    /// Exact [`LocalCompletionModel::model_id`]. An id loaded as more than one
    /// instance is [`LocalCompletionError::Ambiguous`].
    ModelId(String),
}

impl LocalModelSelector {
    fn value(&self) -> &str {
        match self {
            Self::InstanceId(value) | Self::ModelId(value) => value,
        }
    }
}

/// Chat roles accepted by the local-only completion. Tool and function
/// messages are deliberately not representable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCompletionRole {
    System,
    User,
    Assistant,
}

impl LocalCompletionRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// One chat message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalCompletionMessage {
    pub role: LocalCompletionRole,
    pub content: String,
}

/// Bounds applied to one completion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalCompletionLimits {
    /// Sent as `max_completion_tokens`. Must be greater than zero. The
    /// backend reserves `prompt_tokens + max_completion_tokens` against the
    /// model context and rejects the request when it does not fit. Nothing is
    /// truncated.
    pub max_completion_tokens: u32,
    /// Sum of UTF-8 bytes over all message contents, checked before model
    /// resolution. Must be greater than zero.
    pub max_prompt_bytes: usize,
    /// Upper bound on `content` bytes the caller accepts. A larger completion
    /// is reported as [`LocalCompletionError::OutputTooLarge`], never cut.
    pub max_completion_bytes: usize,
    /// Optional supervisor deadline measured from admission. On expiry the
    /// supervisor cancels the backend and keeps waiting for termination, then
    /// reports [`LocalCompletionError::DeadlineExceeded`].
    pub deadline: Option<Duration>,
    /// Finite, within `[0.0, 2.0]`.
    pub temperature: f32,
    pub seed: Option<u64>,
}

/// A completion request.
#[derive(Clone, Debug)]
pub struct LocalCompletionRequest {
    pub model: LocalModelSelector,
    /// At least one message.
    pub messages: Vec<LocalCompletionMessage>,
    pub limits: LocalCompletionLimits,
    /// Passed through as OpenAI `response_format`. The default backend parses
    /// it but does not enforce it; callers validate output themselves.
    pub response_format: Option<serde_json::Value>,
    /// Metadata only, for diagnostics. Never enters the prompt. At most
    /// [`MAX_REQUEST_LABEL_BYTES`].
    pub request_label: Option<String>,
}

/// A locally loaded model instance and whether it may serve local-only
/// completions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalCompletionModel {
    pub instance_id: String,
    pub model_id: String,
    pub model_ref: String,
    pub profile: String,
    pub context_length: u32,
    pub backend: String,
    pub eligibility: LocalCompletionEligibility,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalCompletionEligibility {
    Eligible,
    Ineligible(LocalCompletionIneligibility),
}

/// Why a loaded model may not serve local-only completions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalCompletionIneligibility {
    /// Stage zero of a split model; activations would cross peers.
    SplitStage { stage_id: String },
    /// A generation receipt sink is configured and the backend cannot
    /// suppress it per request.
    ReceiptsEnabled,
    /// A prefix KV cache is configured and the backend cannot bypass it per
    /// request.
    KvCachePersistence { mode: String },
    /// The backend does not observe request cancellation.
    BackendIgnoresCancellation,
    /// Not the embedded skippy backend.
    NotSkippy { backend: String },
}

/// How the model stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalCompletionFinish {
    /// The model emitted end of generation.
    Stop,
    /// `max_completion_tokens` was consumed. The output is complete up to the
    /// budget and was not truncated by the SDK.
    Length,
    Other(String),
}

/// A reduced, bounded completion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalCompletionResponse {
    pub model_id: String,
    pub instance_id: String,
    /// Bounded by [`LocalCompletionLimits::max_completion_bytes`].
    pub content: String,
    pub finish_reason: LocalCompletionFinish,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Bytes of `reasoning_content` the backend returned and the SDK did not
    /// expose. Zero when the model produced none.
    pub reasoning_bytes_discarded: usize,
    /// Measured from admission to backend termination.
    pub elapsed: Duration,
}

/// Closed classification of a backend failure. Carries no backend message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCompletionBackendErrorKind {
    InvalidRequest,
    Authentication,
    Permission,
    NotFound,
    RateLimit,
    PayloadTooLarge,
    Timeout,
    Internal,
    ServiceUnavailable,
    ContextLengthExceeded,
    UnsupportedFeature,
    Cancelled,
    Unknown,
}

impl From<Option<OpenAiErrorKind>> for LocalCompletionBackendErrorKind {
    fn from(kind: Option<OpenAiErrorKind>) -> Self {
        match kind {
            Some(OpenAiErrorKind::InvalidRequest) => Self::InvalidRequest,
            Some(OpenAiErrorKind::Authentication) => Self::Authentication,
            Some(OpenAiErrorKind::Permission) => Self::Permission,
            Some(OpenAiErrorKind::NotFound) => Self::NotFound,
            Some(OpenAiErrorKind::RateLimit) => Self::RateLimit,
            Some(OpenAiErrorKind::PayloadTooLarge) => Self::PayloadTooLarge,
            Some(OpenAiErrorKind::Timeout) => Self::Timeout,
            Some(OpenAiErrorKind::Internal) => Self::Internal,
            Some(OpenAiErrorKind::ServiceUnavailable) => Self::ServiceUnavailable,
            Some(OpenAiErrorKind::ContextLengthExceeded) => Self::ContextLengthExceeded,
            Some(OpenAiErrorKind::UnsupportedFeature) => Self::UnsupportedFeature,
            Some(OpenAiErrorKind::Cancelled) => Self::Cancelled,
            None => Self::Unknown,
        }
    }
}

/// Why a syntactically successful backend response was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalCompletionResponseRejection {
    /// The response carried a non-empty `tool_calls` value.
    ToolCalls,
    /// The backend reported `finish_reason: tool_calls`.
    FinishToolCalls,
    /// The backend reported `finish_reason: content_filter`.
    ContentFilter,
    /// No choice or more than one choice.
    ChoiceCount { count: usize },
    /// The single choice had no content or empty content.
    EmptyContent,
    /// The message role was not `assistant`.
    UnexpectedRole,
}

/// Errors of the local-only completion path. Every variant is content free.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LocalCompletionError {
    #[error("invalid local completion request: {0}")]
    InvalidRequest(String),
    #[error("model is not loaded on this runtime: {model}")]
    ModelNotLoadedLocally { model: String },
    #[error("model id {model} matches {} loaded instances", .instances.len())]
    Ambiguous {
        model: String,
        instances: Vec<String>,
    },
    #[error("model {model} is not eligible for local-only completion: {reason:?}")]
    Ineligible {
        model: String,
        reason: LocalCompletionIneligibility,
    },
    #[error("runtime is not serving models")]
    NotServing,
    #[error("a local completion is already in flight")]
    Busy,
    #[error("runtime control channel is unavailable")]
    ControlUnavailable,
    #[error("runtime is shutting down")]
    ShuttingDown,
    #[error("backend admission timed out")]
    AdmissionTimeout,
    #[error("prompt plus completion budget does not fit the model context")]
    ContextBudgetExceeded,
    #[error("deadline exceeded; backend terminated")]
    DeadlineExceeded,
    #[error("cancelled; backend terminated")]
    Cancelled,
    #[error("completion exceeded {max_bytes} bytes")]
    OutputTooLarge { max_bytes: usize },
    #[error("backend response rejected: {0:?}")]
    UnsupportedResponseShape(LocalCompletionResponseRejection),
    #[error("backend error: status {status}, kind {kind:?}")]
    Backend {
        status: u16,
        kind: LocalCompletionBackendErrorKind,
    },
    #[error("runtime dropped the response before admission")]
    ResponseDropped,
    /// The supervising task ended without observing the backend return (it
    /// was aborted or panicked). The model lease and admission permits are
    /// kept held for the life of the runtime because termination is unknown.
    #[error("backend termination unknown; model lease kept held")]
    TerminationUnknown,
}

/// Terminal outcome of one completion.
pub type LocalCompletionOutcome = Result<LocalCompletionResponse, LocalCompletionError>;

/// Sent back over the control channel once a completion is admitted.
pub struct LocalCompletionAdmitted {
    pub instance_id: String,
    pub model_id: String,
    pub outcome: oneshot::Receiver<LocalCompletionOutcome>,
}

impl std::fmt::Debug for LocalCompletionAdmitted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalCompletionAdmitted")
            .field("instance_id", &self.instance_id)
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

/// Clone + Send + Sync cancellation handle for a UI timer or button.
#[derive(Clone, Debug)]
pub struct LocalCompletionCancel(CancellationToken);

impl LocalCompletionCancel {
    pub fn cancel(&self) {
        self.0.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
}

/// An admitted completion. Dropping the job cancels the generation; the
/// supervisor task that owns the admission permit and the model lease keeps
/// them until the backend has actually terminated.
pub struct LocalCompletionJob {
    cancellation: CancellationToken,
    instance_id: String,
    model_id: String,
    outcome: oneshot::Receiver<LocalCompletionOutcome>,
    settled: Option<LocalCompletionOutcome>,
}

impl std::fmt::Debug for LocalCompletionJob {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalCompletionJob")
            .field("instance_id", &self.instance_id)
            .field("model_id", &self.model_id)
            .field("cancelled", &self.cancellation.is_cancelled())
            .field("settled", &self.settled.is_some())
            .finish()
    }
}

impl LocalCompletionJob {
    pub(crate) fn new(cancellation: CancellationToken, admitted: LocalCompletionAdmitted) -> Self {
        Self {
            cancellation,
            instance_id: admitted.instance_id,
            model_id: admitted.model_id,
            outcome: admitted.outcome,
            settled: None,
        }
    }

    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn cancel_handle(&self) -> LocalCompletionCancel {
        LocalCompletionCancel(self.cancellation.clone())
    }

    /// Idempotent cooperative stop; the backend observes it between decode
    /// steps.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Resolves after the backend returned and the model lease and admission
    /// permit were released, for success, cancellation, deadline and backend
    /// errors alike, with two exceptions the caller must treat as
    /// "termination unknown": [`LocalCompletionError::TerminationUnknown`]
    /// (the supervising task ended without observing the backend return; the
    /// lease and permits are kept held) and, conservatively,
    /// [`LocalCompletionError::ResponseDropped`] (the runtime dropped the
    /// request before admission). A caller's unresolved-work guard stays held
    /// on both.
    pub async fn wait(mut self) -> LocalCompletionOutcome {
        if let Some(settled) = self.settled.take() {
            return settled;
        }
        (&mut self.outcome)
            .await
            .unwrap_or(Err(LocalCompletionError::ResponseDropped))
    }

    /// Bounded, non consuming poll for an unresolved-work guard. `None` means
    /// no terminal outcome yet. Once `Some`, the same value is returned by
    /// [`Self::wait`], with the same two exceptions documented there:
    /// `TerminationUnknown` and `ResponseDropped` do not prove that the
    /// backend terminated.
    pub async fn wait_until(
        &mut self,
        deadline: tokio::time::Instant,
    ) -> Option<LocalCompletionOutcome> {
        if let Some(settled) = &self.settled {
            return Some(settled.clone());
        }
        match tokio::time::timeout_at(deadline, &mut self.outcome).await {
            Ok(received) => {
                let outcome = received.unwrap_or(Err(LocalCompletionError::ResponseDropped));
                self.settled = Some(outcome.clone());
                Some(outcome)
            }
            Err(_) => None,
        }
    }
}

impl Drop for LocalCompletionJob {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

/// How a model was loaded, recorded at load time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalModelLoadShape {
    /// Whole model in this process.
    SingleStage,
    /// Stage zero of a split; activations cross peers.
    SplitStageZero,
}

impl LocalModelLoadShape {
    /// Derive the shape from the backend's own topology facts rather than
    /// from which constructor loaded it: the stage-zero loader is also used
    /// for whole package models whose config has no upstream or downstream.
    pub fn for_capability(capability: &LocalCompletionCapability) -> Self {
        if capability.single_stage {
            Self::SingleStage
        } else {
            Self::SplitStageZero
        }
    }
}

/// In-flight local completions on one loaded model. Owned by the model
/// handle; unload and shutdown cancel and drain through it before the native
/// runtime is released.
#[derive(Default)]
pub struct LocalCompletionLeases {
    in_flight: AtomicUsize,
    drained: Notify,
    active: Mutex<Vec<CancellationToken>>,
}

impl LocalCompletionLeases {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Register one in-flight completion. The lease releases on drop.
    pub fn begin(self: &Arc<Self>, cancellation: CancellationToken) -> LocalCompletionLease {
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        if let Ok(mut active) = self.active.lock() {
            active.push(cancellation.clone());
        }
        LocalCompletionLease {
            leases: Arc::clone(self),
            cancellation,
        }
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Acquire)
    }

    /// Request a cooperative stop of every in-flight completion.
    pub fn cancel_all(&self) {
        if let Ok(active) = self.active.lock() {
            for token in active.iter() {
                token.cancel();
            }
        }
    }

    /// Wait until no completion is in flight. Never times out: a backend that
    /// ignores cancellation keeps its lease, and the native handle must not be
    /// released underneath it.
    pub async fn drained(&self) {
        loop {
            if self.in_flight() == 0 {
                return;
            }
            let notified = self.drained.notified();
            if self.in_flight() == 0 {
                return;
            }
            notified.await;
        }
    }

    /// Cancel everything and wait for the drain.
    pub async fn cancel_and_drain(&self) {
        self.cancel_all();
        self.drained().await;
    }

    /// Blocking backstop for synchronous shutdown paths. Cancels, then polls
    /// until drained, logging while it waits.
    pub fn cancel_and_block_until_drained(&self) {
        self.cancel_all();
        let started = Instant::now();
        let mut last_warning = started;
        while self.in_flight() > 0 {
            std::thread::sleep(DRAIN_POLL_INTERVAL);
            if last_warning.elapsed() >= DRAIN_WARN_INTERVAL {
                last_warning = Instant::now();
                tracing::warn!(
                    in_flight = self.in_flight(),
                    waited_ms = started.elapsed().as_millis() as u64,
                    "waiting for local completions to terminate before releasing the model"
                );
            }
        }
    }

    fn release(&self, cancellation: &CancellationToken) {
        if let Ok(mut active) = self.active.lock()
            && let Some(index) = active.iter().position(|token| token.is_same(cancellation))
        {
            active.swap_remove(index);
        }
        let remaining = self
            .in_flight
            .fetch_sub(1, Ordering::AcqRel)
            .saturating_sub(1);
        if remaining == 0 {
            self.drained.notify_waiters();
        }
    }
}

/// One in-flight completion's lease on its model. Dropping releases it.
pub struct LocalCompletionLease {
    leases: Arc<LocalCompletionLeases>,
    cancellation: CancellationToken,
}

impl Drop for LocalCompletionLease {
    fn drop(&mut self) {
        self.leases.release(&self.cancellation);
    }
}

/// A backend plus the lease that keeps its model loaded.
pub struct LeasedLocalBackend {
    pub backend: Arc<dyn OpenAiBackend>,
    pub lease: LocalCompletionLease,
}

/// A loaded model instance as seen by the resolver. Implemented by the
/// runtime's model records and by test fakes.
pub trait LocalCompletionCandidate {
    fn instance_id(&self) -> &str;
    /// The id the backend answers to.
    fn model_id(&self) -> &str;
    fn model_ref(&self) -> &str;
    fn profile(&self) -> &str;
    fn context_length(&self) -> u32;
    fn backend_label(&self) -> &str;
    fn eligibility(&self) -> LocalCompletionEligibility;
    /// Lease the backend for one completion. `None` when the record has no
    /// in-process backend.
    fn lease_backend(&self, cancellation: CancellationToken) -> Option<LeasedLocalBackend>;

    fn describe(&self) -> LocalCompletionModel {
        LocalCompletionModel {
            instance_id: self.instance_id().to_string(),
            model_id: self.model_id().to_string(),
            model_ref: self.model_ref().to_string(),
            profile: self.profile().to_string(),
            context_length: self.context_length(),
            backend: self.backend_label().to_string(),
            eligibility: self.eligibility(),
        }
    }
}

/// Decide eligibility from load shape, backend label and capability facts.
pub fn eligibility_for(
    load_shape: LocalModelLoadShape,
    backend_label: &str,
    stage_id: &str,
    capability: &LocalCompletionCapability,
) -> LocalCompletionEligibility {
    use LocalCompletionIneligibility as Why;
    let reason = if backend_label != "skippy" {
        Some(Why::NotSkippy {
            backend: backend_label.to_string(),
        })
    } else if load_shape == LocalModelLoadShape::SplitStageZero || !capability.single_stage {
        Some(Why::SplitStage {
            stage_id: stage_id.to_string(),
        })
    } else if !capability.honours_request_context {
        Some(Why::BackendIgnoresCancellation)
    } else if capability.receipts_configured && !capability.local_only_policy_supported {
        Some(Why::ReceiptsEnabled)
    } else if capability.kv_cache_configured && !capability.local_only_policy_supported {
        Some(Why::KvCachePersistence {
            mode: "configured".to_string(),
        })
    } else {
        None
    };
    match reason {
        Some(reason) => LocalCompletionEligibility::Ineligible(reason),
        None => LocalCompletionEligibility::Eligible,
    }
}

/// Pick exactly one candidate for `selector`, or fail closed.
pub fn resolve_local_completion_target<'a, C>(
    candidates: impl IntoIterator<Item = &'a C>,
    selector: &LocalModelSelector,
) -> Result<&'a C, LocalCompletionError>
where
    C: LocalCompletionCandidate + ?Sized + 'a,
{
    let matches: Vec<&'a C> = candidates
        .into_iter()
        .filter(|candidate| match selector {
            LocalModelSelector::InstanceId(id) => candidate.instance_id() == id,
            LocalModelSelector::ModelId(id) => candidate.model_id() == id,
        })
        .collect();
    match matches.as_slice() {
        [] => Err(LocalCompletionError::ModelNotLoadedLocally {
            model: selector.value().to_string(),
        }),
        [single] => match single.eligibility() {
            LocalCompletionEligibility::Eligible => Ok(single),
            LocalCompletionEligibility::Ineligible(reason) => {
                Err(LocalCompletionError::Ineligible {
                    model: selector.value().to_string(),
                    reason,
                })
            }
        },
        many => Err(LocalCompletionError::Ambiguous {
            model: selector.value().to_string(),
            instances: many
                .iter()
                .map(|candidate| candidate.instance_id().to_string())
                .collect(),
        }),
    }
}

/// Validate a request before any model resolution or content handling.
pub fn validate_local_completion_request(
    request: &LocalCompletionRequest,
) -> Result<(), LocalCompletionError> {
    use LocalCompletionError::InvalidRequest;
    let selector = request.model.value();
    if selector.trim().is_empty() {
        return Err(InvalidRequest("model selector is empty".into()));
    }
    if selector.len() > MAX_MODEL_SELECTOR_BYTES {
        return Err(InvalidRequest(format!(
            "model selector exceeds {MAX_MODEL_SELECTOR_BYTES} bytes"
        )));
    }
    if matches!(selector.trim(), "auto" | "mesh") {
        return Err(InvalidRequest(
            "virtual model selectors are not allowed for local-only completion".into(),
        ));
    }
    if request.messages.is_empty() {
        return Err(InvalidRequest("messages is empty".into()));
    }
    let limits = &request.limits;
    if limits.max_completion_tokens == 0 {
        return Err(InvalidRequest("max_completion_tokens must be > 0".into()));
    }
    if limits.max_prompt_bytes == 0 || limits.max_completion_bytes == 0 {
        return Err(InvalidRequest("byte limits must be > 0".into()));
    }
    if !limits.temperature.is_finite() || !(0.0..=2.0).contains(&limits.temperature) {
        return Err(InvalidRequest(
            "temperature must be finite within [0, 2]".into(),
        ));
    }
    if limits
        .deadline
        .is_some_and(|deadline| deadline.is_zero() || deadline > MAX_DEADLINE)
    {
        return Err(InvalidRequest(format!(
            "deadline must be within (0, {}s] when set",
            MAX_DEADLINE.as_secs()
        )));
    }
    let prompt_bytes: usize = request
        .messages
        .iter()
        .map(|message| message.content.len())
        .sum();
    if prompt_bytes > limits.max_prompt_bytes {
        return Err(InvalidRequest(format!(
            "prompt is {prompt_bytes} bytes; limit is {}",
            limits.max_prompt_bytes
        )));
    }
    if request
        .request_label
        .as_ref()
        .is_some_and(|label| label.len() > MAX_REQUEST_LABEL_BYTES)
    {
        return Err(InvalidRequest(format!(
            "request_label exceeds {MAX_REQUEST_LABEL_BYTES} bytes"
        )));
    }
    Ok(())
}

/// Build the backend request. Every optional OpenAI field is `None` and
/// `extra` is empty, which keeps mesh hooks disabled.
pub fn build_chat_request(
    model_id: &str,
    request: &LocalCompletionRequest,
) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model_id.to_string(),
        messages: request
            .messages
            .iter()
            .map(|message| ChatMessage {
                role: message.role.as_str().to_string(),
                content: Some(MessageContent::Text(message.content.clone())),
                extra: BTreeMap::new(),
            })
            .collect(),
        stream: false,
        max_tokens: None,
        max_completion_tokens: Some(request.limits.max_completion_tokens),
        temperature: Some(request.limits.temperature),
        top_p: None,
        n: None,
        logprobs: None,
        top_logprobs: None,
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        response_format: request.response_format.clone(),
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        user: None,
        stop: None,
        seed: request.limits.seed,
        reasoning: None,
        reasoning_effort: None,
        prompt_cache_key: None,
        prompt_cache_retention: None,
        stream_options: None,
        extra: BTreeMap::new(),
    }
}

/// Identity of the leased target, carried into the response.
#[derive(Clone, Debug)]
pub struct LocalCompletionIdentity {
    pub model_id: String,
    pub instance_id: String,
}

/// Reduce a backend response to the bounded shape, rejecting any metadata the
/// fixed response type cannot expose.
pub fn reduce_chat_response(
    response: &ChatCompletionResponse,
    identity: &LocalCompletionIdentity,
    limits: &LocalCompletionLimits,
    elapsed: Duration,
) -> Result<LocalCompletionResponse, LocalCompletionError> {
    use LocalCompletionError::UnsupportedResponseShape as Rejected;
    use LocalCompletionResponseRejection as Why;
    let [choice] = response.choices.as_slice() else {
        return Err(Rejected(Why::ChoiceCount {
            count: response.choices.len(),
        }));
    };
    if choice.message.role != "assistant" {
        return Err(Rejected(Why::UnexpectedRole));
    }
    if tool_calls_present(choice.message.tool_calls.as_ref()) {
        return Err(Rejected(Why::ToolCalls));
    }
    let finish_reason = match choice.finish_reason.as_ref() {
        Some(FinishReason::ToolCalls) => return Err(Rejected(Why::FinishToolCalls)),
        Some(FinishReason::ContentFilter) => return Err(Rejected(Why::ContentFilter)),
        Some(FinishReason::Stop) => LocalCompletionFinish::Stop,
        Some(FinishReason::Length) => LocalCompletionFinish::Length,
        None => LocalCompletionFinish::Other("none".to_string()),
    };
    let content = choice
        .message
        .content
        .as_deref()
        .filter(|content| !content.is_empty())
        .ok_or(Rejected(Why::EmptyContent))?;
    if content.len() > limits.max_completion_bytes {
        return Err(LocalCompletionError::OutputTooLarge {
            max_bytes: limits.max_completion_bytes,
        });
    }
    if finish_reason == LocalCompletionFinish::Length
        && response.usage.completion_tokens < limits.max_completion_tokens
    {
        // The backend stopped for length before the requested budget: the
        // budget was clamped to the context, which this path never allows.
        return Err(LocalCompletionError::ContextBudgetExceeded);
    }
    Ok(LocalCompletionResponse {
        model_id: identity.model_id.clone(),
        instance_id: identity.instance_id.clone(),
        content: content.to_string(),
        finish_reason,
        prompt_tokens: response.usage.prompt_tokens,
        completion_tokens: response.usage.completion_tokens,
        reasoning_bytes_discarded: choice
            .message
            .reasoning_content
            .as_ref()
            .map_or(0, String::len),
        elapsed,
    })
}

fn tool_calls_present(tool_calls: Option<&serde_json::Value>) -> bool {
    match tool_calls {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Array(items)) => !items.is_empty(),
        Some(_) => true,
    }
}

/// Map a backend error to the closed error set.
pub fn classify_backend_error(
    error: &OpenAiError,
    selector: &LocalModelSelector,
    cancelled_by_caller: bool,
) -> LocalCompletionError {
    match error.kind() {
        Some(OpenAiErrorKind::Cancelled) if cancelled_by_caller => LocalCompletionError::Cancelled,
        Some(OpenAiErrorKind::Timeout) => LocalCompletionError::AdmissionTimeout,
        Some(OpenAiErrorKind::ContextLengthExceeded) => LocalCompletionError::ContextBudgetExceeded,
        Some(OpenAiErrorKind::NotFound) => LocalCompletionError::ModelNotLoadedLocally {
            model: selector.value().to_string(),
        },
        kind => LocalCompletionError::Backend {
            status: error.status().as_u16(),
            kind: kind.into(),
        },
    }
}

/// Everything the supervisor needs to run one admitted completion.
pub struct LocalCompletionExecution {
    pub backend: Arc<dyn OpenAiBackend>,
    pub identity: LocalCompletionIdentity,
    pub request: LocalCompletionRequest,
    pub cancellation: CancellationToken,
    pub admitted_at: Instant,
}

/// Run one admitted completion to termination.
///
/// The returned future resolves only after the backend future returned. On
/// deadline the cancellation token is set and the backend future is still
/// awaited; the result is then [`LocalCompletionError::DeadlineExceeded`].
pub async fn execute_local_completion(
    execution: LocalCompletionExecution,
) -> LocalCompletionOutcome {
    let LocalCompletionExecution {
        backend,
        identity,
        request,
        cancellation,
        admitted_at,
    } = execution;
    let chat = build_chat_request(&identity.model_id, &request);
    let context =
        OpenAiRequestContext::with_cancellation(cancellation.clone()).with_local_only_policy();
    let mut backend_future = backend.chat_completion_with_context(chat, context);
    let mut settled = None;
    let deadline_hit = match request.limits.deadline {
        None => false,
        Some(deadline) => {
            let deadline_at = tokio::time::Instant::from_std(admitted_at + deadline);
            tokio::select! {
                biased;
                result = &mut backend_future => {
                    settled = Some(result);
                    false
                }
                () = tokio::time::sleep_until(deadline_at) => {
                    // Cancel cooperatively. Termination is still awaited below.
                    cancellation.cancel();
                    true
                }
            }
        }
    };
    // Never report before the backend future has returned.
    let result = match settled {
        Some(result) => result,
        None => backend_future.await,
    };
    let elapsed = admitted_at.elapsed();
    if deadline_hit {
        return Err(LocalCompletionError::DeadlineExceeded);
    }
    match result {
        Err(error) => Err(classify_backend_error(
            &error,
            &request.model,
            cancellation.is_cancelled(),
        )),
        Ok(_) if cancellation.is_cancelled() => Err(LocalCompletionError::Cancelled),
        Ok(response) => reduce_chat_response(&response, &identity, &request.limits, elapsed),
    }
}

#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
