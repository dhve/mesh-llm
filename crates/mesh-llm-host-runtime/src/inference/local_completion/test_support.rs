//! Fake backend shared by the local completion unit tests.

use super::*;
use async_trait::async_trait;
use openai_frontend::{ChatCompletionStream, ModelObject, OpenAiResult, Usage};
use std::sync::atomic::AtomicBool;
use tokio::sync::Notify;

pub(crate) fn response(
    content: &str,
    usage: Usage,
    finish: FinishReason,
) -> ChatCompletionResponse {
    ChatCompletionResponse::new_with_reason("qwen", content, usage, finish)
}

pub(crate) enum Behaviour {
    Respond(ChatCompletionResponse),
    /// Observe cancellation between "decode steps" and return the backend's
    /// cancelled error.
    HonourCancellation,
    /// Ignore cancellation and return only once `release` fires.
    IgnoreCancellationUntilReleased,
    Fail(OpenAiError),
}

pub(crate) struct FakeBackend {
    pub(crate) behaviour: Behaviour,
    pub(crate) release: Notify,
    pub(crate) seen_request: std::sync::Mutex<Option<ChatCompletionRequest>>,
    pub(crate) seen_local_only: AtomicBool,
    pub(crate) returned: AtomicBool,
}

impl FakeBackend {
    pub(crate) fn new(behaviour: Behaviour) -> Arc<Self> {
        Arc::new(Self {
            behaviour,
            release: Notify::new(),
            seen_request: std::sync::Mutex::new(None),
            seen_local_only: AtomicBool::new(false),
            returned: AtomicBool::new(false),
        })
    }
}

#[async_trait]
impl OpenAiBackend for FakeBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("qwen")])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.chat_completion_with_context(request, OpenAiRequestContext::new())
            .await
    }

    async fn chat_completion_with_context(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionResponse> {
        *self.seen_request.lock().expect("seen request lock") = Some(request);
        self.seen_local_only
            .store(context.requires_local_only_policy(), Ordering::Release);
        let result = match &self.behaviour {
            Behaviour::Respond(response) => Ok(response.clone()),
            Behaviour::HonourCancellation => {
                context.cancellation_token().cancelled().await;
                Err(OpenAiError::cancelled("request cancelled"))
            }
            Behaviour::IgnoreCancellationUntilReleased => {
                self.release.notified().await;
                Ok(response("late", Usage::new(3, 2), FinishReason::Stop))
            }
            Behaviour::Fail(error) => Err(error.clone()),
        };
        self.returned.store(true, Ordering::Release);
        result
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        Err(OpenAiError::unsupported("not needed"))
    }
}
