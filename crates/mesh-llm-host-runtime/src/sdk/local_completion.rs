//! SDK surface for local-only completions: a cloneable client that takes the
//! single admission permit before anything is enqueued on the runtime
//! control channel.

use std::sync::Arc;

use openai_frontend::CancellationToken;
use tokio::sync::{Semaphore, mpsc, oneshot};

use super::EmbeddedServeHandle;
use crate::api::RuntimeControlRequest;
pub use crate::inference::local_completion::{
    LocalCompletionAdmitted, LocalCompletionBackendErrorKind, LocalCompletionCancel,
    LocalCompletionCapability, LocalCompletionEligibility, LocalCompletionError,
    LocalCompletionFinish, LocalCompletionIneligibility, LocalCompletionJob, LocalCompletionLimits,
    LocalCompletionMessage, LocalCompletionModel, LocalCompletionOutcome, LocalCompletionRequest,
    LocalCompletionResponse, LocalCompletionResponseRejection, LocalCompletionRole,
    LocalModelSelector, MAX_DEADLINE, MAX_MODEL_SELECTOR_BYTES, MAX_REQUEST_LABEL_BYTES,
    validate_local_completion_request,
};

/// Cancels the request's token if dropped while armed: a `submit` future that
/// is dropped between enqueueing and receiving the admission reply must not
/// leave an orphan generation running with nobody able to cancel it.
struct ArmedCancellation {
    cancellation: CancellationToken,
    armed: bool,
}

impl ArmedCancellation {
    fn new(cancellation: CancellationToken) -> Self {
        Self {
            cancellation,
            armed: true,
        }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for ArmedCancellation {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.cancel();
        }
    }
}

/// Cloneable client for local-only completions. Holds no lock on the handle,
/// so a caller can drop its own runtime lock before awaiting.
#[derive(Clone)]
pub struct LocalCompletionClient {
    control_tx: mpsc::UnboundedSender<RuntimeControlRequest>,
    admission: Arc<Semaphore>,
}

impl std::fmt::Debug for LocalCompletionClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalCompletionClient")
            .field("available_permits", &self.admission.available_permits())
            .finish()
    }
}

impl LocalCompletionClient {
    pub(crate) fn new(
        control_tx: mpsc::UnboundedSender<RuntimeControlRequest>,
        admission: Arc<Semaphore>,
    ) -> Self {
        Self {
            control_tx,
            admission,
        }
    }

    /// Content free. Lists every model instance this runtime has loaded
    /// locally with its eligibility. Never lists remote or split peers.
    pub async fn local_models(&self) -> Result<Vec<LocalCompletionModel>, LocalCompletionError> {
        let (resp, rx) = oneshot::channel();
        self.control_tx
            .send(RuntimeControlRequest::LocalCompletionModels { resp })
            .map_err(|_| LocalCompletionError::ControlUnavailable)?;
        rx.await
            .map_err(|_| LocalCompletionError::ResponseDropped)?
    }

    /// Takes the single admission permit BEFORE anything is enqueued on the
    /// control channel. Returns after the runtime resolved the model, leased
    /// it, and started the supervisor. Returns `Busy` immediately when a
    /// completion is already admitted through any clone of this client.
    pub async fn submit(
        &self,
        request: LocalCompletionRequest,
    ) -> Result<LocalCompletionJob, LocalCompletionError> {
        validate_local_completion_request(&request)?;
        let permit = self
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| LocalCompletionError::Busy)?;
        let cancellation = CancellationToken::new();
        let armed = ArmedCancellation::new(cancellation.clone());
        let (resp, rx) = oneshot::channel();
        self.control_tx
            .send(RuntimeControlRequest::LocalCompletion {
                request,
                cancellation: cancellation.clone(),
                permit,
                resp,
            })
            .map_err(|_| LocalCompletionError::ControlUnavailable)?;
        let admitted = rx
            .await
            .map_err(|_| LocalCompletionError::ResponseDropped)??;
        // From here the job owns cancellation-on-drop.
        armed.disarm();
        Ok(LocalCompletionJob::new(cancellation, admitted))
    }
}

impl EmbeddedServeHandle {
    /// Cheap. The client is `Clone + Send + Sync` and holds no lock on the
    /// handle. Fails only when the runtime control channel is already gone.
    pub fn local_completion_client(&self) -> Result<LocalCompletionClient, LocalCompletionError> {
        let control_tx = self
            .control_tx
            .as_ref()
            .ok_or(LocalCompletionError::ControlUnavailable)?
            .clone();
        Ok(LocalCompletionClient::new(
            control_tx,
            Arc::clone(&self.local_completion_admission),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::local_completion::{
        LocalCompletionLimits, LocalCompletionMessage, LocalCompletionRole, LocalModelSelector,
    };
    use std::time::Duration;

    fn sample_request() -> LocalCompletionRequest {
        LocalCompletionRequest {
            model: LocalModelSelector::InstanceId("embedded-1".into()),
            messages: vec![LocalCompletionMessage {
                role: LocalCompletionRole::User,
                content: "hello".into(),
            }],
            limits: LocalCompletionLimits {
                max_completion_tokens: 8,
                max_prompt_bytes: 1024,
                max_completion_bytes: 1024,
                deadline: Some(Duration::from_secs(5)),
                temperature: 0.0,
                seed: None,
            },
            response_format: None,
            request_label: None,
        }
    }

    fn client() -> (
        LocalCompletionClient,
        mpsc::UnboundedReceiver<RuntimeControlRequest>,
        Arc<Semaphore>,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        let admission = Arc::new(Semaphore::new(1));
        (
            LocalCompletionClient::new(tx, Arc::clone(&admission)),
            rx,
            admission,
        )
    }

    #[tokio::test]
    async fn submit_takes_the_permit_before_enqueueing_and_round_trips_through_the_channel() {
        let (client, mut rx, admission) = client();
        let submit = tokio::spawn({
            let client = client.clone();
            async move { client.submit(sample_request()).await }
        });
        let message = rx.recv().await.expect("request enqueued");
        // The permit was taken before the send: nothing is left to acquire.
        assert_eq!(admission.available_permits(), 0);
        let RuntimeControlRequest::LocalCompletion {
            request,
            cancellation,
            permit,
            resp,
        } = message
        else {
            panic!("expected a LocalCompletion request");
        };
        assert_eq!(
            request.model,
            LocalModelSelector::InstanceId("embedded-1".into())
        );
        assert!(!cancellation.is_cancelled());
        let (outcome_tx, outcome_rx) = oneshot::channel();
        resp.send(Ok(LocalCompletionAdmitted {
            instance_id: "embedded-1".into(),
            model_id: "qwen".into(),
            outcome: outcome_rx,
        }))
        .expect("reply delivered");
        let mut job = submit.await.expect("task").expect("admitted");
        assert_eq!(job.instance_id(), "embedded-1");
        assert_eq!(job.model_id(), "qwen");
        // Still running: a bounded wait yields None and the permit stays held.
        assert!(
            job.wait_until(tokio::time::Instant::now() + Duration::from_millis(20))
                .await
                .is_none()
        );
        assert_eq!(admission.available_permits(), 0);
        // A second submission is refused without touching the channel.
        assert_eq!(
            client.submit(sample_request()).await.err(),
            Some(LocalCompletionError::Busy)
        );
        assert!(rx.try_recv().is_err());
        // The supervisor releases the permit only after termination.
        drop(permit);
        outcome_tx
            .send(Err(LocalCompletionError::Cancelled))
            .expect("outcome delivered");
        assert_eq!(job.wait().await, Err(LocalCompletionError::Cancelled));
        assert_eq!(admission.available_permits(), 1);
    }

    #[tokio::test]
    async fn submit_releases_the_permit_when_the_runtime_is_gone() {
        let (client, rx, admission) = client();
        drop(rx);
        assert_eq!(
            client.submit(sample_request()).await.err(),
            Some(LocalCompletionError::ControlUnavailable)
        );
        assert_eq!(admission.available_permits(), 1);
    }

    #[tokio::test]
    async fn dropped_reply_is_reported_not_hung() {
        let (client, mut rx, admission) = client();
        let submit = tokio::spawn({
            let client = client.clone();
            async move { client.submit(sample_request()).await }
        });
        let message = rx.recv().await.expect("request enqueued");
        drop(message);
        assert_eq!(
            submit.await.expect("task").err(),
            Some(LocalCompletionError::ResponseDropped)
        );
        assert_eq!(admission.available_permits(), 1);
    }

    #[tokio::test]
    async fn invalid_requests_never_reach_the_channel() {
        let (client, mut rx, _admission) = client();
        let mut invalid = sample_request();
        invalid.messages.clear();
        assert!(matches!(
            client.submit(invalid).await,
            Err(LocalCompletionError::InvalidRequest(_))
        ));
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn dropping_a_job_cancels_its_token() {
        let (client, mut rx, _admission) = client();
        let submit = tokio::spawn({
            let client = client.clone();
            async move { client.submit(sample_request()).await }
        });
        let RuntimeControlRequest::LocalCompletion {
            cancellation,
            resp,
            permit,
            ..
        } = rx.recv().await.expect("request enqueued")
        else {
            panic!("expected a LocalCompletion request");
        };
        let (_outcome_tx, outcome_rx) = oneshot::channel();
        resp.send(Ok(LocalCompletionAdmitted {
            instance_id: "embedded-1".into(),
            model_id: "qwen".into(),
            outcome: outcome_rx,
        }))
        .expect("reply delivered");
        let job = submit.await.expect("task").expect("admitted");
        assert!(!cancellation.is_cancelled());
        drop(job);
        assert!(cancellation.is_cancelled());
        drop(permit);
    }

    #[tokio::test]
    async fn a_submit_dropped_while_awaiting_admission_cancels_the_enqueued_request() {
        let (client, mut rx, _admission) = client();
        let submit = tokio::spawn({
            let client = client.clone();
            async move { client.submit(sample_request()).await }
        });
        let RuntimeControlRequest::LocalCompletion {
            cancellation,
            permit,
            resp,
            ..
        } = rx.recv().await.expect("request enqueued")
        else {
            panic!("expected a LocalCompletion request");
        };
        assert!(!cancellation.is_cancelled());
        // The caller abandons the submit before the runtime replies.
        submit.abort();
        let _ = submit.await;
        assert!(
            cancellation.is_cancelled(),
            "the runtime-side token must be cancelled so an admitted generation stops"
        );
        drop(resp);
        drop(permit);
    }

    #[tokio::test]
    async fn a_submit_dropped_before_the_runtime_dispatches_arrives_already_cancelled() {
        let (client, mut rx, admission) = client();
        let submit = tokio::spawn({
            let client = client.clone();
            async move { client.submit(sample_request()).await }
        });
        // Wait until the request is in the channel, then abandon the caller
        // before anything dequeues it.
        tokio::time::sleep(Duration::from_millis(20)).await;
        submit.abort();
        let _ = submit.await;
        let RuntimeControlRequest::LocalCompletion {
            cancellation,
            permit,
            ..
        } = rx.recv().await.expect("request stayed enqueued")
        else {
            panic!("expected a LocalCompletion request");
        };
        assert!(
            cancellation.is_cancelled(),
            "the runtime dequeues an already-cancelled request and finishes immediately"
        );
        assert_eq!(
            admission.available_permits(),
            0,
            "the permit travels with the request"
        );
        drop(permit);
        assert_eq!(admission.available_permits(), 1);
    }

    #[tokio::test]
    async fn handle_without_control_channel_cannot_make_a_client() {
        let handle = EmbeddedServeHandle::detached_for_tests();
        assert_eq!(
            handle.local_completion_client().err(),
            Some(LocalCompletionError::ControlUnavailable)
        );
    }
}
