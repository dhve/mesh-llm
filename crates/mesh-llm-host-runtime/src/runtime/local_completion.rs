//! Control-loop side of the local-only completion: resolve a loaded model
//! from dynamic handles or handle-owned startup registrations, lease it, take
//! the runtime slot, and hand the
//! generation to a supervisor task that owns every permit until the backend
//! terminates. Nothing here touches mesh targets, the ingress or peers.

use std::collections::HashMap;
use std::sync::Arc;

use openai_frontend::CancellationToken;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

mod startup_registry;
pub(super) use startup_registry::{StartupLocalCompletions, StartupRegistration};

use super::model_lifecycle::RuntimeModelHandleEntry;
use crate::api;
use crate::inference::local_completion::{
    LeasedLocalBackend, LocalCompletionAdmitted, LocalCompletionCandidate,
    LocalCompletionEligibility, LocalCompletionError, LocalCompletionExecution,
    LocalCompletionIdentity, LocalCompletionLease, LocalCompletionModel, LocalCompletionOutcome,
    LocalCompletionRequest, eligibility_for, execute_local_completion,
    resolve_local_completion_target, validate_local_completion_request,
};

/// One admitted local completion per runtime, defended on the runtime side
/// in addition to the client side permit.
pub(super) fn new_local_completion_slot() -> Arc<Semaphore> {
    Arc::new(Semaphore::new(1))
}

/// The only runtime state the local-only path may read: the loaded local
/// model records and the runtime slot. Routing targets, peers, the ingress
/// and the mesh node are not reachable from here by construction.
pub(super) struct LocalCompletionAdmissionView<'a> {
    pub(super) runtime_models: &'a HashMap<String, RuntimeModelHandleEntry>,
    pub(super) slot: &'a Arc<Semaphore>,
    pub(super) startup_models: StartupLocalCompletions,
}

/// Dispatch the local-completion control requests. Returns the request back
/// unchanged when it is not one of them, so the caller's match handles it.
/// This is the single production entry point for both variants.
pub(super) fn handle_local_completion_control(
    view: LocalCompletionAdmissionView<'_>,
    cmd: api::RuntimeControlRequest,
) -> Option<api::RuntimeControlRequest> {
    match cmd {
        api::RuntimeControlRequest::LocalCompletionModels { resp } => {
            let _ = resp.send(list_local_completion_models(&view));
            None
        }
        api::RuntimeControlRequest::LocalCompletion {
            request,
            cancellation,
            permit,
            resp,
        } => {
            let _ = resp.send(admit(&view, request, cancellation, permit));
            None
        }
        other => Some(other),
    }
}

/// Test-only in-process model record. Lives here so the production enum in
/// `runtime/local.rs` can name it under `cfg(test)`.
#[cfg(test)]
pub(super) struct FakeLocalModel {
    pub(super) served_model_id: String,
    pub(super) stage_id: String,
    pub(super) backend: Arc<dyn openai_frontend::OpenAiBackend>,
    pub(super) load_shape: crate::inference::local_completion::LocalModelLoadShape,
    pub(super) capability: crate::inference::local_completion::LocalCompletionCapability,
    pub(super) leases: Arc<crate::inference::local_completion::LocalCompletionLeases>,
}

/// A loaded runtime model viewed as a resolver candidate.
pub(super) struct RuntimeModelCandidate<'a> {
    pub(super) instance_id: &'a str,
    pub(super) entry: &'a RuntimeModelHandleEntry,
}

impl LocalCompletionCandidate for RuntimeModelCandidate<'_> {
    fn instance_id(&self) -> &str {
        self.instance_id
    }

    fn model_id(&self) -> &str {
        self.entry
            .handle
            .local_completion_target()
            .map_or(self.entry.model_name.as_str(), |target| {
                target.served_model_id
            })
    }

    fn model_ref(&self) -> &str {
        &self.entry.model_name
    }

    fn profile(&self) -> &str {
        &self.entry.profile
    }

    fn context_length(&self) -> u32 {
        self.entry.handle.context_length
    }

    fn backend_label(&self) -> &str {
        &self.entry.handle.backend
    }

    fn eligibility(&self) -> LocalCompletionEligibility {
        match self.entry.handle.local_completion_target() {
            Some(target) => eligibility_for(
                target.load_shape,
                &self.entry.handle.backend,
                target.stage_id,
                &target.capability,
            ),
            None => LocalCompletionEligibility::Ineligible(
                crate::inference::local_completion::LocalCompletionIneligibility::NotSkippy {
                    backend: self.entry.handle.backend.clone(),
                },
            ),
        }
    }

    fn lease_backend(&self, cancellation: CancellationToken) -> Option<LeasedLocalBackend> {
        let target = self.entry.handle.local_completion_target()?;
        Some(LeasedLocalBackend {
            backend: target.backend,
            lease: target.leases.begin(cancellation),
        })
    }
}

/// Content free listing of every loaded runtime model with eligibility.
fn list_local_completion_models(
    view: &LocalCompletionAdmissionView<'_>,
) -> Result<Vec<LocalCompletionModel>, LocalCompletionError> {
    let startup = view.startup_models.lock()?;
    let mut models: Vec<LocalCompletionModel> = view
        .runtime_models
        .iter()
        .map(|(instance_id, entry)| RuntimeModelCandidate { instance_id, entry }.describe())
        .collect();
    models.extend(startup.values().map(LocalCompletionCandidate::describe));
    models.sort_by(|left, right| left.instance_id.cmp(&right.instance_id));
    Ok(models)
}

/// Admit one local completion after validation, model resolution, leasing
/// and slot acquisition; the generation itself runs in a spawned supervisor
/// that releases the lease, the runtime slot and the client permit only after
/// the backend future returned.
fn admit(
    view: &LocalCompletionAdmissionView<'_>,
    request: LocalCompletionRequest,
    cancellation: CancellationToken,
    client_permit: OwnedSemaphorePermit,
) -> Result<LocalCompletionAdmitted, LocalCompletionError> {
    validate_local_completion_request(&request)?;
    let candidates: Vec<RuntimeModelCandidate<'_>> = view
        .runtime_models
        .iter()
        .map(|(instance_id, entry)| RuntimeModelCandidate { instance_id, entry })
        .collect();
    let startup = view.startup_models.lock()?;
    let candidates = candidates
        .iter()
        .map(|candidate| candidate as &dyn LocalCompletionCandidate)
        .chain(
            startup
                .values()
                .map(|candidate| candidate as &dyn LocalCompletionCandidate),
        );
    let target = resolve_local_completion_target(candidates, &request.model)?;
    let identity = LocalCompletionIdentity {
        model_id: target.model_id().to_string(),
        instance_id: target.instance_id().to_string(),
    };
    let runtime_permit = view
        .slot
        .clone()
        .try_acquire_owned()
        .map_err(|_| LocalCompletionError::Busy)?;
    let leased = target.lease_backend(cancellation.clone()).ok_or_else(|| {
        LocalCompletionError::Ineligible {
            model: identity.model_id.clone(),
            reason: crate::inference::local_completion::LocalCompletionIneligibility::NotSkippy {
                backend: target.backend_label().to_string(),
            },
        }
    })?;
    let (outcome_tx, outcome_rx) = oneshot::channel::<LocalCompletionOutcome>();
    let execution = LocalCompletionExecution {
        backend: leased.backend,
        identity: identity.clone(),
        request,
        cancellation,
        admitted_at: std::time::Instant::now(),
    };
    spawn_supervisor(
        execution,
        TerminalOwnership::new(leased.lease, runtime_permit, client_permit, outcome_tx),
    );
    Ok(LocalCompletionAdmitted {
        instance_id: identity.instance_id,
        model_id: identity.model_id,
        outcome: outcome_rx,
    })
}

/// Everything whose release must wait for the backend to actually return:
/// the model lease, the runtime slot, the client's admission permit, and the
/// caller's outcome channel.
///
/// `complete` releases them in order and reports the outcome. If the owning
/// task is aborted or panics before that, `Drop` runs instead: the lease and
/// both permits are deliberately kept held for the life of the runtime (the
/// blocking generation may still be running and nothing can prove otherwise)
/// and the caller receives [`LocalCompletionError::TerminationUnknown`], so a
/// closed channel is never the only signal.
pub(super) struct TerminalOwnership {
    lease: Option<LocalCompletionLease>,
    runtime_permit: Option<OwnedSemaphorePermit>,
    client_permit: Option<OwnedSemaphorePermit>,
    outcome_tx: Option<oneshot::Sender<LocalCompletionOutcome>>,
}

impl TerminalOwnership {
    pub(super) fn new(
        lease: LocalCompletionLease,
        runtime_permit: OwnedSemaphorePermit,
        client_permit: OwnedSemaphorePermit,
        outcome_tx: oneshot::Sender<LocalCompletionOutcome>,
    ) -> Self {
        Self {
            lease: Some(lease),
            runtime_permit: Some(runtime_permit),
            client_permit: Some(client_permit),
            outcome_tx: Some(outcome_tx),
        }
    }

    /// Called only after the backend future returned. Releases the lease,
    /// then the runtime slot, then the client permit, then reports.
    fn complete(mut self, outcome: LocalCompletionOutcome) {
        drop(self.lease.take());
        drop(self.runtime_permit.take());
        drop(self.client_permit.take());
        if let Some(outcome_tx) = self.outcome_tx.take() {
            let _ = outcome_tx.send(outcome);
        }
    }
}

impl Drop for TerminalOwnership {
    fn drop(&mut self) {
        let Some(outcome_tx) = self.outcome_tx.take() else {
            // `complete` already ran; nothing is held.
            return;
        };
        // Abort or panic before termination: keep every hold.
        if let Some(lease) = self.lease.take() {
            std::mem::forget(lease);
        }
        if let Some(permit) = self.runtime_permit.take() {
            std::mem::forget(permit);
        }
        if let Some(permit) = self.client_permit.take() {
            std::mem::forget(permit);
        }
        tracing::error!(
            "local completion supervisor ended before the backend returned; \
             keeping the model lease and admission permits held"
        );
        let _ = outcome_tx.send(Err(LocalCompletionError::TerminationUnknown));
    }
}

/// Spawn the task that owns the generation until the backend returns.
pub(super) fn spawn_supervisor(
    execution: LocalCompletionExecution,
    ownership: TerminalOwnership,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(supervise(execution, ownership))
}

async fn supervise(execution: LocalCompletionExecution, ownership: TerminalOwnership) {
    let label = execution.request.request_label.clone();
    let instance_id = execution.identity.instance_id.clone();
    let outcome = execute_local_completion(execution).await;
    tracing::info!(
        instance_id = %instance_id,
        request_label = label.as_deref().unwrap_or(""),
        outcome = outcome_label(&outcome),
        "local completion terminated"
    );
    ownership.complete(outcome);
}

fn outcome_label(outcome: &LocalCompletionOutcome) -> &'static str {
    match outcome {
        Ok(_) => "completed",
        Err(LocalCompletionError::Cancelled) => "cancelled",
        Err(LocalCompletionError::DeadlineExceeded) => "deadline_exceeded",
        Err(LocalCompletionError::ContextBudgetExceeded) => "context_budget_exceeded",
        Err(LocalCompletionError::OutputTooLarge { .. }) => "output_too_large",
        Err(LocalCompletionError::UnsupportedResponseShape(_)) => "response_rejected",
        Err(LocalCompletionError::AdmissionTimeout) => "admission_timeout",
        Err(_) => "failed",
    }
}

#[cfg(test)]
mod tests;
