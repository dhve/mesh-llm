//! Startup tasks own their handles outside the dynamic control-loop map.
//! Resolve and lease these targets under the same lock that retires them.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use openai_frontend::{CancellationToken, OpenAiBackend};

use super::*;
use crate::inference::local_completion::LocalCompletionLeases;
use crate::runtime::local::LocalRuntimeModelHandle;

type Targets = Mutex<HashMap<String, StartupTarget>>;

#[derive(Clone, Default)]
pub(in crate::runtime) struct StartupLocalCompletions(Arc<Targets>);

impl StartupLocalCompletions {
    pub(super) fn lock(
        &self,
    ) -> Result<MutexGuard<'_, HashMap<String, StartupTarget>>, LocalCompletionError> {
        self.0
            .lock()
            .map_err(|_| LocalCompletionError::ControlUnavailable)
    }

    pub(in crate::runtime) fn register(
        &self,
        instance_id: &str,
        model_ref: &str,
        profile: &str,
        handle: &LocalRuntimeModelHandle,
    ) -> Option<StartupRegistration> {
        let target = handle.local_completion_target()?;
        let token = Arc::new(());
        let entry = StartupTarget {
            metadata: LocalCompletionModel {
                instance_id: instance_id.into(),
                model_id: target.served_model_id.into(),
                model_ref: model_ref.into(),
                profile: profile.into(),
                context_length: handle.context_length,
                backend: handle.backend.clone(),
                eligibility: eligibility_for(
                    target.load_shape,
                    &handle.backend,
                    target.stage_id,
                    &target.capability,
                ),
            },
            backend: target.backend,
            leases: target.leases,
            token: token.clone(),
        };
        self.0.lock().ok()?.insert(instance_id.into(), entry);
        Some(StartupRegistration {
            registry: Arc::downgrade(&self.0),
            instance_id: instance_id.into(),
            token,
        })
    }
}

/// Owned by the model handle, ahead of its backend in field-drop order.
/// An old split generation must not retire a newer registration of the same id.
pub(in crate::runtime) struct StartupRegistration {
    registry: Weak<Targets>,
    instance_id: String,
    token: Arc<()>,
}
impl Drop for StartupRegistration {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade()
            && let Ok(mut targets) = registry.lock()
            && targets
                .get(&self.instance_id)
                .is_some_and(|entry| Arc::ptr_eq(&entry.token, &self.token))
        {
            targets.remove(&self.instance_id);
        }
    }
}

pub(super) struct StartupTarget {
    metadata: LocalCompletionModel,
    backend: Arc<dyn OpenAiBackend>,
    leases: Arc<LocalCompletionLeases>,
    token: Arc<()>,
}
impl LocalCompletionCandidate for StartupTarget {
    fn instance_id(&self) -> &str {
        &self.metadata.instance_id
    }
    fn model_id(&self) -> &str {
        &self.metadata.model_id
    }
    fn model_ref(&self) -> &str {
        &self.metadata.model_ref
    }
    fn profile(&self) -> &str {
        &self.metadata.profile
    }
    fn context_length(&self) -> u32 {
        self.metadata.context_length
    }
    fn backend_label(&self) -> &str {
        &self.metadata.backend
    }
    fn eligibility(&self) -> LocalCompletionEligibility {
        self.metadata.eligibility.clone()
    }
    fn lease_backend(&self, cancellation: CancellationToken) -> Option<LeasedLocalBackend> {
        // The registry guard remains held by admission until this lease exists.
        Some(LeasedLocalBackend {
            backend: self.backend.clone(),
            lease: self.leases.begin(cancellation),
        })
    }
}
