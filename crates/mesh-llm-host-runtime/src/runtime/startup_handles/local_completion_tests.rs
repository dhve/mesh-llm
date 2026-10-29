//! Startup registration must feed the private API even when the dynamic map is empty.
use super::*;
use crate::inference::local_completion::test_support::{Behaviour, FakeBackend, response};
use crate::inference::local_completion::{
    LocalCompletionAdmitted, LocalCompletionCapability, LocalCompletionEligibility,
    LocalCompletionError, LocalCompletionIneligibility, LocalCompletionLeases,
    LocalCompletionLimits, LocalCompletionMessage, LocalCompletionRequest, LocalCompletionRole,
    LocalModelLoadShape, LocalModelSelector,
};
use crate::runtime::local::{LocalRuntimeBackendHandle, openai_guardrail_policy_handle};
use crate::runtime::local_completion::{
    FakeLocalModel, LocalCompletionAdmissionView, handle_local_completion_control,
    new_local_completion_slot,
};
use openai_frontend::{FinishReason, Usage};
use std::collections::HashMap;

struct Fixture {
    node: mesh::Node,
    tunnel: tunnel::Manager,
    config: plugin::MeshConfig,
    target: Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    registry: RuntimeInstanceRegistry,
    ledger: RuntimeCapacityLedger,
    processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    usage: DashboardContextUsage,
    lifecycle: Arc<tokio::sync::Mutex<InstanceLifecycleRecord>>,
    port: Arc<AtomicU16>,
    path: PathBuf,
    survey: survey::SurveyTelemetry,
    telemetry: skippy::SkippyTelemetryOptions,
    slot: Arc<tokio::sync::Semaphore>,
}
impl Fixture {
    async fn new() -> Self {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let tunnel = tunnel::Manager::start(
            node.clone(),
            tokio::sync::mpsc::channel(1).1,
            tokio::sync::mpsc::channel(1).1,
            tokio::sync::mpsc::channel(1).1,
        )
        .await
        .unwrap();
        Self {
            slot: new_local_completion_slot(),
            node,
            tunnel,
            config: plugin::MeshConfig::default(),
            target: Arc::new(tokio::sync::watch::channel(election::ModelTargets::default()).0),
            registry: RuntimeInstanceRegistry::default(),
            ledger: RuntimeCapacityLedger::default(),
            processes: Arc::default(),
            usage: Arc::default(),
            lifecycle: Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
                InstanceLifecycleState::Warming,
                50,
            ))),
            port: Arc::new(AtomicU16::new(0)),
            path: PathBuf::from("startup-fixture.gguf"),
            survey: survey::SurveyTelemetry::disabled(),
            telemetry: skippy::SkippyTelemetryOptions::default(),
        }
    }
    fn context<'a>(&'a self, instance: &'a str) -> StartupLoopContext<'a> {
        StartupLoopContext {
            node: &self.node,
            config: &self.config,
            tunnel_mgr: &self.tunnel,
            target_tx: &self.target,
            model_path: &self.path,
            model_ref: "startup-fixture",
            preindexed_split_package: None,
            config_model_id: None,
            runtime_profile: "startup",
            readiness_index: 0,
            instance_id: instance,
            primary_model_name: "qwen",
            mmproj_path: None,
            ctx_size: Some(4096),
            pinned_gpu: None,
            device_override: None,
            runtime_capacity_ledger: &self.ledger,
            cache_type_k: None,
            cache_type_v: None,
            n_batch: None,
            n_ubatch: None,
            flash_attention: skippy_protocol::FlashAttentionType::Auto,
            parallel_override: None,
            local_source_required: true,
            resource_planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            openai_guardrail_policy: openai_guardrail_policy_handle(
                openai_frontend::GuardrailMode::Disabled,
            ),
            skippy_telemetry: &self.telemetry,
            survey_telemetry: &self.survey,
            launch_kind: survey::SurveyLaunchKind::Startup,
            dashboard_processes: &self.processes,
            dashboard_context_usage: &self.usage,
            runtime_instance_registry: &self.registry,
            console_state: None,
            api_port: 0,
            runtime_data_producer: None,
            lifecycle: &self.lifecycle,
            lifecycle_port: &self.port,
        }
    }
    async fn register(&self, instance: &str, handle: &LocalRuntimeModelHandle) {
        // Each startup launch/replacement has its own Warming lifecycle record.
        let lifecycle = Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
            InstanceLifecycleState::Warming,
            50,
        )));
        let mut context = self.context(instance);
        context.lifecycle = &lifecycle;
        startup_register_loaded_runtime(&context, "qwen", handle).await;
    }
    async fn submit(
        &self,
        selector: LocalModelSelector,
    ) -> Result<LocalCompletionAdmitted, LocalCompletionError> {
        let request = LocalCompletionRequest {
            model: selector,
            messages: vec![LocalCompletionMessage {
                role: LocalCompletionRole::User,
                content: "ping".into(),
            }],
            limits: LocalCompletionLimits {
                max_completion_tokens: 8,
                max_prompt_bytes: 1024,
                max_completion_bytes: 1024,
                deadline: None,
                temperature: 0.0,
                seed: None,
            },
            response_format: None,
            request_label: None,
        };
        let (resp, result) = tokio::sync::oneshot::channel();
        let models = HashMap::new();
        let permit = Arc::new(tokio::sync::Semaphore::new(1))
            .acquire_owned()
            .await
            .unwrap();
        assert!(
            handle_local_completion_control(
                LocalCompletionAdmissionView {
                    runtime_models: &models,
                    slot: &self.slot,
                    startup_models: self.registry.startup_local_completions.clone(),
                },
                api::RuntimeControlRequest::LocalCompletion {
                    request,
                    cancellation: openai_frontend::CancellationToken::new(),
                    permit,
                    resp,
                }
            )
            .is_none()
        );
        result.await.unwrap()
    }
    async fn list(&self) -> Vec<crate::inference::local_completion::LocalCompletionModel> {
        let models = HashMap::new();
        let slot = new_local_completion_slot();
        let (resp, result) = tokio::sync::oneshot::channel();
        assert!(
            handle_local_completion_control(
                LocalCompletionAdmissionView {
                    runtime_models: &models,
                    slot: &slot,
                    startup_models: self.registry.startup_local_completions.clone(),
                },
                api::RuntimeControlRequest::LocalCompletionModels { resp }
            )
            .is_none()
        );
        result.await.unwrap().unwrap()
    }
}
fn model(shape: LocalModelLoadShape) -> LocalRuntimeModelHandle {
    model_with_backend(
        shape,
        FakeBackend::new(Behaviour::Respond(response(
            "LOCAL",
            Usage::new(4, 1),
            FinishReason::Stop,
        ))),
    )
}
fn model_with_backend(
    shape: LocalModelLoadShape,
    backend: Arc<FakeBackend>,
) -> LocalRuntimeModelHandle {
    LocalRuntimeModelHandle {
        startup_registration: Default::default(),
        port: 0,
        backend: "skippy".into(),
        context_length: 4096,
        slots: 1,
        capabilities: crate::models::ModelCapabilities::default(),
        inner: LocalRuntimeBackendHandle::LocalCompletionFake(FakeLocalModel {
            served_model_id: "qwen".into(),
            stage_id: "stage-0".into(),
            backend,
            load_shape: shape,
            capability: LocalCompletionCapability {
                single_stage: shape == LocalModelLoadShape::SingleStage,
                receipts_configured: false,
                kv_cache_configured: false,
                honours_request_context: true,
                local_only_policy_supported: true,
            },
            leases: LocalCompletionLeases::new(),
        }),
    }
}
#[tokio::test]
async fn startup_registration_exposes_initial_and_extra_models_without_dynamic_loads() {
    let fixture = Fixture::new().await;
    let first = model(LocalModelLoadShape::SingleStage);
    fixture.register("startup-1", &first).await;
    let listed = fixture.list().await;
    assert_eq!(
        listed.len(),
        1,
        "normal startup must be visible without dynamic runtime_models"
    );
    assert_eq!(listed[0].instance_id, "startup-1");
    assert_eq!(listed[0].eligibility, LocalCompletionEligibility::Eligible);
    let admitted = fixture
        .submit(LocalModelSelector::InstanceId("startup-1".into()))
        .await
        .unwrap();
    assert_eq!(admitted.instance_id, "startup-1");
    assert!(admitted.outcome.await.unwrap().is_ok());
    let extra = model(LocalModelLoadShape::SingleStage);
    fixture.register("startup-2", &extra).await;
    assert_eq!(
        fixture.list().await.len(),
        2,
        "extra startup instance retained"
    );
    assert!(
        matches!(fixture.submit(LocalModelSelector::ModelId("qwen".into())).await,
        Err(LocalCompletionError::Ambiguous { instances, .. }) if instances.len() == 2)
    );
    first.shutdown().await;
    extra.shutdown().await;
}

#[tokio::test]
async fn startup_split_replacement_and_old_generation_retirement_are_isolated() {
    let fixture = Fixture::new().await;
    let split = model(LocalModelLoadShape::SplitStageZero);
    fixture.register("startup-1", &split).await;
    assert!(matches!(
        fixture
            .submit(LocalModelSelector::InstanceId("startup-1".into()))
            .await,
        Err(LocalCompletionError::Ineligible {
            reason: LocalCompletionIneligibility::SplitStage { .. },
            ..
        })
    ));
    // The common registration seam is used by split replacement and local fallback.
    let replacement = model(LocalModelLoadShape::SingleStage);
    fixture.register("startup-1", &replacement).await;
    split.shutdown().await;
    let listed = fixture.list().await;
    assert_eq!(
        listed.len(),
        1,
        "old generation cannot remove replacement or duplicate its alias"
    );
    assert_eq!(listed[0].eligibility, LocalCompletionEligibility::Eligible);
    let admitted = fixture
        .submit(LocalModelSelector::InstanceId("startup-1".into()))
        .await
        .unwrap();
    assert!(admitted.outcome.await.unwrap().is_ok());
    drop(replacement);
    assert!(
        fixture.list().await.is_empty(),
        "owner drop must retire registration"
    );
    assert!(matches!(
        fixture
            .submit(LocalModelSelector::InstanceId("startup-1".into()))
            .await,
        Err(LocalCompletionError::ModelNotLoadedLocally { .. })
    ));
}

#[tokio::test]
async fn startup_shutdown_retires_admission_before_waiting_for_held_completion() {
    let fixture = Fixture::new().await;
    let backend = FakeBackend::new(Behaviour::IgnoreCancellationUntilReleased);
    let handle = model_with_backend(LocalModelLoadShape::SingleStage, backend.clone());
    fixture.register("startup-1", &handle).await;
    let admitted = fixture
        .submit(LocalModelSelector::InstanceId("startup-1".into()))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while backend.seen_request.lock().unwrap().is_none() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let shutdown = tokio::spawn(handle.shutdown());
    tokio::time::timeout(Duration::from_secs(2), async {
        while !fixture.list().await.is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("retirement must precede backend drain");
    assert!(
        !shutdown.is_finished(),
        "shutdown must still wait for backend termination"
    );
    assert!(matches!(
        fixture
            .submit(LocalModelSelector::InstanceId("startup-1".into()))
            .await,
        Err(LocalCompletionError::ModelNotLoadedLocally { .. })
    ));
    backend.release.notify_one();
    tokio::time::timeout(Duration::from_secs(2), shutdown)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        admitted.outcome.await.unwrap(),
        Err(LocalCompletionError::Cancelled)
    );
}

#[tokio::test]
async fn startup_targets_are_private_to_their_runtime_registry() {
    let first = Fixture::new().await;
    let second = Fixture::new().await;
    let handle = model(LocalModelLoadShape::SingleStage);
    first.register("startup-1", &handle).await;
    assert_eq!(first.list().await.len(), 1);
    assert!(second.list().await.is_empty());
    assert!(matches!(
        second
            .submit(LocalModelSelector::InstanceId("startup-1".into()))
            .await,
        Err(LocalCompletionError::ModelNotLoadedLocally { .. })
    ));
    handle.shutdown().await;
}
