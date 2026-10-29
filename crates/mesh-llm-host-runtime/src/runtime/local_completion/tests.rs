use super::*;
use crate::inference::election;
use crate::inference::local_completion::test_support::{Behaviour, FakeBackend, response};
use crate::inference::local_completion::{
    LocalCompletionCapability, LocalCompletionIneligibility, LocalCompletionJob,
    LocalCompletionLeases, LocalCompletionLimits, LocalCompletionMessage, LocalCompletionRole,
    LocalModelLoadShape, LocalModelSelector,
};
use crate::runtime::capacity::{
    RuntimeCapacityLedger, RuntimeCapacityPool, RuntimeCapacityRequest,
};
use crate::runtime::local::{LocalRuntimeBackendHandle, LocalRuntimeModelHandle};
use crate::runtime::{InstanceLifecycleRecord, InstanceLifecycleState};
use crate::sdk::LocalCompletionClient;
use openai_frontend::{FinishReason, Usage};
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::sync::mpsc;

/// A genuine runtime model record (the type the control loop keys by
/// instance id) whose backend is an in-process fake.
fn fake_entry(
    instance_id: &str,
    model_id: &str,
    backend: Arc<FakeBackend>,
    load_shape: LocalModelLoadShape,
) -> (RuntimeModelHandleEntry, Arc<LocalCompletionLeases>) {
    let leases = LocalCompletionLeases::new();
    let capacity_reservation = RuntimeCapacityLedger::default()
        .reserve(RuntimeCapacityRequest {
            instance_id: instance_id.to_string(),
            model_name: model_id.to_string(),
            pool: RuntimeCapacityPool::Node,
            capacity_bytes: 1,
            required_bytes: 0,
        })
        .expect("zero-byte reservation");
    let entry = RuntimeModelHandleEntry {
        model_name: model_id.to_string(),
        profile: String::new(),
        handle: LocalRuntimeModelHandle {
            port: 0,
            backend: "skippy".to_string(),
            context_length: 4096,
            slots: 1,
            capabilities: crate::models::ModelCapabilities::default(),
            inner: LocalRuntimeBackendHandle::LocalCompletionFake(FakeLocalModel {
                served_model_id: model_id.to_string(),
                stage_id: "stage-0".to_string(),
                backend,
                load_shape,
                capability: LocalCompletionCapability {
                    single_stage: load_shape == LocalModelLoadShape::SingleStage,
                    receipts_configured: false,
                    kv_cache_configured: false,
                    honours_request_context: true,
                    local_only_policy_supported: true,
                },
                leases: Arc::clone(&leases),
            }),
        },
        capacity_reservation,
        lifecycle: Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
            InstanceLifecycleState::Serving,
            50,
        ))),
    };
    (entry, leases)
}

fn view<'a>(
    models: &'a HashMap<String, RuntimeModelHandleEntry>,
    slot: &'a Arc<Semaphore>,
) -> LocalCompletionAdmissionView<'a> {
    LocalCompletionAdmissionView {
        runtime_models: models,
        slot,
    }
}

fn request_for(selector: LocalModelSelector) -> LocalCompletionRequest {
    let mut request = test_request();
    request.model = selector;
    request
}

/// Drive one control request through the production dispatch function.
async fn dispatch_one(
    control_rx: &mut mpsc::UnboundedReceiver<api::RuntimeControlRequest>,
    models: &HashMap<String, RuntimeModelHandleEntry>,
    slot: &Arc<Semaphore>,
) {
    let cmd = control_rx
        .recv()
        .await
        .expect("a control request was enqueued");
    assert!(
        handle_local_completion_control(view(models, slot), cmd).is_none(),
        "local completion variants are handled here"
    );
}

#[tokio::test]
async fn dispatch_uses_the_loaded_local_target_and_never_a_remote_or_router_trap() {
    // Traps: a peer backend for the same model id that must see zero
    // requests, and a routing table that advertises it remotely. Neither is
    // reachable from the admission view by construction.
    let peer = FakeBackend::new(Behaviour::Respond(response(
        "PEER",
        Usage::new(1, 1),
        FinishReason::Stop,
    )));
    let mut routing = election::ModelTargets::default();
    routing
        .targets
        .entry("qwen".to_string())
        .or_default()
        .push(election::InferenceTarget::Remote(
            iroh::SecretKey::generate().public(),
        ));
    routing
        .targets
        .entry("remote-only".to_string())
        .or_default()
        .push(election::InferenceTarget::Remote(
            iroh::SecretKey::generate().public(),
        ));

    let local = FakeBackend::new(Behaviour::Respond(response(
        "LOCAL",
        Usage::new(4, 1),
        FinishReason::Stop,
    )));
    let split = FakeBackend::new(Behaviour::Respond(response(
        "SPLIT",
        Usage::new(4, 1),
        FinishReason::Stop,
    )));
    let (local_entry, local_leases) = fake_entry(
        "embedded-1",
        "qwen",
        local.clone(),
        LocalModelLoadShape::SingleStage,
    );
    let (split_entry, _) = fake_entry(
        "embedded-2",
        "big-split",
        split.clone(),
        LocalModelLoadShape::SplitStageZero,
    );
    let mut models = HashMap::new();
    models.insert("embedded-1".to_string(), local_entry);
    models.insert("embedded-2".to_string(), split_entry);
    let slot = new_local_completion_slot();
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    let client = LocalCompletionClient::new(control_tx, Arc::new(Semaphore::new(1)));

    // Listing goes through the production dispatch and reports eligibility.
    let listing = tokio::spawn({
        let client = client.clone();
        async move { client.local_models().await }
    });
    dispatch_one(&mut control_rx, &models, &slot).await;
    let listed = listing.await.expect("task").expect("listing");
    assert_eq!(listed.len(), 2);
    assert_eq!(
        listed[0].eligibility,
        LocalCompletionEligibility::Eligible,
        "{listed:?}"
    );
    assert!(matches!(
        listed[1].eligibility,
        LocalCompletionEligibility::Ineligible(LocalCompletionIneligibility::SplitStage { .. })
    ));

    // A model id loaded locally resolves to the loaded instance and runs on
    // its backend with the local-only policy; the peer never sees content.
    let submit = tokio::spawn({
        let client = client.clone();
        async move {
            client
                .submit(request_for(LocalModelSelector::ModelId("qwen".into())))
                .await
        }
    });
    dispatch_one(&mut control_rx, &models, &slot).await;
    let job = submit.await.expect("task").expect("admitted");
    assert_eq!(job.instance_id(), "embedded-1");
    let completed = job.wait().await.expect("completed");
    assert_eq!(completed.content, "LOCAL");
    assert!(local.seen_local_only.load(Ordering::Acquire));
    let seen = local
        .seen_request
        .lock()
        .expect("seen lock")
        .clone()
        .expect("local backend saw the request");
    assert!(seen.extra.is_empty() && seen.tools.is_none());
    assert!(
        peer.seen_request.lock().expect("peer lock").is_none(),
        "the remote trap never received the prompt"
    );
    assert_eq!(local_leases.in_flight(), 0);
    assert_eq!(slot.available_permits(), 1);

    // A model known only to the routing table is not loaded locally: no
    // fallback to the router, no content leaves.
    let submit = tokio::spawn({
        let client = client.clone();
        async move {
            client
                .submit(request_for(LocalModelSelector::ModelId(
                    "remote-only".into(),
                )))
                .await
        }
    });
    dispatch_one(&mut control_rx, &models, &slot).await;
    assert_eq!(
        submit.await.expect("task").err(),
        Some(LocalCompletionError::ModelNotLoadedLocally {
            model: "remote-only".into()
        })
    );
    assert!(
        routing.targets.contains_key("remote-only"),
        "trap stayed available"
    );

    // An ineligible local record is refused before any content is passed.
    let submit = tokio::spawn({
        let client = client.clone();
        async move {
            client
                .submit(request_for(LocalModelSelector::InstanceId(
                    "embedded-2".into(),
                )))
                .await
        }
    });
    dispatch_one(&mut control_rx, &models, &slot).await;
    assert!(matches!(
        submit.await.expect("task"),
        Err(LocalCompletionError::Ineligible {
            reason: LocalCompletionIneligibility::SplitStage { .. },
            ..
        })
    ));
    assert!(split.seen_request.lock().expect("split lock").is_none());
    assert!(peer.seen_request.lock().expect("peer lock").is_none());

    // Unrelated control requests are handed back to the caller's match.
    let (join_tx, _join_rx) = oneshot::channel();
    let other = api::RuntimeControlRequest::Join {
        invite_token: "token".into(),
        resp: join_tx,
    };
    assert!(matches!(
        handle_local_completion_control(view(&models, &slot), other),
        Some(api::RuntimeControlRequest::Join { .. })
    ));
}

/// Drives the production outer dispatcher, `run_auto_handle_control_request`,
/// with a genuine loop context. Two traps are connected to the paths a
/// wrongly routed request would take: the routing table advertises the
/// local model id at a TCP listener that counts connections (the generic
/// ingress would proxy there), and a second model id only as a remote
/// peer. The local-only path must use the loaded record and leave both
/// traps untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_control_loop_binding_dispatches_locally_and_never_touches_routing_traps() {
    use crate::runtime::control_loop::run_auto_handle_control_request;
    use crate::runtime::dashboard::DashboardContextUsage;
    use crate::runtime::local::{RuntimeEvent, openai_guardrail_policy_handle};
    use crate::runtime::model_reconciliation::{
        ModelTargetReconciliationPolicy, ModelTargetReconciliationState,
    };
    use crate::runtime::runtime_registry::RuntimeInstanceRegistry;
    use crate::runtime::startup_handles::StartupReadyReporter;
    use crate::runtime::{RunAutoRuntimeLoopContext, RuntimeOptions, survey};
    use crate::{mesh, plugin};

    // Connected trap 1: the routing table says "qwen" is served at this
    // listener; the generic ingress would open a TCP connection to it.
    let trap = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("trap listener");
    let trap_port = trap.local_addr().expect("trap addr").port();
    let trap_hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let trap_task = tokio::spawn({
        let hits = Arc::clone(&trap_hits);
        async move {
            loop {
                if trap.accept().await.is_ok() {
                    hits.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
    });
    let mut routing = election::ModelTargets::default();
    routing
        .targets
        .entry("qwen".to_string())
        .or_default()
        .push(election::InferenceTarget::Local(trap_port));
    // Connected trap 2: a model that exists only as a remote peer.
    routing
        .targets
        .entry("remote-only".to_string())
        .or_default()
        .push(election::InferenceTarget::Remote(
            iroh::SecretKey::generate().public(),
        ));
    let (target_tx, _target_rx) = tokio::sync::watch::channel(routing);
    let target_tx = Arc::new(target_tx);

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node");
    let options = RuntimeOptions::default();
    let config = plugin::MeshConfig::default();
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    let (runtime_event_tx, _runtime_event_rx) = mpsc::unbounded_channel::<RuntimeEvent>();
    let local = FakeBackend::new(Behaviour::Respond(response(
        "LOCAL",
        Usage::new(4, 1),
        FinishReason::Stop,
    )));
    let (entry, leases) = fake_entry(
        "embedded-1",
        "qwen",
        local.clone(),
        LocalModelLoadShape::SingleStage,
    );
    let mut runtime_models = HashMap::new();
    runtime_models.insert("embedded-1".to_string(), entry);
    let mut runtime_survey_models = HashMap::new();
    let mut managed_models = HashMap::new();
    let runtime_capacity_ledger = RuntimeCapacityLedger::default();
    let mut next_runtime_instance_sequence = 1_u64;
    let runtime_instance_registry: RuntimeInstanceRegistry =
        Arc::new(tokio::sync::Mutex::new(HashMap::new()));
    let dashboard_processes = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let dashboard_context_usage: DashboardContextUsage =
        Arc::new(tokio::sync::Mutex::new(HashMap::new()));
    let slot = new_local_completion_slot();
    let survey_telemetry = survey::SurveyTelemetry::disabled();
    let startup_ready_reporter =
        StartupReadyReporter::new(&[], "http://127.0.0.1:0/v1".to_string(), None, 0, None);
    let openai_guardrail_policy =
        openai_guardrail_policy_handle(openai_frontend::GuardrailMode::Disabled);
    let mut ctx = RunAutoRuntimeLoopContext {
        options: &options,
        config: &config,
        node: &node,
        primary_model_name: "qwen",
        target_tx: &target_tx,
        control_tx: &control_tx,
        runtime_models: &mut runtime_models,
        runtime_survey_models: &mut runtime_survey_models,
        managed_models: &mut managed_models,
        runtime_capacity_ledger: &runtime_capacity_ledger,
        next_runtime_instance_sequence: &mut next_runtime_instance_sequence,
        runtime_instance_registry: &runtime_instance_registry,
        dashboard_processes: &dashboard_processes,
        dashboard_context_usage: &dashboard_context_usage,
        local_completion_slot: &slot,
        console_state: None,
        runtime_data_producer: None,
        runtime_event_tx: &runtime_event_tx,
        survey_telemetry: &survey_telemetry,
        startup_ready_reporter: &startup_ready_reporter,
        openai_guardrail_policy: &openai_guardrail_policy,
        model_target_reconciliation_policy: ModelTargetReconciliationPolicy::default(),
        model_target_reconciliation_state: ModelTargetReconciliationState::default(),
    };
    let client = LocalCompletionClient::new(control_tx.clone(), Arc::new(Semaphore::new(1)));

    // Listing through the real binding.
    let listing = tokio::spawn({
        let client = client.clone();
        async move { client.local_models().await }
    });
    let cmd = control_rx.recv().await.expect("listing request");
    assert!(
        !run_auto_handle_control_request(&mut ctx, cmd).await,
        "a listing is not a shutdown"
    );
    let listed = listing.await.expect("task").expect("listing");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].model_id, "qwen");
    assert_eq!(listed[0].eligibility, LocalCompletionEligibility::Eligible);

    // A model id that is BOTH loaded locally and advertised at the trap
    // listener runs on the loaded record; the trap sees no connection.
    let submit = tokio::spawn({
        let client = client.clone();
        async move {
            client
                .submit(request_for(LocalModelSelector::ModelId("qwen".into())))
                .await
        }
    });
    let cmd = control_rx.recv().await.expect("submit request");
    assert!(!run_auto_handle_control_request(&mut ctx, cmd).await);
    let job = submit.await.expect("task").expect("admitted");
    assert_eq!(job.instance_id(), "embedded-1");
    let completed = job.wait().await.expect("completed");
    assert_eq!(completed.content, "LOCAL");
    assert!(local.seen_local_only.load(Ordering::Acquire));

    // A model known only to the routing table never falls back to it.
    let submit = tokio::spawn({
        let client = client.clone();
        async move {
            client
                .submit(request_for(LocalModelSelector::ModelId(
                    "remote-only".into(),
                )))
                .await
        }
    });
    let cmd = control_rx.recv().await.expect("submit request");
    assert!(!run_auto_handle_control_request(&mut ctx, cmd).await);
    assert_eq!(
        submit.await.expect("task").err(),
        Some(LocalCompletionError::ModelNotLoadedLocally {
            model: "remote-only".into()
        })
    );

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        trap_hits.load(Ordering::SeqCst),
        0,
        "the generic ingress target was never contacted"
    );
    assert_eq!(leases.in_flight(), 0);
    assert_eq!(slot.available_permits(), 1);
    trap_task.abort();
}

#[tokio::test]
async fn unload_drains_an_in_flight_local_completion_before_releasing_the_model() {
    let backend = FakeBackend::new(Behaviour::IgnoreCancellationUntilReleased);
    let (entry, leases) = fake_entry(
        "embedded-1",
        "qwen",
        backend.clone(),
        LocalModelLoadShape::SingleStage,
    );
    let mut models = HashMap::new();
    models.insert("embedded-1".to_string(), entry);
    let slot = new_local_completion_slot();
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    let client = LocalCompletionClient::new(control_tx, Arc::new(Semaphore::new(1)));

    let submit = tokio::spawn({
        let client = client.clone();
        async move {
            client
                .submit(request_for(LocalModelSelector::InstanceId(
                    "embedded-1".into(),
                )))
                .await
        }
    });
    dispatch_one(&mut control_rx, &models, &slot).await;
    let mut job = submit.await.expect("task").expect("admitted");
    assert_eq!(leases.in_flight(), 1);

    // Production unload path: the record leaves the map and the handle is
    // shut down. Shutdown must wait for the in-flight completion.
    let entry = models.remove("embedded-1").expect("loaded record");
    let shutdown = tokio::spawn(entry.handle.shutdown());
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !shutdown.is_finished(),
        "shutdown waits while the backend still runs"
    );
    assert!(
        job.is_cancelled(),
        "the drain cancelled the completion cooperatively"
    );
    backend.release.notify_one();
    tokio::time::timeout(Duration::from_secs(2), shutdown)
        .await
        .expect("shutdown completes once the backend returned")
        .expect("shutdown task");
    assert_eq!(leases.in_flight(), 0);
    assert_eq!(
        job.wait_until(tokio::time::Instant::now() + Duration::from_secs(2))
            .await,
        Some(Err(LocalCompletionError::Cancelled))
    );
    assert_eq!(slot.available_permits(), 1);
}

fn test_request() -> LocalCompletionRequest {
    LocalCompletionRequest {
        model: LocalModelSelector::InstanceId("embedded-1".into()),
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
    }
}

struct Supervised {
    handle: tokio::task::JoinHandle<()>,
    job: LocalCompletionJob,
    leases: Arc<LocalCompletionLeases>,
    runtime_slot: Arc<Semaphore>,
    client_slot: Arc<Semaphore>,
    backend: Arc<FakeBackend>,
}

fn supervised(behaviour: Behaviour) -> Supervised {
    let backend = FakeBackend::new(behaviour);
    let leases = LocalCompletionLeases::new();
    let runtime_slot = Arc::new(Semaphore::new(1));
    let client_slot = Arc::new(Semaphore::new(1));
    let cancellation = CancellationToken::new();
    let lease = leases.begin(cancellation.clone());
    let runtime_permit = runtime_slot
        .clone()
        .try_acquire_owned()
        .expect("runtime slot");
    let client_permit = client_slot
        .clone()
        .try_acquire_owned()
        .expect("client slot");
    let (outcome_tx, outcome_rx) = oneshot::channel();
    let execution = LocalCompletionExecution {
        backend: backend.clone(),
        identity: LocalCompletionIdentity {
            model_id: "qwen".into(),
            instance_id: "embedded-1".into(),
        },
        request: test_request(),
        cancellation: cancellation.clone(),
        admitted_at: std::time::Instant::now(),
    };
    let handle = spawn_supervisor(
        execution,
        TerminalOwnership::new(lease, runtime_permit, client_permit, outcome_tx),
    );
    let job = LocalCompletionJob::new(
        cancellation,
        LocalCompletionAdmitted {
            instance_id: "embedded-1".into(),
            model_id: "qwen".into(),
            outcome: outcome_rx,
        },
    );
    Supervised {
        handle,
        job,
        leases,
        runtime_slot,
        client_slot,
        backend,
    }
}

#[tokio::test]
async fn normal_completion_releases_lease_and_permits_before_reporting() {
    let mut supervised = supervised(Behaviour::Respond(response(
        "pong",
        Usage::new(4, 1),
        FinishReason::Stop,
    )));
    let outcome = supervised
        .job
        .wait_until(tokio::time::Instant::now() + Duration::from_secs(2))
        .await
        .expect("terminal outcome");
    assert_eq!(outcome.expect("completed").content, "pong");
    // Observed after the outcome arrived: every hold is already released.
    assert_eq!(supervised.leases.in_flight(), 0);
    assert_eq!(supervised.runtime_slot.available_permits(), 1);
    assert_eq!(supervised.client_slot.available_permits(), 1);
    supervised
        .handle
        .await
        .expect("supervisor finished cleanly");
}

#[tokio::test]
async fn dropping_the_job_keeps_holds_until_the_backend_actually_returns() {
    let supervised = supervised(Behaviour::IgnoreCancellationUntilReleased);
    let Supervised {
        handle,
        job,
        leases,
        runtime_slot,
        client_slot,
        backend,
    } = supervised;
    drop(job); // caller gave up: cancels cooperatively
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(leases.in_flight(), 1, "lease held while the backend runs");
    assert_eq!(runtime_slot.available_permits(), 0);
    assert_eq!(client_slot.available_permits(), 0);
    backend.release.notify_one();
    handle.await.expect("supervisor finished");
    assert_eq!(leases.in_flight(), 0);
    assert_eq!(runtime_slot.available_permits(), 1);
    assert_eq!(client_slot.available_permits(), 1);
}

#[tokio::test]
async fn aborting_the_supervisor_keeps_holds_and_reports_termination_unknown() {
    let supervised = supervised(Behaviour::IgnoreCancellationUntilReleased);
    let Supervised {
        handle,
        job,
        leases,
        runtime_slot,
        client_slot,
        backend,
    } = supervised;
    tokio::time::sleep(Duration::from_millis(20)).await;
    handle.abort();
    let outcome = job.wait().await;
    assert_eq!(outcome, Err(LocalCompletionError::TerminationUnknown));
    // Nothing can prove the blocking generation stopped, so nothing is
    // released, even after the fake backend later returns.
    backend.release.notify_one();
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(leases.in_flight(), 1);
    assert_eq!(runtime_slot.available_permits(), 0);
    assert_eq!(client_slot.available_permits(), 0);
}

#[tokio::test]
async fn cancellation_outcome_arrives_only_after_release() {
    let mut supervised = supervised(Behaviour::HonourCancellation);
    supervised.job.cancel();
    let outcome = supervised
        .job
        .wait_until(tokio::time::Instant::now() + Duration::from_secs(2))
        .await
        .expect("terminal outcome");
    assert_eq!(outcome, Err(LocalCompletionError::Cancelled));
    assert_eq!(supervised.leases.in_flight(), 0);
    assert_eq!(supervised.client_slot.available_permits(), 1);
    assert!(
        supervised
            .backend
            .returned
            .load(std::sync::atomic::Ordering::Acquire)
    );
}

struct FakeCandidate {
    instance_id: &'static str,
    model_id: &'static str,
    eligibility: LocalCompletionEligibility,
}

impl LocalCompletionCandidate for FakeCandidate {
    fn instance_id(&self) -> &str {
        self.instance_id
    }
    fn model_id(&self) -> &str {
        self.model_id
    }
    fn model_ref(&self) -> &str {
        self.model_id
    }
    fn profile(&self) -> &str {
        ""
    }
    fn context_length(&self) -> u32 {
        4096
    }
    fn backend_label(&self) -> &str {
        "skippy"
    }
    fn eligibility(&self) -> LocalCompletionEligibility {
        self.eligibility.clone()
    }
    fn lease_backend(&self, _cancellation: CancellationToken) -> Option<LeasedLocalBackend> {
        None
    }
}

fn eligible(instance_id: &'static str, model_id: &'static str) -> FakeCandidate {
    FakeCandidate {
        instance_id,
        model_id,
        eligibility: LocalCompletionEligibility::Eligible,
    }
}

#[test]
fn resolver_fails_closed_for_unknown_models() {
    let candidates = [eligible("embedded-1", "qwen")];
    let error = resolve_local_completion_target(
        candidates.iter(),
        &LocalModelSelector::ModelId("other".into()),
    )
    .err()
    .expect("unknown model must not resolve");
    assert_eq!(
        error,
        LocalCompletionError::ModelNotLoadedLocally {
            model: "other".into()
        }
    );
}

#[test]
fn resolver_rejects_ambiguous_model_ids_but_accepts_instance_ids() {
    let candidates = [
        eligible("embedded-1", "qwen"),
        eligible("embedded-2", "qwen"),
    ];
    let error = resolve_local_completion_target(
        candidates.iter(),
        &LocalModelSelector::ModelId("qwen".into()),
    )
    .err()
    .expect("two instances of one model id are ambiguous");
    assert!(
        matches!(error, LocalCompletionError::Ambiguous { ref instances, .. } if instances.len() == 2)
    );
    let resolved = resolve_local_completion_target(
        candidates.iter(),
        &LocalModelSelector::InstanceId("embedded-2".into()),
    )
    .expect("instance id selects exactly one");
    assert_eq!(resolved.instance_id(), "embedded-2");
}

#[test]
fn resolver_refuses_ineligible_models_before_any_content_is_used() {
    let candidates = [FakeCandidate {
        instance_id: "embedded-1",
        model_id: "qwen",
        eligibility: LocalCompletionEligibility::Ineligible(
            LocalCompletionIneligibility::SplitStage {
                stage_id: "stage-0".into(),
            },
        ),
    }];
    let error = resolve_local_completion_target(
        candidates.iter(),
        &LocalModelSelector::InstanceId("embedded-1".into()),
    )
    .err()
    .expect("split stage must not resolve");
    assert!(matches!(
        error,
        LocalCompletionError::Ineligible {
            reason: LocalCompletionIneligibility::SplitStage { .. },
            ..
        }
    ));
}
