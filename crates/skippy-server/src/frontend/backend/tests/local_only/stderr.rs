//! Exercise the environment-enabled stderr sink in isolated test processes.
use super::*;

const PROBE: &str =
    "frontend::backend::tests::local_only::stderr::local_only_stderr_generation_probe";
const SCOPE_ENV: &str = "SKIPPY_LOCAL_ONLY_STDERR_SCOPE";

/// Removing the local telemetry cap must expose the Debug driver record here.
/// Child processes initialize the stderr OnceLock from their own environment.
#[test]
#[ignore = "needs the real GGUF and native runtime fixture"]
fn local_only_policy_stderr_override_suppresses_debug_and_token_records() {
    let ordinary = probe_records("ordinary");
    assert!(
        ordinary
            .iter()
            .any(|attrs| attrs.contains_key("llama_stage.predicted_token")),
        "the environment-enabled ordinary stderr sink must emit generated token IDs"
    );
    assert!(
        ordinary
            .iter()
            .any(|attrs| attrs.contains_key("skippy.scheduler.generation_driver")),
        "the environment override must enable the ordinary Debug path"
    );

    let local = probe_records("local");
    assert!(
        !local.is_empty(),
        "local Summary telemetry must reach stderr"
    );
    for attrs in &local {
        assert!(
            !attrs.contains_key("skippy.scheduler.generation_driver"),
            "local stderr must suppress Debug records despite the environment override"
        );
        assert!(
            !attrs.contains_key("llama_stage.predicted_token")
                && !attrs.contains_key("skippy.kv.predicted_token"),
            "local stderr must not contain generated token IDs"
        );
    }
}

fn probe_records(scope: &str) -> Vec<serde_json::Map<String, serde_json::Value>> {
    let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
        .args(["--exact", PROBE, "--ignored", "--nocapture"])
        .env(SCOPE_ENV, scope)
        .env("SKIPPY_TELEMETRY_STDERR", "1")
        .output()
        .expect("run isolated stderr generation probe");
    assert!(
        output.status.success(),
        "{scope} generation probe failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8_lossy(&output.stderr)
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|record| {
            record
                .get("attributes")
                .and_then(|attrs| attrs.as_object())
                .cloned()
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "child probe run by the stderr override regression"]
async fn local_only_stderr_generation_probe() {
    let scope = std::env::var(SCOPE_ENV).expect("probe scope supplied by parent test");
    assert!(matches!(scope.as_str(), "ordinary" | "local"));
    let fixture = fixture::real_decode_fixture(true);
    let context = if scope == "local" {
        OpenAiRequestContext::new().with_local_only_policy()
    } else {
        OpenAiRequestContext::new()
    };
    let response = fixture
        .backend
        .chat_completion_with_context(fixture.request, context)
        .await
        .expect("stderr probe generation succeeds");
    assert!(response.usage.completion_tokens > 0);
    assert!(
        response.choices[0]
            .message
            .content
            .as_deref()
            .is_some_and(|content| !content.trim().is_empty())
    );
}
