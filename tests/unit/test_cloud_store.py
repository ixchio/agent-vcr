"""Phase 7 tests: Sandbox Runtime, Long-term Vector Memory, and Distributed Workflows.

Validates:
1. Sandbox isolated execution, timeouts, filesystem diffs, and VCR recording.
2. Long-term semantic vector memory with secret redaction and VCR session recall.
3. Pluggable distributed workflow backends (Local and Inngest with step memoization).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder
from agent_vcr.sandbox import (
    E2BSandbox,
    LocalSandbox,
    SandboxConfig,
    create_sandbox,
)
from agent_vcr.superagent import ToolGateway
from agent_vcr.superagent.vector_memory import (
    InMemoryVectorStore,
    MemoryRecord,
    QdrantVectorStore,
    create_vector_memory,
    default_embedder,
)
from agent_vcr.workflows import (
    RunRecord,
    RunStatus,
    StepKind,
    WorkflowDef,
    WorkflowEngine,
    WorkflowStep,
)
from agent_vcr.workflows.backend import (
    InngestWorkflowBackend,
    LocalWorkflowBackend,
    create_workflow_backend,
)

# ===========================================================================
# 1. Sandbox Runtime Tests
# ===========================================================================


class TestLocalSandbox:
    """Tests for LocalSandbox isolation, command execution, and filesystem diffs."""

    def test_command_execution_success(self, tmp_path: Path) -> None:
        with LocalSandbox(workdir=tmp_path / "sandbox") as sb:
            assert sb.is_alive
            res = sb.execute_command("echo 'Agent VCR Sandbox'")
            assert res.success
            assert res.exit_code == 0
            assert "Agent VCR Sandbox" in res.stdout
            assert not res.timed_out
            assert res.duration_ms >= 0.0

    def test_command_failure_and_stderr(self, tmp_path: Path) -> None:
        with LocalSandbox(workdir=tmp_path / "sandbox") as sb:
            res = sb.execute_command("ls /path/does/not/exist_12345")
            assert not res.success
            assert res.exit_code != 0
            assert len(res.stderr) > 0 or len(res.stdout) > 0

    def test_command_timeout(self, tmp_path: Path) -> None:
        config = SandboxConfig(timeout_seconds=0.3)
        with LocalSandbox(config=config, workdir=tmp_path / "sandbox") as sb:
            res = sb.execute_command("sleep 2", timeout=0.2)
            assert res.timed_out
            assert not res.success
            assert res.exit_code == -1
            assert res.error is not None
            assert "timed out" in res.error.lower()

    def test_environment_variables(self, tmp_path: Path) -> None:
        config = SandboxConfig(env={"SANDBOX_VAR": "vcr_active"})
        with LocalSandbox(config=config, workdir=tmp_path / "sandbox") as sb:
            res = sb.execute_command("echo $SANDBOX_VAR")
            assert res.success
            assert "vcr_active" in res.stdout

            # Test per-command override
            res_override = sb.execute_command("echo $EXTRA_VAR", env={"EXTRA_VAR": "custom_override"})
            assert res_override.success
            assert "custom_override" in res_override.stdout

    def test_file_read_write(self, tmp_path: Path) -> None:
        with LocalSandbox(workdir=tmp_path / "sandbox") as sb:
            sb.write_file("nested/dir/notes.txt", "Sandbox file content")
            content = sb.read_file("nested/dir/notes.txt")
            assert content == "Sandbox file content"

            # Binary read/write
            sb.write_file("binary.bin", b"\x00\x01\x02\x03")
            b_content = sb.read_file("binary.bin", binary=True)
            assert b_content == b"\x00\x01\x02\x03"

            files = sb.list_files()
            assert "nested/dir/notes.txt" in files
            assert "binary.bin" in files

    def test_path_traversal_prevention(self, tmp_path: Path) -> None:
        with LocalSandbox(workdir=tmp_path / "sandbox") as sb:
            with pytest.raises(PermissionError):
                sb.read_file("../../outside.txt")

            with pytest.raises(PermissionError):
                sb.write_file("../forbidden.txt", "escape")

    def test_filesystem_snapshot_and_diff(self, tmp_path: Path) -> None:
        with LocalSandbox(workdir=tmp_path / "sandbox") as sb:
            # 1. Initial snapshot with file1
            sb.write_file("file1.txt", "version 1")
            snap1 = sb.snapshot_filesystem()
            assert "file1.txt" in snap1.entries
            assert snap1.entries["file1.txt"].size == len("version 1")

            # 2. Add file2, modify file1, delete nothing yet
            sb.write_file("file2.txt", "hello file 2")
            sb.write_file("file1.txt", "version 2 (modified)")
            snap2 = sb.snapshot_filesystem()

            diff1 = snap1.diff(snap2)
            assert diff1.has_changes
            assert "file2.txt" in diff1.added
            assert "file1.txt" in diff1.modified
            assert diff1.deleted == []

            # 3. Delete file2
            target = sb.root_path / "file2.txt"
            target.unlink()
            diff2 = sb.diff_filesystem(snap2)
            assert "file2.txt" in diff2.deleted

    def test_vcr_recording_integration(self, tmp_path: Path) -> None:
        rec_dir = tmp_path / "vcr"
        recorder = VCRRecorder(output_dir=str(rec_dir), auto_save=False)
        recorder.start_session("sandbox_test_session")

        with LocalSandbox(workdir=tmp_path / "sandbox") as sb:
            sb.write_file("code.py", "print('hello')")
            snap = sb.snapshot_filesystem()

            res = sb.execute_command("python3 code.py")
            sb.write_file("output.txt", "done")
            diff = sb.diff_filesystem(snap)

            # Record step in VCR
            sb.record_execution(recorder, "python3 code.py", res, diff=diff)

        bundle_path = recorder.save()
        player = VCRPlayer.load(bundle_path)
        assert len(player.frames) == 1
        frame = player.frames[0]
        assert "sandbox_exec" in frame.node_name
        output = frame.output_state.get("result", frame.output_state)
        assert output["stdout"].strip() == "hello"
        assert "output.txt" in output["filesystem_diff"]["added"]

    def test_symlink_escape_protection(self, tmp_path: Path) -> None:
        outside_file = tmp_path / "outside_secret.txt"
        outside_file.write_text("secret_password_123")

        with LocalSandbox(workdir=tmp_path / "sandbox") as sb:
            # Create symlink pointing to external file
            symlink_target = sb.root_path / "symlink_secret"
            symlink_target.symlink_to(outside_file)

            # 1. read_file should block reading through symlink
            with pytest.raises(PermissionError):
                sb.read_file("symlink_secret")

            # 2. snapshot_filesystem should ignore external symlink
            snap = sb.snapshot_filesystem()
            assert "symlink_secret" not in snap.entries

            # 3. list_files should exclude external symlink
            files = sb.list_files()
            assert "symlink_secret" not in files

    def test_sandbox_lifecycle_restart(self) -> None:
        sb = LocalSandbox()
        sb.start()
        assert sb.is_alive
        root1 = sb.root_path
        assert root1.exists()

        sb.stop()
        assert not sb.is_alive
        with pytest.raises(RuntimeError):
            _ = sb.root_path

        # Restart creates a fresh workspace
        sb.start()
        assert sb.is_alive
        root2 = sb.root_path
        assert root2.exists()
        assert root1 != root2
        sb.stop()

    def test_vcr_record_execution_with_error(self, tmp_path: Path) -> None:
        rec_dir = tmp_path / "vcr_err"
        recorder = VCRRecorder(output_dir=str(rec_dir), auto_save=False)
        recorder.start_session("error_test")

        with LocalSandbox(workdir=tmp_path / "sandbox") as sb:
            res = sb.execute_command("nonexistent_binary_xyz_123")
            assert not res.success
            sb.record_execution(recorder, "nonexistent_binary_xyz_123", res)

        bundle_path = recorder.save()
        player = VCRPlayer.load(bundle_path)
        assert len(player.frames) == 1
        frame = player.frames[0]
        assert frame.metadata.error_message is not None or "error" in frame.output_state


class TestE2BSandboxAdapter:
    """Tests for E2B microVM adapter and factory creation."""

    def test_missing_api_key_raises(self) -> None:
        old_key = os.environ.pop("E2B_API_KEY", None)
        try:
            with pytest.raises((ValueError, RuntimeError)):
                sb = create_sandbox(provider="e2b", api_key=None)
                sb.start()
        finally:
            if old_key:
                os.environ["E2B_API_KEY"] = old_key

    def test_injected_client_execution(self) -> None:
        mock_client = MagicMock()
        mock_cmd_res = MagicMock(stdout="e2b output\n", stderr="", exit_code=0)
        mock_client.commands.run.return_value = mock_cmd_res
        mock_client.files.read.return_value = "remote file content"

        sandbox = E2BSandbox(client=mock_client)
        sandbox.start()
        assert sandbox.is_alive

        res = sandbox.execute_command("echo 123")
        assert res.success
        assert "e2b output" in res.stdout

        content = sandbox.read_file("test.txt")
        assert content == "remote file content"

        sandbox.write_file("remote.txt", "data")
        mock_client.files.write.assert_called_with("remote.txt", "data")

        sandbox.stop()
        assert not sandbox.is_alive
        mock_client.kill.assert_called_once()

    def test_factory_providers(self, tmp_path: Path) -> None:
        local_sb = create_sandbox(provider="local", workdir=tmp_path / "factory_local")
        assert isinstance(local_sb, LocalSandbox)
        local_sb.close()

        mock_client = MagicMock()
        e2b_sb = create_sandbox(provider="e2b", client=mock_client)
        assert isinstance(e2b_sb, E2BSandbox)
        e2b_sb.close()

        with pytest.raises((ValueError, Exception)):
            create_sandbox(provider="unsupported_provider")

    def test_e2b_command_error_and_code_handling(self) -> None:
        mock_client = MagicMock()
        mock_res = MagicMock(stdout="", stderr="fatal error 1", exit_code=127, error="Command not found")
        mock_client.commands.run.return_value = mock_res

        sb = E2BSandbox(client=mock_client)
        sb.start()
        res = sb.execute_command("bad_cmd")
        assert not res.success
        assert res.exit_code == 127
        assert "Command not found" in str(res.error)

    def test_e2b_snapshot_path_normalization(self) -> None:
        mock_client = MagicMock()
        # Mock sha256 output from remote find
        mock_res = MagicMock(
            stdout="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  ./src/app.py\n"
                   "d41d8cd98f00b204e9800998ecf8427e  /root/config.json\n",
            stderr="",
            exit_code=0,
            error=None,
        )
        mock_client.commands.run.return_value = mock_res

        sb = E2BSandbox(client=mock_client)
        sb.start()
        snap = sb.snapshot_filesystem(".")
        assert "src/app.py" in snap.entries
        assert "root/config.json" in snap.entries


# ===========================================================================
# 2. Long-term Semantic Vector Memory Tests
# ===========================================================================


class TestVectorMemory:
    """Tests for semantic vector memory, secret redaction, and VCR recall."""

    def test_default_embedder_is_deterministic_and_normalized(self) -> None:
        first = default_embedder("sensitive agent memory")
        second = default_embedder("sensitive agent memory")
        assert first == second
        assert sum(value * value for value in first) == pytest.approx(1.0)

    def test_add_and_search(self) -> None:
        mem = create_vector_memory(backend="memory")
        id1 = mem.add("Agent VCR provides time travel debugging for autonomous AI", metadata={"tag": "vcr"})
        id2 = mem.add("LangGraph graphs manage agent state checkpointers", metadata={"tag": "langgraph"})
        mem.add("Kubernetes clusters orchestrate container workloads", metadata={"tag": "infra"})

        assert mem.count() == 3
        record = mem.get(id1)
        assert record is not None
        assert "Agent VCR" in record.text
        assert record.metadata["tag"] == "vcr"

        # Semantic query
        results = mem.search("time travel debugging and checkpoints", limit=2)
        assert len(results) > 0
        top = results[0]
        assert top.record.id in (id1, id2)
        assert top.score > 0.0

    def test_secret_redaction_before_storage(self) -> None:
        mem = create_vector_memory(backend="memory")
        text_with_secrets = "Connecting to database with api_key=sk-1234567890abcdef and password=TopSecret123"
        rec_id = mem.add(text_with_secrets, metadata={"token": "ghp_abcdef1234567890"})

        rec = mem.get(rec_id)
        assert rec is not None
        assert "sk-1234567890abcdef" not in rec.text
        assert "TopSecret123" not in rec.text
        assert "[REDACTED]" in rec.text
        # Metadata redaction
        assert "ghp_abcdef1234567890" not in str(rec.metadata)
        assert "[REDACTED]" in str(rec.metadata["token"])

    def test_namespaces_and_filters(self) -> None:
        mem = create_vector_memory(backend="memory")
        mem.add("System message in team alpha", namespace="alpha", metadata={"role": "admin"})
        mem.add("User message in team alpha", namespace="alpha", metadata={"role": "user"})
        mem.add("System message in team beta", namespace="beta", metadata={"role": "admin"})

        assert mem.count("alpha") == 2
        assert mem.count("beta") == 1

        # Search restricted to namespace
        res_alpha = mem.search("System message", namespace="alpha")
        assert all(r.record.namespace == "alpha" for r in res_alpha)

        # Search with metadata filter
        res_filtered = mem.search("message", namespace="alpha", filter_metadata={"role": "user"})
        assert len(res_filtered) == 1
        assert res_filtered[0].record.metadata["role"] == "user"

        # Clear namespace
        deleted_count = mem.clear("alpha")
        assert deleted_count == 2
        assert mem.count("alpha") == 0
        assert mem.count("beta") == 1

    def test_batch_and_json_export(self, tmp_path: Path) -> None:
        mem = InMemoryVectorStore()
        batch = [
            MemoryRecord(text="Step 1 completed successfully", metadata={"step": 1}),
            MemoryRecord(text="Step 2 encountered warning", metadata={"step": 2}),
        ]
        ids = mem.add_batch(batch)
        assert len(ids) == 2

        export_file = tmp_path / "memory_export.json"
        mem.export_json(export_file)
        assert export_file.exists()

        mem2 = InMemoryVectorStore()
        mem2.load_json(export_file)
        assert mem2.count() == 2
        assert mem2.get(ids[0]) is not None

    def test_index_vcr_session_recall(self, tmp_path: Path) -> None:
        # 1. Create a recorded VCR session
        rec_dir = tmp_path / "vcr_session"
        recorder = VCRRecorder(output_dir=str(rec_dir), auto_save=False)
        recorder.start_session("knowledge_session")
        recorder.record_step("search_api", {"q": "python"}, {"results": ["pytest", "pydantic"]})
        recorder.record_error("deploy_step", {}, ValueError("authentication failed on port 8080"), latency_ms=1.0)
        bundle_path = recorder.save()

        # 2. Index the session into vector memory
        mem = create_vector_memory(backend="memory")
        indexed_count = mem.index_vcr_session(bundle_path, namespace="history")
        assert indexed_count == 2

        # 3. Query the past failure from memory
        results = mem.search("authentication error port 8080", namespace="history")
        assert len(results) > 0
        assert "authentication failed on port 8080" in results[0].text

    def test_qdrant_adapter_with_mock(self) -> None:
        mock_qdrant = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = "mock-id-1"
        mock_hit.score = 0.95
        mock_hit.payload = {
            "text": "Retrieved from Qdrant vector memory",
            "namespace": "default",
            "timestamp": 12345.0,
            "category": "qa",
        }
        mock_hit.vector = [0.1] * 128
        mock_qdrant.search.return_value = [mock_hit]
        mock_qdrant.get_collections.return_value.collections = []

        qdrant_store = QdrantVectorStore(client=mock_qdrant)
        qdrant_store.add("Sample knowledge snippet", metadata={"category": "qa"})
        mock_qdrant.upsert.assert_called_once()

        hits = qdrant_store.search("knowledge query")
        assert len(hits) == 1
        assert hits[0].record.text == "Retrieved from Qdrant vector memory"
        assert hits[0].score == 0.95

    def test_qdrant_custom_string_id_normalization(self) -> None:
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value.collections = []

        qdrant_store = QdrantVectorStore(client=mock_qdrant)
        # Non-UUID string ID
        custom_id = "agent-step-node-999"
        returned_id = qdrant_store.add("Sample info", id=custom_id)
        assert returned_id == custom_id

        # Verify point struct has valid UUID representation
        upsert_call = mock_qdrant.upsert.call_args
        point = upsert_call.kwargs["points"][0]
        # point.id is a normalized UUID string, not raw arbitrary text
        import uuid
        parsed_uuid = uuid.UUID(str(point.id))
        assert parsed_uuid is not None
        assert point.payload["_orig_id"] == custom_id

    def test_batch_vector_reembedding_on_secret_redaction(self) -> None:
        mem = InMemoryVectorStore()
        # Fake dummy vector representing old text
        old_vec = [1.0] * 128
        raw_text = "Database connection password=TopSecretSecret123"
        rec = MemoryRecord(id="rec-1", text=raw_text, vector=old_vec)

        mem.add_batch([rec])
        stored = mem.get("rec-1")
        assert stored is not None
        assert "TopSecretSecret123" not in stored.text
        # Vector must NOT be the old dummy vector, it must have been recomputed from redacted text
        assert stored.vector != old_vec

    def test_index_vcr_session_indexes_inputs_and_tool_errors(self, tmp_path: Path) -> None:
        rec_dir = tmp_path / "vcr_inputs"
        recorder = VCRRecorder(output_dir=str(rec_dir), auto_save=False)
        recorder.start_session("input_test_session")
        recorder.record_step("sql_query", {"query": "SELECT * FROM users"}, {"rows": 5})
        recorder.record_tool_call("api_fetch", {"url": "https://example.com/api"}, {}, 10.0, error="Connection refused: 502")
        bundle_path = recorder.save()

        mem = create_vector_memory(backend="memory")
        count = mem.index_vcr_session(bundle_path, namespace="history")
        assert count == 2

        # Search by query input
        res_input = mem.search("SELECT users", namespace="history")
        assert len(res_input) > 0
        assert "SELECT * FROM users" in res_input[0].text

        # Search by tool error
        res_error = mem.search("Connection refused 502", namespace="history")
        assert len(res_error) > 0
        assert "Connection refused: 502" in res_error[0].text


# ===========================================================================
# 3. Pluggable Distributed Workflow Backend Tests
# ===========================================================================


class TestWorkflowBackends:
    """Tests for LocalWorkflowBackend and Inngest distributed dispatch."""

    def test_local_workflow_backend(self, tmp_path: Path) -> None:
        gw = ToolGateway()

        @gw.tool("format_name", parameters={"type": "object"})
        def format_name(name: str = "") -> dict:
            return {"formatted": name.upper()}

        engine = WorkflowEngine(gateway=gw, vcr_dir=str(tmp_path / "vcr"))
        backend = LocalWorkflowBackend(engine=engine)

        wf = WorkflowDef(
            name="name_flow",
            version=1,
            steps=[
                WorkflowStep(
                    id="step1",
                    kind=WorkflowStep.model_fields["kind"].default,  # "tool"
                    ref="format_name",
                    inputs={"name": "agent vcr"},
                )
            ],
        )

        record = backend.dispatch(wf, inputs={"user": "admin"})
        assert isinstance(record, RunRecord)
        assert record.status == RunStatus.COMMITTED
        assert "step1" in record.outputs
        assert record.outputs["step1"]["formatted"] == "AGENT VCR"

        # Check get_run
        loaded = backend.get_run(record.run_id)
        assert loaded is not None
        assert loaded.run_id == record.run_id

    def test_inngest_workflow_backend_simulation(self, tmp_path: Path) -> None:
        backend = InngestWorkflowBackend(
            app_id="agent-vcr-test",
            simulate=True,
            vcr_dir=str(tmp_path / "vcr"),
        )

        wf = WorkflowDef(
            name="distributed_order",
            version=2,
            steps=[
                WorkflowStep(id="verify_cart", kind=StepKind.TOOL, ref="cart_check"),
                WorkflowStep(id="process_payment", kind=StepKind.TOOL, ref="stripe_charge"),
            ],
        )

        # 1. Format event check
        event = backend.format_event(wf, inputs={"order_id": "ord_999"})
        assert event["name"] == "agent_vcr/distributed_order.triggered"
        assert event["data"]["workflow"] == "distributed_order"
        assert event["data"]["version"] == 2
        assert event["data"]["inputs"]["order_id"] == "ord_999"
        assert event["data"]["steps"] == ["verify_cart", "process_payment"]

        # 2. Dispatch
        run_id = backend.dispatch(wf, inputs={"order_id": "ord_999"})
        assert isinstance(run_id, str)
        assert len(backend.dispatched_events) == 1

        # Check run stored in run store
        run_rec = backend.get_run(run_id)
        assert run_rec is not None
        assert run_rec.workflow == "distributed_order"

        # 3. Step memoization dispatch
        step_res = backend.dispatch_step(
            run_id,
            wf.steps[0],
            inputs={"item_count": 2},
        )
        assert step_res["memoized"] is True
        assert step_res["step_id"] == "verify_cart"
        assert len(backend.dispatched_events) == 2

        # 4. Cancellation
        cancelled = backend.cancel_run(run_id)
        assert cancelled is True
        updated_rec = backend.get_run(run_id)
        assert updated_rec is not None
        assert updated_rec.status == RunStatus.ABORTED

    def test_inngest_backend_with_mock_client(self) -> None:
        mock_client = MagicMock()
        backend = InngestWorkflowBackend(client=mock_client)

        wf = WorkflowDef(
            name="cloud_event",
            version=1,
            steps=[WorkflowStep(id="s1", kind=StepKind.TOOL, ref="ping")],
        )

        run_id = backend.dispatch(wf, inputs={"key": "value"})
        assert isinstance(run_id, str)
        mock_client.send.assert_called_once()

    def test_workflow_backend_factory(self) -> None:
        local_backend = create_workflow_backend("local")
        assert isinstance(local_backend, LocalWorkflowBackend)

        inngest_backend = create_workflow_backend("inngest", simulate_inngest=True)
        assert isinstance(inngest_backend, InngestWorkflowBackend)

        auto_backend = create_workflow_backend("auto")
        assert isinstance(auto_backend, LocalWorkflowBackend)

        with pytest.raises(ValueError, match="Unknown workflow backend"):
            create_workflow_backend("unsupported_backend")

    def test_inngest_step_memoization_and_concurrency_isolation(self, tmp_path: Path) -> None:
        backend = InngestWorkflowBackend(
            app_id="agent-vcr-memo",
            simulate=True,
            vcr_dir=str(tmp_path / "vcr"),
        )
        wf = WorkflowDef(
            name="memo_test",
            version=1,
            steps=[WorkflowStep(id="llm_generate", kind=StepKind.TOOL, ref="generate_summary")],
        )

        run1 = backend.dispatch(wf, inputs={"run": 1})
        run2 = backend.dispatch(wf, inputs={"run": 2})
        assert run1 != run2

        # Step memoization on run1
        output1 = {"summary": "Deterministic result for run 1"}
        res1 = backend.dispatch_step(run1, wf.steps[0], inputs={"prompt": "test 1"}, output=output1)
        assert res1["memoized"] is True
        assert res1["output"] == output1

        # Re-dispatching step on run1 returns cached=True without re-running
        res1_replay = backend.dispatch_step(run1, wf.steps[0], inputs={"prompt": "test 1"})
        assert res1_replay["cached"] is True
        assert res1_replay["output"] == output1

        # Run 2 has independent memoization space
        output2 = {"summary": "Deterministic result for run 2"}
        res2 = backend.dispatch_step(run2, wf.steps[0], inputs={"prompt": "test 2"}, output=output2)
        assert res2["output"] == output2
        assert res2["output"] != output1

        # Check persistence in run store
        rec1 = backend.get_run(run1)
        assert rec1 is not None
        assert "llm_generate" in rec1.outputs
        assert rec1.outputs["llm_generate"] == output1
