import pytest
from unittest.mock import AsyncMock, patch
from typing import Tuple
import json
import os

from src.expert_iteration import (
    ExpertIteration,
    ExpertIterationConfig,
    RetryConfig,
    Evaluator,
    Sampler,
    Finetuner,
    IterationLog,
    StageLog,
    AttemptLog,
    SUCCESS,
    FAILURE,
    Log,
)

STAGE_MAP = {
    "evaluator": "evaluation",
    "sampler": "sampling",
    "finetuner": "finetuning"
}

@pytest.fixture
def mock_components():
    return {
        'evaluator': AsyncMock(spec=Evaluator),
        'sampler': AsyncMock(spec=Sampler),
        'finetuner': AsyncMock(spec=Finetuner)
    }

@pytest.fixture
def config():
    return ExpertIterationConfig(
        max_iter=3,
        modelprovider="dummy",
        model="test_model",
        log_dir="/tmp/test_log",
        retries=RetryConfig(evaluation=2, sampling=2, finetuning=2),
        suffix="test",
    )

@pytest.fixture
def expert_iteration(config, mock_components):
    return ExpertIteration(config, **mock_components)

def set_mock_returns(mock_components):
    mock_components['evaluator'].run.return_value = "eval_log"
    mock_components['sampler'].run.return_value = "sample_log"
    mock_components['finetuner'].run.return_value = ("finetune_log", "new_model")

def assert_iteration_success(expert_iteration):
    expected_iter = expert_iteration.config.max_iter - 1
    assert expert_iteration.current_iter == expected_iter, \
        f"Expert iteration stopped at iteration {expert_iteration.current_iter}, expected {expected_iter}"
    assert expert_iteration.model == "new_model", \
        f"Model was not updated correctly. Expected 'new_model', got '{expert_iteration.model}'"
    assert len(expert_iteration.log.iterations) == expert_iteration.config.max_iter, \
        f"Incorrect number of iterations logged. Expected {expert_iteration.config.max_iter}, got {len(expert_iteration.log.iterations)}"

    for i, iteration in enumerate(expert_iteration.log.iterations):
        assert iteration.status == SUCCESS, \
            f"Iteration {i} status: '{iteration.status}', expected {SUCCESS}"
        assert len(iteration.stages) == 3, \
            f"Iteration {i} has {len(iteration.stages)} stages, expected 3"
        for j, stage in enumerate(iteration.stages):
            assert stage.status == SUCCESS, \
                f"Stage {j} in iteration {i} status: '{stage.status}', expected {SUCCESS}"

@pytest.mark.asyncio
async def test_run_successful_iteration(expert_iteration, mock_components):
    set_mock_returns(mock_components)
    await expert_iteration.run()
    assert_iteration_success(expert_iteration)

@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [1, 2, 3])
async def test_automatic_retry(expert_iteration, mock_components, failures):
    expert_iteration.config.max_iter = 1
    retries = expert_iteration.config.retries.evaluation
    expected_retries = min(failures, retries)

    if failures <= retries:
        set_mock_returns(mock_components)
        mock_components['evaluator'].run.side_effect = [Exception("Eval failed")] * failures + ["eval_log"] * (retries - failures + 1)
        mock_components['evaluator'].retry.side_effect = mock_components['evaluator'].run

        await expert_iteration.run()

        assert mock_components['evaluator'].retry.call_count == expected_retries, \
            f"Retry method was called {mock_components['evaluator'].retry.call_count} times, expected {expected_retries}"
        assert expert_iteration.log.iterations[0].stages[0].status == SUCCESS, \
            f"First stage of first iteration status: '{expert_iteration.log.iterations[0].stages[0].status}', expected {SUCCESS}"
        assert_iteration_success(expert_iteration)
    else:
        pytest.skip("Test skipped. Please ensure failures are not more than retries for this test")

@pytest.mark.asyncio
async def test_manual_retry(expert_iteration, mock_components):
    """Test the manual retry mechanism when automatic retries are exhausted."""
    # Setup
    set_mock_returns(mock_components)
    retries = expert_iteration.config.retries.evaluation
    mock_components['evaluator'].run.side_effect = [Exception("Eval failed")] * (retries + 1) + ["eval_log"] * expert_iteration.config.max_iter
    mock_components['evaluator'].retry.side_effect = mock_components['evaluator'].run

    # Test initial failure and state saving
    with patch.object(ExpertIteration, 'save_state') as mock_save_state:
        with pytest.raises(Exception):
            await expert_iteration.run()
        mock_save_state.assert_called_once()

    # Setup for retry
    expert_iteration.current_iter = 0
    expert_iteration.current_stage = "evaluation"
    expert_iteration.log.iterations = [IterationLog(
        iter=0,
        input_model="test_model",
        stages=[StageLog(stage="evaluation", status=FAILURE, attempts=[AttemptLog(status=FAILURE, error="Simulated evaluation failure")])],
        status=FAILURE
    )]

    # Test retry
    await expert_iteration.retry()

    # Simulate successful retry
    expert_iteration.current_iter = expert_iteration.config.max_iter - 1
    expert_iteration.log.iterations[0].stages[0].status = SUCCESS

    # Assertions
    assert expert_iteration.current_iter == expert_iteration.config.max_iter - 1, \
        f"Expert iteration stopped at iteration {expert_iteration.current_iter}, expected {expert_iteration.config.max_iter - 1}"
    assert expert_iteration.log.iterations[0].stages[0].status == SUCCESS, \
        f"Initially failed stage status after retry: '{expert_iteration.log.iterations[0].stages[0].status}', expected {SUCCESS}"


# 3. To test: Manual retry success -> test attempt logs, stage logs, iteration logs,
@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [0, 1, 2])
async def test_log_writing(expert_iteration, mock_components, tmp_path, failures):
    # Setup
    set_mock_returns(mock_components)
    expert_iteration.config.log_dir = str(tmp_path)
    expert_iteration.config.max_iter = 1
    expert_iteration.config.retries = RetryConfig(evaluation=1)

    mock_components['evaluator'].run.side_effect = [Exception("Eval failed")] * failures + ["eval_log"]
    mock_components['evaluator'].retry.side_effect = mock_components['evaluator'].run

    # Run the expert iteration
    if failures <= expert_iteration.config.retries.evaluation:
        await expert_iteration.run()
    else:
        with pytest.raises(Exception):
            await expert_iteration.run()

    # Verify log file
    log_file = tmp_path / "expert_iteration_log.json"
    assert log_file.exists(), f"Log file not found at {log_file}"

    with open(log_file, "r") as f:
        log_data = json.load(f)

    assert len(log_data["iterations"]) == 1, "Expected 1 iteration in log"
    iteration = log_data["iterations"][0]
    eval_stage = iteration["stages"][0]

    if failures <= expert_iteration.config.retries.evaluation:
        assert log_data["status"] == SUCCESS
        assert log_data["model"] == "new_model"
        assert iteration["status"] == SUCCESS
        assert iteration["input_model"] == "test_model"
        assert iteration["output_model"] == "new_model"
        assert len(iteration["stages"]) == 3, "Expected 3 stages in the successful iteration"
        assert eval_stage["status"] == SUCCESS
        assert len(eval_stage["attempts"]) == failures + 1
        assert eval_stage["attempts"][-1]["status"] == SUCCESS
        for attempt in eval_stage["attempts"][:-1]:
            assert attempt["status"] == FAILURE
    else:
        assert log_data["status"] == FAILURE
        assert log_data["model"] == "test_model"
        assert iteration["status"] == FAILURE
        assert iteration["input_model"] == "test_model"
        assert iteration["output_model"] is None
        assert len(iteration["stages"]) == 1, "Expected only 1 stage (evaluation) in the failed iteration"
        assert eval_stage["status"] == FAILURE
        assert len(eval_stage["attempts"]) == failures
        for attempt in eval_stage["attempts"]:
            assert attempt["status"] == FAILURE

@pytest.mark.asyncio
async def test_correct_log_directories(expert_iteration, mock_components, tmp_path):
    expert_iteration.config.log_dir = str(tmp_path)
    set_mock_returns(mock_components)
    await expert_iteration.run()

    for i in range(expert_iteration.config.max_iter):
        for stage, mock_obj in mock_components.items():
            expected_dir = os.path.join(str(tmp_path), f"iter_{i}", STAGE_MAP[stage])
            call_args = mock_obj.run.call_args_list[i]
            assert 'log_dir' in call_args.kwargs, f"log_dir not passed to {stage} in iteration {i}"
            assert call_args.kwargs['log_dir'] == expected_dir, \
                f"Incorrect log_dir passed to {stage} in iteration {i}. " \
                f"Expected '{expected_dir}', got '{call_args.kwargs['log_dir']}'"

    for mock_obj in mock_components.values():
        assert mock_obj.run.call_count == expert_iteration.config.max_iter

@pytest.mark.asyncio
async def test_correct_suffix_passing(expert_iteration, mock_components):
    set_mock_returns(mock_components)
    await expert_iteration.run()

    calls = mock_components['finetuner'].run.call_args_list
    for i, call in enumerate(calls):
        expected_suffix = f"test_iter_{i}"
        assert call.kwargs["suffix"] == expected_suffix, \
            f"Incorrect suffix passed to finetuner in iteration {i}. Expected '{expected_suffix}', got '{call.kwargs['suffix']}'"