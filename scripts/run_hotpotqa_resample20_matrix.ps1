$ErrorActionPreference = "Stop"

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $workspaceRoot ".venv\Scripts\python.exe"
$runnerPath = Join-Path $workspaceRoot "scripts\run_hotpotqa_eval.py"
$splitPath = Join-Path $workspaceRoot "data\evaluations\hotpotqa_resample20_seed20260805\questions.jsonl"
$substratePath = Join-Path $workspaceRoot "artifacts\hotpotqa_benchmark_exact"
$outputRoot = Join-Path $workspaceRoot "runs\hotpotqa_resample20_seed20260805"

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$userOpenAIKey = [Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "User")
if (-not $userOpenAIKey) {
    throw "OPENAI_API_KEY is missing from the Windows User environment"
}
$env:OPENAI_API_KEY = $userOpenAIKey
$userOpenAIKey = $null

$jobs = @(
    @{
        Label = "luna_v2_a"
        Config = "configs\agentic_hotpotqa_luna_v2.yaml"
        Skill = "skills\hotpotqa_v2_a_evidence_adaptive.md"
        Output = "luna\v2\a_evidence_adaptive"
    },
    @{
        Label = "luna_v2_b"
        Config = "configs\agentic_hotpotqa_luna_v2.yaml"
        Skill = "skills\hotpotqa_v2_b_chunk_entity_expert.md"
        Output = "luna\v2\b_chunk_entity_expert"
    },
    @{
        Label = "luna_v2_c"
        Config = "configs\agentic_hotpotqa_luna_v2.yaml"
        Skill = "skills\hotpotqa_v2_c_posthoc_guarded.md"
        Output = "luna\v2\c_posthoc_guarded"
    },
    @{
        Label = "luna_v3_a"
        Config = "configs\agentic_hotpotqa_luna_v3.yaml"
        Skill = "skills\hotpotqa_v3_a_baseline_clean.md"
        Output = "luna\v3\a_baseline_clean"
    },
    @{
        Label = "luna_v3_b"
        Config = "configs\agentic_hotpotqa_luna_v3.yaml"
        Skill = "skills\hotpotqa_v3_b_chunk_entity_expert.md"
        Output = "luna\v3\b_chunk_entity_expert"
    },
    @{
        Label = "luna_v3_c"
        Config = "configs\agentic_hotpotqa_luna_v3.yaml"
        Skill = "skills\hotpotqa_v3_c_posthoc_guarded.md"
        Output = "luna\v3\c_posthoc_guarded"
    },
    @{
        Label = "qwen_v2_a"
        Config = "configs\agentic_hotpotqa_qwen35_9b_v2.yaml"
        Skill = "skills\hotpotqa_v2_a_evidence_adaptive.md"
        Output = "qwen\v2\a_evidence_adaptive"
    },
    @{
        Label = "qwen_v2_b"
        Config = "configs\agentic_hotpotqa_qwen35_9b_v2.yaml"
        Skill = "skills\hotpotqa_v2_b_chunk_entity_expert.md"
        Output = "qwen\v2\b_chunk_entity_expert"
    },
    @{
        Label = "qwen_v2_c"
        Config = "configs\agentic_hotpotqa_qwen35_9b_v2.yaml"
        Skill = "skills\hotpotqa_v2_c_posthoc_guarded.md"
        Output = "qwen\v2\c_posthoc_guarded"
    },
    @{
        Label = "qwen_v3_a"
        Config = "configs\agentic_hotpotqa_qwen35_9b_v3.yaml"
        Skill = "skills\hotpotqa_v3_a_baseline_clean.md"
        Output = "qwen\v3\a_baseline_clean"
    },
    @{
        Label = "qwen_v3_b"
        Config = "configs\agentic_hotpotqa_qwen35_9b_v3.yaml"
        Skill = "skills\hotpotqa_v3_b_chunk_entity_expert.md"
        Output = "qwen\v3\b_chunk_entity_expert"
    },
    @{
        Label = "qwen_v3_c"
        Config = "configs\agentic_hotpotqa_qwen35_9b_v3.yaml"
        Skill = "skills\hotpotqa_v3_c_posthoc_guarded.md"
        Output = "qwen\v3\c_posthoc_guarded"
    }
)

Set-Location -LiteralPath $workspaceRoot
foreach ($job in $jobs) {
    $configPath = Join-Path $workspaceRoot $job.Config
    $skillPath = Join-Path $workspaceRoot $job.Skill
    $jobOutput = Join-Path $outputRoot $job.Output
    $summaryPath = Join-Path $jobOutput "summary.json"
    if (Test-Path -LiteralPath $summaryPath) {
        Write-Output "MATRIX SKIP $($job.Label) summary already exists"
        continue
    }

    $runnerArguments = @(
        $runnerPath,
        "--substrate", $substratePath,
        "--split", $splitPath,
        "--config", $configPath,
        "--skill", $skillPath,
        "--output", $jobOutput,
        "--expected-count", "20"
    )
    if (Test-Path -LiteralPath $jobOutput) {
        $runnerArguments += "--resume"
    }

    Write-Output "MATRIX START $($job.Label)"
    & $pythonExecutable @runnerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Matrix job failed: $($job.Label)"
    }
    Write-Output "MATRIX DONE $($job.Label)"
}

Write-Output "MATRIX COMPLETE"
