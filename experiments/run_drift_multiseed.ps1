# Reproduce the independent synthetic concept-drift experiment; not the paper's MLHAT.
# Example (without W&B): .\experiments\run_drift_multiseed.ps1
# Example (with W&B):    .\experiments\run_drift_multiseed.ps1 -Track
param(
    [int[]] $Seeds = @(42, 43, 44),
    [int] $Samples = 3000,
    [int] $DriftAt = 1500,
    [switch] $Track,
    [switch] $Overwrite,
    [string] $Project = 'mlhat17-reproduction'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
foreach ($seed in $Seeds) {
    foreach ($model in @('stage2', 'stage3')) {
        $existing = Join-Path $root "results\numeric_stage3_${model}_seed${seed}.json"
        if ((Test-Path $existing) -and -not $Overwrite) {
            throw "Result already exists: $existing. To rerun and replace it, pass -Overwrite."
        }
        Write-Host "RUN $model / seed $seed"
        $runArgs = @('.\experiments\run_stage3.py', '--model', $model,
                     '--samples', "$Samples", '--drift-at', "$DriftAt",
                     '--seed', "$seed")
        if ($Track) { $runArgs += @('--wandb', '--project', $Project) }
        & python @runArgs
        if ($LASTEXITCODE -ne 0) { throw "Failed: $model / seed $seed" }
    }
}
$analysisArgs = @('.\experiments\analyze_drift.py', '--seeds')
foreach ($seed in $Seeds) { $analysisArgs += "$seed" }
$analysisArgs += @('--samples', "$Samples", '--drift-at', "$DriftAt")
& python @analysisArgs
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed' }
